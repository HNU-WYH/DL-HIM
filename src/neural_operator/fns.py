import numpy as np
import torch

from box import Box
from torch import nn
import torch.nn.functional as F
from .base import NeuralOperatorBase

from src.utils.gen1d_util import generate_x_nodes


class FNS1d(NeuralOperatorBase):
    def __init__(self, config: Box):
        super().__init__(config)

        # FNS Configuration
        fns_config = config.training.get("fns_setting", Box({"act": "gelu", "hidden": 32}))
        act = fns_config.get("act", "gelu")
        hidden = fns_config.get("hidden", 32)

        # Architecture Type
        self.arch_type = fns_config.get("meta_lambda", "unet").lower()
        if self.arch_type == "fno":
            modes = fns_config.get("fno_modes", 16)
            lfreq = fns_config.get("fno_lfreq", 4)
            self.meta_lambda = FNOMetaLambda1D(act=act, hidden=hidden, modes=modes, Lfreq=lfreq)
        else:
            # Default to UNet1D based on their implement on Diffusion equations
            self.meta_lambda = UNet1D(act=act, hidden=hidden)

        # Grid Properties
        # The resolution of meta-λ in FNS is fixed
        self.fns_num_x_nodes = config.data.mesh.grid_num
        self.fns_x_nodes_np = generate_x_nodes(grid_type=config.data.mesh.grid_type, num_points=self.fns_num_x_nodes)
        self.register_buffer("fns_x_nodes_torch",
                             torch.tensor(self.fns_x_nodes_np[1:-1, None], dtype=torch.float32, device=self.device))

        # define a hard constraint H(x) = x * (x - 1.0)
        # This forces the solution to be 0 at x=0 and x=1
        self.use_hard_cons = config.training.don_setting.hard_cons

        if self.use_hard_cons:
            self.hard_constraints = lambda x: x * (1.0 - x)
        else:
            self.hard_constraints = None

        # Normalization buffers
        self.register_buffer("k_mean", torch.zeros(1, dtype=torch.float32, device= self.device))
        self.register_buffer("k_sigma", torch.zeros(1, dtype=torch.float32, device= self.device))

        # Initialize Sub-networks (Meta-T)
        self.meta1 = MetaT1D(1, 4, act=act)
        self.meta2 = MetaT1D(4, 4, act=act)
        self.meta3 = MetaT1D(4, 1, act=act)

        self.to(self.device)

    def _normalize_k(self, k):
        """
        Normalize the parameter function k(x)
        """
        if torch.all(self.k_sigma == 0):
            raise ValueError("k_mean and k_sigma must be set before normalization.")

        k_norm = (k - self.k_mean) / self.k_sigma
        return k_norm

    @staticmethod
    def odd_extension(r: torch.Tensor) -> torch.Tensor:
        """
        Convolution in torch has the shape: [Batch, channel, length]
        """
        B, C, M = r.shape
        r_ext = torch.zeros(B, C, 2 * (M + 1), device=r.device, dtype=r.dtype)

        r_ext[:, :, 1: M+1] = r
        r_ext[:, :, M+2:] = -torch.flip(r, dims=(-1, ))

        return r_ext

    @staticmethod
    def ik2_id(M: int, device, dtype) -> torch.Tensor:
        L = 2 * (M + 1)

        k = torch.arange(-M-1, M+1, device=device, dtype=dtype) * torch.pi
        nz = k != 0

        ik2 = torch.zeros_like(k)
        ik2[nz] = 1.0 / (k[nz]**2)
        ik2[~nz] = 1.0
        return ik2.view(1,1,L).to(torch.cfloat)

    @staticmethod
    def transition(x_hat: torch.Tensor, Ws):
        """
        Apply sequence of convolution kernel in frequency domain
        """
        for W in Ws:
            x_hat = FNS1d.complex_conv1d(x_hat, W, padding=1)
        return x_hat

    @staticmethod
    def complex_conv1d(x, w, padding=1):
        """
            Per-sample complex convolution via grouped conv (Every sample has its own kernel).
            x: [B, Cin, L] complex
            w: [B, Cout, Cin, K] complex
            return: [B, Cout, L] complex
        """
        assert torch.is_complex(x) and torch.is_complex(w)
        B, Cin, L = x.shape
        B2, Cout, Cin2, K = w.shape
        assert Cin2 == Cin
        assert B == B2

        xr, xi = x.real, x.imag  # [B, Cin, L]
        wr, wi = w.real, w.imag  # [B, Cout, Cin, K]

        def group_conv(inp: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            # Grouped convolution to handle per-sample weights
            inp_g = inp.reshape(1, B * Cin, L)
            w_g = weight.reshape(B * Cout, Cin, K)
            out = F.conv1d(inp_g, w_g, padding=padding, groups=B)
            return out.reshape(B, Cout, out.size(-1))

        yr = group_conv(xr, wr) - group_conv(xi, wi)
        yi = group_conv(xr, wi) + group_conv(xi, wr)
        return torch.complex(yr, yi)

    def forward(self, k_x, f_x, a_mats=None, compute_du: bool = False, trunk_input=None, **kwargs):
        """
        :param k_x: [B, N_grid].
        :param f_x: [B, N_grid-2] (Interior points).
        :param a_mats: Not used in FNS forward pass, kept for interface compatibility.
        :param compute_du: whether to compute the gradient of solution inside the model
        :param trunk_input: e.g. shape = [query points, n dim] (query points = don grid points - 2)
        :return: dict with "u_pred" (the correction).
        """
        # Input shape handling to match meta-net requirements
        if k_x.ndim == 1: k_x = k_x[None, :]      # [B, N]
        if f_x.ndim == 1: f_x = f_x[None, :]      # [B, N-2]

        if k_x.ndim == 2: k_x = k_x.unsqueeze(1)  # [B, 1, N]
        if f_x.ndim == 2: f_x = f_x.unsqueeze(1)  # [B, 1, N-2]

        k_x = k_x.to(self.device)
        f_x = f_x.to(self.device)

        # 1. Normalize Coefficient k (Conditioning signal)
        k_norm = self._normalize_k(k_x)

        # 2. Prepare Residual (Odd Extension)
        # f_x represents the residual 'r' on interior points
        rsym = self.odd_extension(f_x)  # [B, 1, L_ext]
        B, _, L_ext = rsym.shape
        M = f_x.shape[-1]

        # 3. Generate Operators via Meta-Learning
        # Meta-nets take normalized k as input
        W1 = self.meta1(k_norm).to(torch.cfloat)  # Transition 1, [B, k]
        W2 = self.meta2(k_norm).to(torch.cfloat)  # Transition 2, [B, k]
        W3 = self.meta3(k_norm).to(torch.cfloat)  # Transition 3, [B, k]

        # Generate spectral weights theta matched to the extended length
        band_len = 2 * (L_ext // 8) + 1
        start = L_ext // 2 - L_ext // 8
        end = L_ext // 2 + L_ext // 8 + 1

        if self.arch_type == "fno":
            # FNO
            weights_theta = self.meta_lambda(k_norm, band_len)
        else:
            # UNet
            weights_theta = self.meta_lambda(k_norm, L_ext)
            weights_theta = weights_theta[:, :, start:end]

        # 4. FFT Pipeline
        # 4.1 Define low frequency indices and Physical Prior for Diffusion
        ik2 = self.ik2_id(M, rsym.device, rsym.dtype)[:, :, start:end]

        # 4.2 IFFT the residual
        r_hat = torch.fft.ifft(rsym, dim=-1)                             # FFT, freq = 0 is in idx = 0
        r_hat = torch.fft.fftshift(r_hat, dim=-1)[:, :, start:end]         # move to center, and use center only

        # 4.3 Handling the eigenvalues
        r_hat = self.transition(r_hat, [W1, W2, W3])
        out_hat = r_hat * weights_theta * ik2
        out_hat = self.transition(
            x_hat=out_hat, Ws=[W3.transpose(1, 2).flip(-1).conj(),
                               W2.transpose(1, 2).flip(-1).conj(),
                               W1.transpose(1, 2).flip(-1).conj()],
        )

        # 4.4 Zero padding to original size
        out_hat = F.pad(out_hat, (start, L_ext - end), mode='constant', value=0)

        # 4.5 FFT to the error
        out_hat = torch.fft.ifftshift(out_hat, dim=-1)
        e_full = torch.fft.fft(out_hat, dim=-1).real
        u_pred = e_full[:, :, 1:M+1]  # [B, 1, M]
        u_pred = u_pred.squeeze(1)

        # Apply hard constraints
        if self.hard_constraints is not None:
            if trunk_input is None: trunk_input = self.fns_x_nodes_torch
            trunk_input = trunk_input.to(self.device)  # (num_x-2, 1)
            H_x = self.hard_constraints(trunk_input).squeeze(-1)  # (num_x-2)
            u_pred = u_pred * H_x  # (B, num_x-2) * (num_x-2) -> (B, num_x-2)
        else:
            u_pred = u_pred

        return {
            "u_pred": u_pred,
            "du_pred": None,
        }

    def predict(self, k_x: np.ndarray, f_x: np.ndarray,
                x_k: np.ndarray = None, x_f: np.ndarray = None,
                **kwargs):
        """
            Minimal prediction entry point: calls forward() directly.

            Required shapes (interior-only):
              - k_x: (B, N_full)
              - f_x: (B, N_int)

            Dtypes:
              - k_x: float32/float64 (internally you may normalize / cast)
              - f_x: float32/float64

            Returns:
              - same dict as forward(), e.g. {"u_pred": (B, N_int), "du_pred": None, ...}
        """
        if isinstance(k_x, np.ndarray):
            k_x = torch.as_tensor(k_x, dtype=torch.float32, device=self.device)
        if isinstance(f_x, np.ndarray):
            f_x = torch.as_tensor(f_x, dtype= torch.float32, device=self.device)

        # Accept (N_int,) and promote to (1, N_int)
        if k_x.dim() == 1:
            k_x = k_x.unsqueeze(0)
        if f_x.dim() == 1:
            f_x = f_x.unsqueeze(0)

        if k_x.dim() != 2 or f_x.dim() != 2:
            raise ValueError(f"Expected 2D tensors (B, N_int). Got k_x {k_x.shape}, f_x {f_x.shape}")

        if k_x.shape[0] != f_x.shape[0]:
            raise ValueError(f"k_x and f_x must have the same batch size. Got {k_x.shape[0]} vs {f_x.shape[0]}")

        # Interpolate if needed
        batch_size = f_x.shape[0]
        k_x, f_x, query_points = self._preprocess_input(k_x, x_k, f_x, x_f, batch_size)  # (D,), (D-2,), (G-2, 1)

        # If query_points is None, use the internal x_nodes for predictions

        if query_points.ndim == 1:
            query_points = query_points[:, None]
        query_points = torch.as_tensor(query_points, dtype=torch.float32, device=self.device)  # (G-2, 1)

        self.eval()
        with torch.no_grad():
            return self.forward(k_x, f_x, trunk_input=query_points, **kwargs)["u_pred"].squeeze().cpu().detach().numpy()

    def _preprocess_input(self, k_x, x_k, f_x, x_f, batch_size=None):
        if batch_size is None:
            batch_size = f_x.shape[0]

        if k_x.shape[-1] != self.fns_num_x_nodes:
            if isinstance(k_x, torch.Tensor):
                k_x = k_x.detach().cpu().numpy()                                            # (G, )

            if x_k is None:
                x_k = np.linspace(0, 1, k_x.shape[-1])                                      # (G, )

            k_x_interp = np.stack([np.interp(self.fns_x_nodes_np, x_k, k_x[i]) for i in range(batch_size)], axis=0)
            k_x = torch.as_tensor(k_x_interp, dtype=torch.float32, device=self.device)
        else:
            k_x = torch.as_tensor(k_x, dtype=torch.float32, device=self.device)                    # (D,)

        if f_x.shape[-1] != self.fns_num_x_nodes - 2:
            if isinstance(f_x, torch.Tensor):
                f_x = f_x.detach().cpu().numpy()                                                  # (G - 2,)

            if x_f is None:
                x_f = np.linspace(0, 1, f_x.shape[-1] + 2)[1:-1, None]                                  # (G - 2,)

            f_x = torch.as_tensor(f_x, dtype=torch.float32, device=self.device)
            x_f = torch.as_tensor(x_f, dtype=torch.float32, device=self.device)

        else:
            f_x = torch.as_tensor(f_x, dtype=torch.float32, device=self.device)             # (D - 2,)
            x_f = self.fns_x_nodes_torch

        return k_x, f_x, x_f     # (D,), (D - 2,), (G - 2, 1)


def getActivationFunction(act: str):
    act = act.lower()
    if act == 'relu':
        return nn.ReLU()
    if act == "gelu":
        return nn.GELU()
    if act == 'tanh':
        return nn.Tanh()
    if act == "elu":
        return nn.ELU()
    if act == "leakyrelu":
        return nn.LeakyReLU()
    raise NotImplementedError(f"Unknown activation function: {act}")


class MetaT1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, act: str = "gelu", hidden: int = 64, pool: int = 8):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.cnn = nn.Sequential(
            nn.Conv1d(1, 8, 5, padding=2),
            getActivationFunction(act),
            nn.Conv1d(8, 16, 5, padding=2),
            getActivationFunction(act),
            nn.Conv1d(16, 32, 5, padding=2),
            getActivationFunction(act),
            nn.AdaptiveAvgPool1d(pool),
        )

        # Output dim for the complex 1D kernel [Out_channel, In_chanel 3]
        output_dim = in_channels * out_channels * 3
        self.fnn = nn.Sequential(
            nn.Linear(32 * pool, hidden),
            getActivationFunction(act),
            nn.Linear(hidden, hidden),
            getActivationFunction(act),
            nn.Linear(hidden, 2 * output_dim)
        )

    def forward(self, x):
        """
        x: [B, 1, L]
        """
        z = self.cnn(x).flatten(1)             # [B, 32, pool] -> [B, 32 * pool]
        p = self.fnn(z)                        # [B, 2 * output_dim]

        output_dim = self.in_channels * self.out_channels * 3
        p_real = p[:, :output_dim].view(-1, self.out_channels, self.in_channels, 3)
        p_imag = p[:, output_dim:].view(-1, self.out_channels, self.in_channels, 3)
        return torch.complex(p_real, p_imag)


class ResBlock1D(nn.Module):
    def __init__(self, c, act):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(c, c, 3, padding=1),
            nn.BatchNorm1d(c),
            getActivationFunction(act),
            nn.Conv1d(c, c, 3, padding=1),
        )
        self.act = nn.Sequential(nn.BatchNorm1d(c), getActivationFunction(act))

    def forward(self, x):
        return self.act(self.net(x) + x)


class UNet1D(nn.Module):
    """
    U-Net based Meta-Network to generate spectral filter weights \theta.
    """

    def __init__(self, act="gelu", hidden=32):
        super().__init__()
        self.act = act

        self.in_conv = nn.Sequential(
            nn.Conv1d(1, hidden, 5, padding=2),
            getActivationFunction(act),
            ResBlock1D(hidden, act),
        )

        self.down1 = nn.Sequential(
            nn.Conv1d(hidden, hidden * 2, 5, stride=2, padding=2),
            nn.BatchNorm1d(hidden * 2),
            getActivationFunction(act),
        )

        self.down2 = nn.Sequential(
            nn.Conv1d(hidden * 2, hidden * 4, 5, stride=2, padding=2),
            nn.BatchNorm1d(hidden * 4),
            getActivationFunction(act),
        )

        self.mid = nn.Sequential(
            ResBlock1D(hidden * 4, act),
            ResBlock1D(hidden * 4, act),
        )

        self.up2 = nn.Sequential(
            getActivationFunction(act),
            nn.ConvTranspose1d(hidden * 4, hidden * 2, 4, stride=2, padding=1),
            nn.BatchNorm1d(hidden * 2),
        )
        self.up1 = nn.Sequential(
            getActivationFunction(act),
            nn.ConvTranspose1d(hidden * 2, hidden, 4, stride=2, padding=1),
            nn.BatchNorm1d(hidden),
        )

        self.out_conv = nn.Conv1d(hidden, 1, 1)

    def forward(self, coef_signal: torch.Tensor, Lfft: int) -> torch.Tensor:
        """
            coef_signal: [B, 1, Lc] real
            Lfft: Target length in frequency domain
        """
        # mapping from L to L_ext
        x0 = self.in_conv(coef_signal)
        x0 = F.interpolate(x0, size=Lfft, mode='linear', align_corners=True)

        x1 = self.down1(x0)
        x2 = self.down2(x1)
        xm = self.mid(x2)

        y2 = self.up2(xm)
        # Pad or crop to match skip connection size if needed
        y2 = y2[..., :x1.size(-1)] + x1

        y1 = self.up1(y2)
        y1 = y1[..., :x0.size(-1)] + x0

        out = self.out_conv(y1)  # [B, 1, Lfft]

        # Convert physical prediction to frequency domain weights (complex)
        w = torch.fft.fft(out, dim=-1) / (Lfft ** 0.5)
        w = torch.fft.fftshift(w, dim=-1)
        return w.to(torch.cfloat)


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        scale = 1.0 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batchsize = x.shape[0]

        x_ft = torch.fft.rfft(x, dim=-1)

        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1) // 2 + 1,
                             device=x.device, dtype=torch.cfloat)

        k = min(self.modes, x_ft.shape[-1])
        out_ft[:, :, :k] = torch.einsum("bik,iok->bok", x_ft[:, :, :k], self.weights[:, :, :k])

        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


class FNOMetaLambda1D(nn.Module):
    """
        Meta-Network based on FNO。
    """
    def __init__(self, act="gelu", hidden=32, modes=16, Lfreq=4):
        super().__init__()
        self.act_fn = getActivationFunction(act)
        self.hidden = hidden
        self.Lfreq = Lfreq

        # 1. Lifting
        self.lifting = nn.Conv1d(1, hidden, 1)

        # 2. Fourier Layers
        self.spec1 = SpectralConv1d(hidden, hidden, modes)
        self.spec2 = SpectralConv1d(hidden, hidden, modes)

        # 3. Pointwise paths
        self.w1 = nn.Conv1d(hidden, hidden, 1)
        self.w2 = nn.Conv1d(hidden, hidden, 1)

        # 4. Frequency Query Head
        self.query_head = nn.Sequential(
            nn.Linear(hidden + (2 * Lfreq + 1), hidden),
            self.act_fn,
            nn.Linear(hidden, 2)
        )

    def _get_freq_embedding(self, Lfft: int, device: torch.device) -> torch.Tensor:
        t = torch.linspace(0.0, 1.0, Lfft, device=device).unsqueeze(-1)
        feats = [t]
        for k in range(self.Lfreq):
            w = (2.0 ** k) * 2.0 * torch.pi
            feats.append(torch.sin(w * t))
            feats.append(torch.cos(w * t))
        return torch.cat(feats, dim=-1)  # [Lfft, 2*Lfreq + 1]

    def forward(self, coef_signal: torch.Tensor, Lfft: int) -> torch.Tensor:
        B, _, Lc = coef_signal.shape

        # Lifting
        x = self.lifting(coef_signal)

        # FNO Block 1
        x1 = self.spec1(x) + self.w1(x)
        x = self.act_fn(x1)

        # FNO Block 2
        x2 = self.spec2(x) + self.w2(x)
        x = self.act_fn(x2)

        # Pooling
        z = torch.mean(x, dim=-1)  # [B, hidden]

        # phi: [Lfft, 2*Lfreq+1]
        phi = self._get_freq_embedding(Lfft, coef_signal.device)

        # z_rep: [B, Lfft, hidden], phi_rep: [B, Lfft, F_dim]
        z_rep = z.unsqueeze(1).expand(-1, Lfft, -1)
        phi_rep = phi.unsqueeze(0).expand(B, -1, -1)

        query_input = torch.cat([z_rep, phi_rep], dim=-1)

        out = self.query_head(query_input)  # [B, Lfft, 2]
        w = torch.complex(out[..., 0], out[..., 1])

        # Reshape to [B, 1, Lfft]
        return w.unsqueeze(1).to(torch.cfloat)