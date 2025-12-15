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

        # Grid Properties
        # The resolution of meta-λ in FNS is fixed
        self.fns_num_x_nodes = config.data.mesh.grid_num
        self.fns_x_nodes_np = generate_x_nodes(grid_type=config.data.mesh.grid_type, num_points=self.fns_num_x_nodes)
        self.register_buffer("fns_x_nodes_torch",
                             torch.tensor(self.fns_x_nodes_np[1:-1, None], dtype=torch.float32, device= self.device))

        # Normalization buffers
        self.register_buffer("k_mean", torch.zeros(1, dtype=torch.float32, device= self.device))
        self.register_buffer("k_sigma", torch.zeros(1, dtype=torch.float32, device= self.device))

        # Initialize Sub-networks (Meta-T)
        self.meta1 = MetaT1D(1, 4, act=act)
        self.meta2 = MetaT1D(4, 4, act=act)
        self.meta3 = MetaT1D(4, 1, act=act)

        # Initialize Sub-networks (Meta-λ)
        self.meta_lambda = UNet1D(act=act, hidden = 32)

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
    def ik2_id(self, M: int, device, dtype) -> torch.Tensor:
        L = 2 * (M + 1)

        # 对于偶数的DFT, 一般习惯 [-L/2, L/2 - 1], 需要周期性, 因此不能对称
        # 负频率: -1 ~ -L/2 (一共L/2); 正频率: 0 ~ L/2-1 (一共L/2)
        # (-L/2)mod(L) = (L/2)%L 更好可以接到 (L/2-1)后面
        k = torch.arange(-M-1, M+1, device=device, dtype=dtype)

        ik2 = 1.0 / (k * k + 1e-18)
        ik2[M + 1] = 1.0
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

    def forward(self, k_x, f_x, a_mats = None, **kwargs):
        """
        :param k_x: [B, N_grid-2].
        :param f_x: [B, N_grid-2] (Interior points).
        :param a_mats: Not used in FNS forward pass, kept for interface compatibility.
        :return: dict with "u_pred" (the correction).
        """
        # Input shape handling to match meta-net requirements
        if k_x.ndim == 1: k_x = k_x[None, :]      # [B, N]
        if f_x.ndim == 1: f_x = f_x[None, :]      # [B, N]

        if k_x.ndim == 2: k_x = k_x.unsqueeze(1)  # [B, 1, N]
        if f_x.ndim == 2: f_x = f_x.unsqueeze(1)  # [B, 1, M]

        k_x = k_x.to(self.device)
        f_x = f_x.to(self.device)

        # 1. Normalize Coefficient k (Conditioning signal)
        k_norm = self._normalize_k(k_x)

        # 2. Prepare Residual (Odd Extension)
        # f_x represents the residual 'r' on interior points
        rsym = self.odd_extension_1d(f_x)  # [B, 1, L_ext]
        B, _, L_ext = rsym.shape
        M = f_x.shape[-1]

        # 3. Generate Operators via Meta-Learning
        # Meta-nets take normalized k as input
        W1 = self.meta1(k_norm).to(torch.cfloat)  # Transition 1, [B, k]
        W2 = self.meta2(k_norm).to(torch.cfloat)  # Transition 2, [B, k]
        W3 = self.meta3(k_norm).to(torch.cfloat)  # Transition 3, [B, k]

        # Generate spectral weights theta matched to the extended length
        weights_theta = self.meta_lambda(k_norm, L_ext)

        # 4. FFT Pipeline
        # 4.1 Define low frequency indices and Physical Prior for Diffusion
        start, end = L_ext//2-L_ext//4, L_ext//2+L_ext//4+1
        ik2 = self.ik2_id(M, rsym.device, rsym.dtype)[:,:,start:end]

        # 4.2 IFFT the residual
        r_hat = torch.fft.ifft(rsym, dim=-1)                             # FFT, freq = 0 is in idx = 0
        r_hat = torch.fft.fftshift(r_hat, dim=-1)[:,:,start:end]         # move to center, and use center only

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
        u_pred = e_full[:, :, :M]  # [B, 1, M]
        u_pred = u_pred.squeeze(1)


        # TODO: 这里需要保持兼容 | 并且修改一下梯度生成的逻辑 (和deeponet一起)
        # TODO: deeponet计算residual的梯度那个逻辑是有误的
        # Gradient computation (Numerical or Autograd) could be added here if needed
        # For now, returning None unless explicitly implemented similar to DeepONet
        # Optional: Compute derivatives/residuals if required by trainer (e.g. for loss)
        res, du_pred, dres = None, None, None

        # FNS usually doesn't need to compute its own 'res' inside forward
        # because f_x IS the residual, but for consistency with trainer's compute_loss:
        if self.require_res and a_mats is not None:
            # Be careful: In dynamic training, f_x is residual, so u_pred is correction.
            # This logical branch might need adjustment based on specific trainer usage,
            # but keeping standard interface implementation:
            res = (f_x.squeeze(1)[..., None] - a_mats @ u_pred[..., None]).squeeze(-1)

        return {
            "u_pred": u_pred,
            "res": res,
            "du_pred": du_pred,
            "dres": dres
        }

    def predict(self, k_x: np.ndarray, f_x: np.ndarray,
                x_k: np.ndarray = None, x_f: np.ndarray = None,
                **kwargs):
        pass


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
            nn.Conv2d(8, 16, 5, padding=2),
            getActivationFunction(act),
            nn.Conv2d(16, 32, 5, padding=2),
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

    def __init__(self, act="gelu", base=32):
        super().__init__()
        self.act = act

        self.in_conv = nn.Sequential(
            nn.Conv1d(1, base, 5, padding=2),
            getActivationFunction(act),
            ResBlock1D(base, act),
        )

        self.down1 = nn.Sequential(
            nn.Conv1d(base, base * 2, 5, stride=2, padding=2),
            nn.BatchNorm1d(base * 2),
            getActivationFunction(act),
        )

        self.down2 = nn.Sequential(
            nn.Conv1d(base * 2, base * 4, 5, stride=2, padding=2),
            nn.BatchNorm1d(base * 4),
            getActivationFunction(act),
        )

        self.mid = nn.Sequential(
            ResBlock1D(base * 4, act),
            ResBlock1D(base * 4, act),
        )

        self.up2 = nn.Sequential(
            getActivationFunction(act),
            nn.ConvTranspose1d(base * 4, base * 2, 4, stride=2, padding=1),
            nn.BatchNorm1d(base * 2),
        )
        self.up1 = nn.Sequential(
            getActivationFunction(act),
            nn.ConvTranspose1d(base * 2, base, 4, stride=2, padding=1),
            nn.BatchNorm1d(base),
        )

        self.out_conv = nn.Conv1d(base, 1, 1)

    def forward(self, coef_signal: torch.Tensor, Lfft: int) -> torch.Tensor:
        """
        coef_signal: [B, 1, Lc] real
        Lfft: Target length in frequency domain
        """
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
