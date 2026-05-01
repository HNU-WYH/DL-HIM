import numpy as np
import torch

from box import Box
import torch.nn.functional as F
from .base import NeuralOperatorBase

from src.utils.gen1d_util import generate_x_nodes_1d
from src.utils.gen2d_util import generate_xy_nodes
from src.utils.fns_blocks import MetaT1D, UNet1D, FNOMetaLambda1D, MetaT2D, FNOMetaLambda2D, QueryMetaLambda2D


# =============================================================================
# Neural Operator Based on 1-D FNS
# =============================================================================
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
        self.fns_x_nodes_np = generate_x_nodes_1d(grid_type=config.data.mesh.grid_type, num_points=self.fns_num_x_nodes)
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

    def predict_tensor(self, k_x: torch.Tensor, f_x: torch.Tensor,
                       query_points: torch.Tensor) -> torch.Tensor:
        """
        GPU-native inference: all tensors must already be on self.device.

          k_x:          (B, N_model)      float32
          f_x:          (B, N_model-2)    float32  — can differ from N_model-2 for FNS (size drives output)
          query_points: (N_query, 1)      float32  — solver inner grid coords for hard-constraint scaling

        Returns:
          u_pred: (B, N_query) on self.device, no CPU transfer.
        """
        with torch.no_grad():
            return self.forward(k_x, f_x, trunk_input=query_points)["u_pred"]

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


# =============================================================================
# Neural Operator Based on 2-D FNS
# =============================================================================
class FNS2d(NeuralOperatorBase):
    def __init__(self, config: Box):
        super().__init__(config)

        fns_config = config.training.get("fns_setting", Box({"act": "gelu", "hidden": 32}))
        act    = fns_config.get("act",    "gelu")

        # interior coordinate grid: ((Nx-2)*(Ny-2), 2)
        self.grid_nx,self.grid_ny = config.data.mesh.grid_nx, config.data.mesh.grid_ny
        self.fns_x_nodes_np, self.fns_y_nodes_np = generate_xy_nodes(grid_type=config.data.mesh.grid_type,
                                                                     num_points_x=self.grid_nx,
                                                                     num_points_y=self.grid_ny)

        xi, yi = self.fns_x_nodes_np[1:-1], self.fns_y_nodes_np[1:-1]
        XX, YY = np.meshgrid(xi, yi, indexing='ij')
        xy_flat = np.stack([XX.ravel(), YY.ravel()], axis=-1).astype(np.float32)
        self.register_buffer("fns_xy_nodes_torch",
                             torch.tensor(xy_flat, dtype=torch.float32))

        # hard constraint H(x,y) = x(1-x)*y(1-y)
        self.use_hard_cons = config.training.don_setting.hard_cons
        if self.use_hard_cons:
            self.hard_constraints = lambda x, y: x * (1.0 - x) * y * (1.0 - y)
        else:
            self.hard_constraints = None

        # normalization buffers
        self.register_buffer("k_mean",  torch.zeros(1, dtype=torch.float32))
        self.register_buffer("k_sigma", torch.zeros(1, dtype=torch.float32))

        # meta-lambda
        self.arch_type = fns_config.get("meta_lambda", "fno").lower()
        if self.arch_type == "unet":
            raise NotImplementedError(
                "MetaUNet is not supported for FNS2d. Set meta_lambda='fno' or 'sfno' in fns_setting."
            )
        elif self.arch_type == "sfno":
            self.meta_lambda = FNOMetaLambda2D(fns_config)
        else:  # "fno" (default): query-based, resolution-invariant
            self.meta_lambda = QueryMetaLambda2D(fns_config)

        # meta-T sub-networks (2D kernels: [B, Cout, Cin, 3, 3])
        self.meta1 = MetaT2D(1, 4, act=act)
        self.meta2 = MetaT2D(4, 4, act=act)
        self.meta3 = MetaT2D(4, 1, act=act)

        self.to(self.device)

    def _normalize_k(self, k):
        if torch.all(self.k_sigma == 0):
            raise ValueError("k_mean and k_sigma must be set before normalization.")
        return (k - self.k_mean) / self.k_sigma

    @staticmethod
    def odd_extension_2d(r: torch.Tensor) -> torch.Tensor:
        """r: [B, 1, Mx, My] → [B, 1, 2*(Mx+1), 2*(My+1)], four-quadrant odd extension."""
        B, C, Mx, My = r.shape
        rsym = torch.zeros(B, C, 2*(Mx+1), 2*(My+1), device=r.device, dtype=r.dtype)
        rsym[:, :, 1:Mx+1, 1:My+1] =  r
        rsym[:, :, Mx+2:,  1:My+1] = -torch.flip(r, dims=(2,))
        rsym[:, :, 1:Mx+1, My+2:]  = -torch.flip(r, dims=(3,))
        rsym[:, :, Mx+2:,  My+2:]  =  torch.flip(r, dims=(2, 3))
        return rsym

    @staticmethod
    def ik2_2d(Mx: int, My: int, device, dtype) -> torch.Tensor:
        """Spectral inverse-Laplacian weights: 1/(kx²+ky²), [1,1,Lx,Ly] cfloat."""
        Lx = 2 * (Mx + 1)
        Ly = 2 * (My + 1)
        kx = torch.arange(-Mx-1, Mx+1, device=device, dtype=dtype) * torch.pi
        ky = torch.arange(-My-1, My+1, device=device, dtype=dtype) * torch.pi
        k2 = kx.view(Lx, 1)**2 + ky.view(1, Ly)**2
        nz  = k2 != 0
        ik2 = torch.zeros_like(k2)
        ik2[nz]  = 1.0 / k2[nz]
        ik2[~nz] = 1.0
        return ik2.view(1, 1, Lx, Ly).to(torch.cfloat)

    @staticmethod
    def complex_conv2d(x: torch.Tensor, w: torch.Tensor, padding: int = 1) -> torch.Tensor:
        """
        Per-sample complex 2D convolution via grouped conv.
        x: [B, Cin, H, W] complex
        w: [B, Cout, Cin, Kh, Kw] complex
        """
        assert torch.is_complex(x) and torch.is_complex(w)
        B, Cin, H, W = x.shape
        B2, Cout, Cin2, Kh, Kw = w.shape
        assert Cin == Cin2 and B == B2

        xr, xi = x.real, x.imag
        wr, wi = w.real, w.imag

        def group_conv(inp: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            inp_g = inp.reshape(1, B * Cin, H, W)
            w_g   = weight.reshape(B * Cout, Cin, Kh, Kw)
            out   = F.conv2d(inp_g, w_g, padding=padding, groups=B)
            return out.reshape(B, Cout, out.size(-2), out.size(-1))

        yr = group_conv(xr, wr) - group_conv(xi, wi)
        yi = group_conv(xr, wi) + group_conv(xi, wr)
        return torch.complex(yr, yi)

    @staticmethod
    def transition_2d(x_hat: torch.Tensor, Ws) -> torch.Tensor:
        for W in Ws:
            x_hat = FNS2d.complex_conv2d(x_hat, W, padding=1)
        return x_hat

    def forward(self, k_x, f_x, a_mats=None, compute_du: bool = False,
                trunk_input=None, **kwargs):
        """
        k_x: [B, Nx, Ny] or [B, 1, Nx, Ny]
        f_x: [B, Mx, My] or [B, 1, Mx, My]  (interior, Mx=Nx-2, My=Ny-2)
        Returns: {"u_pred": [B, Mx, My], "du_pred": None}
        """
        if k_x.ndim == 2: k_x = k_x.unsqueeze(0)
        if f_x.ndim == 2: f_x = f_x.unsqueeze(0)
        if k_x.ndim == 3: k_x = k_x.unsqueeze(1)
        if f_x.ndim == 3: f_x = f_x.unsqueeze(1)

        k_x = k_x.to(self.device)
        f_x = f_x.to(self.device)

        Mx, My = f_x.shape[-2], f_x.shape[-1]
        k_norm = self._normalize_k(k_x)

        # odd extension
        rsym = self.odd_extension_2d(f_x)
        Lx_ext, Ly_ext = rsym.shape[-2], rsym.shape[-1]

        # band indices (same ratio as 1D: ±L/8 around centre)
        start_x = Lx_ext // 2 - Lx_ext // 8
        end_x   = Lx_ext // 2 + Lx_ext // 8 + 1
        start_y = Ly_ext // 2 - Ly_ext // 8
        end_y   = Ly_ext // 2 + Ly_ext // 8 + 1
        band_x  = end_x - start_x
        band_y  = end_y - start_y
        assert band_x == band_y, f"Non-square band ({band_x}×{band_y}); use equal grid_nx/grid_ny."

        # meta-T: per-sample 2D conv kernels
        W1 = self.meta1(k_norm).to(torch.cfloat)  # [B, 4, 1, 3, 3]
        W2 = self.meta2(k_norm).to(torch.cfloat)  # [B, 4, 4, 3, 3]
        W3 = self.meta3(k_norm).to(torch.cfloat)  # [B, 1, 4, 3, 3]

        # meta-lambda: spectral filter [B, 1, band, band]
        weights_theta = self.meta_lambda(k_norm, band_x)

        # inverse-Laplacian weights cropped to band
        ik2 = self.ik2_2d(Mx, My, rsym.device, rsym.dtype)[:, :, start_x:end_x, start_y:end_y]

        # 2D FFT pipeline
        r_hat = torch.fft.ifft2(rsym, dim=(-2, -1))
        r_hat = torch.fft.fftshift(r_hat, dim=(-2, -1))[:, :, start_x:end_x, start_y:end_y]

        r_hat   = self.transition_2d(r_hat, [W1, W2, W3])
        out_hat = r_hat * weights_theta * ik2
        out_hat = self.transition_2d(out_hat, [
            W3.transpose(1, 2).flip(-1).flip(-2).conj(),
            W2.transpose(1, 2).flip(-1).flip(-2).conj(),
            W1.transpose(1, 2).flip(-1).flip(-2).conj(),
        ])

        # zero-pad back to extended size, then back-transform
        out_hat = F.pad(out_hat, (start_y, Ly_ext - end_y, start_x, Lx_ext - end_x))
        out_hat = torch.fft.ifftshift(out_hat, dim=(-2, -1))
        e_full  = torch.fft.fft2(out_hat, dim=(-2, -1)).real
        u_pred  = e_full[:, :, 1:Mx+1, 1:My+1].squeeze(1)  # [B, Mx, My]

        # hard constraint H(x,y) = x(1-x)*y(1-y)
        if self.hard_constraints is not None:
            if trunk_input is None:
                trunk_input = self.fns_xy_nodes_torch
            trunk_input = trunk_input.to(self.device)          # ((Mx*My), 2)
            xy  = trunk_input.view(Mx, My, 2)
            H_xy = self.hard_constraints(xy[..., 0], xy[..., 1])
            u_pred = u_pred * H_xy

        return {"u_pred": u_pred, "du_pred": None}

    def predict(self, k_x: np.ndarray, f_x: np.ndarray,
                x_k=None, x_f=None, **kwargs):
        """
        k_x: (B, Nx, Ny) or (Nx, Ny)
        f_x: (B, Mx, My) or (Mx, My)
        Returns: (B, Mx, My) numpy array
        """
        if isinstance(k_x, np.ndarray):
            k_x = torch.as_tensor(k_x, dtype=torch.float32, device=self.device)
        if isinstance(f_x, np.ndarray):
            f_x = torch.as_tensor(f_x, dtype=torch.float32, device=self.device)

        if k_x.dim() == 2: k_x = k_x.unsqueeze(0)
        if f_x.dim() == 2: f_x = f_x.unsqueeze(0)

        if k_x.dim() != 3 or f_x.dim() != 3:
            raise ValueError(f"Expected 3D tensors (B, N, N). Got k_x {k_x.shape}, f_x {f_x.shape}")
        if k_x.shape[0] != f_x.shape[0]:
            raise ValueError(f"Batch size mismatch: {k_x.shape[0]} vs {f_x.shape[0]}")

        batch_size = f_x.shape[0]
        k_x, f_x, query_points = self._preprocess_input(k_x, x_k, f_x, x_f, batch_size)

        query_points = torch.as_tensor(
            query_points.detach().cpu().numpy() if isinstance(query_points, torch.Tensor) else query_points,
            dtype=torch.float32, device=self.device,
        )

        self.eval()
        with torch.no_grad():
            return self.forward(k_x, f_x, trunk_input=query_points, **kwargs)["u_pred"].cpu().numpy()

    def predict_tensor(self, k_x: torch.Tensor, f_x: torch.Tensor,
                       query_points: torch.Tensor) -> torch.Tensor:
        """
        GPU-native inference; all tensors must be on self.device.
          k_x:          (B, Nx, Ny)        float32
          f_x:          (B, Mx, My)        float32
          query_points: ((Mx*My), 2)       float32
        Returns: (B, Mx, My) on self.device
        """
        with torch.no_grad():
            return self.forward(k_x, f_x, trunk_input=query_points)["u_pred"]

    def _preprocess_input(self, k_x, x_k, f_x, x_f, batch_size=None):
        if batch_size is None:
            batch_size = f_x.shape[0]

        # k_x: interpolate to model grid if needed
        if k_x.shape[-2] != self.grid_nx or k_x.shape[-1] != self.grid_ny:
            from scipy.interpolate import RegularGridInterpolator
            if isinstance(k_x, torch.Tensor):
                k_x = k_x.detach().cpu().numpy()
            if x_k is None:
                x_k = (np.linspace(0, 1, k_x.shape[-2]),
                        np.linspace(0, 1, k_x.shape[-1]))
            XX, YY = np.meshgrid(self.fns_x_nodes_np, self.fns_y_nodes_np, indexing='ij')
            xy_tgt  = np.stack([XX.ravel(), YY.ravel()], axis=-1)
            k_list  = []
            for i in range(batch_size):
                fn = RegularGridInterpolator(x_k, k_x[i], method='linear')
                k_list.append(fn(xy_tgt).reshape(self.grid_nx, self.grid_ny))
            k_x = torch.as_tensor(np.stack(k_list), dtype=torch.float32, device=self.device)
        else:
            if isinstance(k_x, torch.Tensor):
                k_x = k_x.to(dtype=torch.float32, device=self.device)
            else:
                k_x = torch.as_tensor(k_x, dtype=torch.float32, device=self.device)

        # f_x: keep as-is (interior only, size must match model grid)
        if isinstance(f_x, torch.Tensor):
            f_x = f_x.to(dtype=torch.float32, device=self.device)
        else:
            f_x = torch.as_tensor(f_x, dtype=torch.float32, device=self.device)

        x_f = self.fns_xy_nodes_torch   # ((Mx*My), 2)
        return k_x, f_x, x_f
