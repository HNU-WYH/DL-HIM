import torch
from torch import nn
import torch.nn.functional as F


# =============================================================================
# Shared Utilities
# =============================================================================

def getActivationFunction(act: str):
    act = act.lower()
    if act == 'relu':      return nn.ReLU()
    if act == 'gelu':      return nn.GELU()
    if act == 'tanh':      return nn.Tanh()
    if act == 'elu':       return nn.ELU()
    if act == 'leakyrelu': return nn.LeakyReLU()
    raise NotImplementedError(f"Unknown activation function: {act}")


# =============================================================================
# 1D Neural Network Components
# =============================================================================

# ----- 1. Residual Block (Post-act) -----
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


# ----- 2. Global Conv (Mul on Spectral) -----
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.modes = modes
        scale = 1.0 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batchsize = x.shape[0]
        x_ft   = torch.fft.rfft(x, dim=-1)
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1) // 2 + 1,
                             device=x.device, dtype=torch.cfloat)
        k = min(self.modes, x_ft.shape[-1])
        out_ft[:, :, :k] = torch.einsum("bik,iok->bok", x_ft[:, :, :k], self.weights[:, :, :k])
        return torch.fft.irfft(out_ft, n=x.size(-1))


# =============================================================================
# 1D Meta Networks
# =============================================================================

# -----------1. Meta-T Network (1D) ------------
class MetaT1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 act: str = "gelu", hidden: int = 64, pool: int = 8):
        super().__init__()
        self.in_channels  = in_channels
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
        output_dim = in_channels * out_channels * 3
        self.fnn = nn.Sequential(
            nn.Linear(32 * pool, hidden),
            getActivationFunction(act),
            nn.Linear(hidden, hidden),
            getActivationFunction(act),
            nn.Linear(hidden, 2 * output_dim),
        )

    def forward(self, x):
        """x: [B, 1, L]"""
        z = self.cnn(x).flatten(1)
        p = self.fnn(z)
        output_dim = self.in_channels * self.out_channels * 3
        p_real = p[:, :output_dim].view(-1, self.out_channels, self.in_channels, 3)
        p_imag = p[:, output_dim:].view(-1, self.out_channels, self.in_channels, 3)
        return torch.complex(p_real, p_imag)


# -----------2. Meta-λ Network (1D) ------------

# -----------2.1 U-Net (Meta-λ, 1D) ------------
class UNet1D(nn.Module):
    def __init__(self, act="gelu", hidden=32):
        super().__init__()
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
        """coef_signal: [B, 1, Lc] → [B, 1, Lfft] cfloat"""
        x0 = self.in_conv(coef_signal)
        x0 = F.interpolate(x0, size=Lfft, mode='linear', align_corners=True)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        xm = self.mid(x2)
        y2 = self.up2(xm)[..., :x1.size(-1)] + x1
        y1 = self.up1(y2)[..., :x0.size(-1)] + x0
        out = self.out_conv(y1)
        w = torch.fft.fft(out, dim=-1) / (Lfft ** 0.5)
        w = torch.fft.fftshift(w, dim=-1)
        return w.to(torch.cfloat)


# -----------2.2 FNO (Meta-λ, 1D) ------------
class FNOMetaLambda1D(nn.Module):
    def __init__(self, act="gelu", hidden=32, modes=16, Lfreq=4):
        super().__init__()
        self.act_fn  = getActivationFunction(act)
        self.hidden  = hidden
        self.Lfreq   = Lfreq
        self.lifting = nn.Conv1d(1, hidden, 1)
        self.spec1   = SpectralConv1d(hidden, hidden, modes)
        self.spec2   = SpectralConv1d(hidden, hidden, modes)
        self.w1      = nn.Conv1d(hidden, hidden, 1)
        self.w2      = nn.Conv1d(hidden, hidden, 1)
        self.query_head = nn.Sequential(
            nn.Linear(hidden + (2 * Lfreq + 1), hidden),
            self.act_fn,
            nn.Linear(hidden, 2),
        )

    def _get_freq_embedding(self, Lfft: int, device) -> torch.Tensor:
        t = torch.linspace(0.0, 1.0, Lfft, device=device).unsqueeze(-1)
        feats = [t]
        for k in range(self.Lfreq):
            w = (2.0 ** k) * 2.0 * torch.pi
            feats.append(torch.sin(w * t))
            feats.append(torch.cos(w * t))
        return torch.cat(feats, dim=-1)  # [Lfft, 2*Lfreq+1]

    def forward(self, coef_signal: torch.Tensor, Lfft: int) -> torch.Tensor:
        """coef_signal: [B, 1, Lc] → [B, 1, Lfft] cfloat"""
        B = coef_signal.shape[0]
        x = self.lifting(coef_signal)
        x = self.act_fn(self.spec1(x) + self.w1(x))
        x = self.act_fn(self.spec2(x) + self.w2(x))
        z   = torch.mean(x, dim=-1)
        phi = self._get_freq_embedding(Lfft, coef_signal.device)
        query_input = torch.cat([
            z.unsqueeze(1).expand(-1, Lfft, -1),
            phi.unsqueeze(0).expand(B, -1, -1),
        ], dim=-1)
        out = self.query_head(query_input)
        w = torch.complex(out[..., 0], out[..., 1])
        return w.unsqueeze(1).to(torch.cfloat)


# =============================================================================
# 2D Neural Network Components
# =============================================================================

# ----- 1. Drop Path -----
class _DropPath(nn.Module):
    """Stochastic depth (Huang et al. 2016), inline to avoid timm dependency."""
    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0. or not self.training:
            return x
        keep  = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask  = torch.rand(shape, dtype=x.dtype, device=x.device).floor_() + keep
        return x * mask / keep


# ----- 2. Layer Norm (channels_first) -----
class _LayerNorm2D(nn.Module):
    """LayerNorm supporting channels_first layout [B, C, H, W]."""
    def __init__(self, normalized_shape: int, eps: float = 1e-6,
                 data_format: str = "channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias   = nn.Parameter(torch.zeros(normalized_shape))
        self.eps    = eps
        self.data_format = data_format
        if data_format not in ("channels_last", "channels_first"):
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


# ----- 3. Residual Block (Post-act, 2D) -----
class ResBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, act):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            getActivationFunction(act),
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=1),
        )
        self.activation = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            getActivationFunction(act),
        )

    def forward(self, x):
        return self.activation(self.layers(x) + x)


# ----- 4. Fourier Conv 2D -----
class _FourierConv2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, wn1: int, wn2: int):
        super().__init__()
        self.out_ch      = out_ch
        self.wn1, self.wn2 = wn1, wn2
        scale = 1.0 / (in_ch * out_ch)
        self.w1 = nn.Parameter(torch.view_as_real(
            torch.randn(in_ch, out_ch, wn1, wn2, dtype=torch.cfloat) * scale))
        self.w2 = nn.Parameter(torch.view_as_real(
            torch.randn(in_ch, out_ch, wn1, wn2, dtype=torch.cfloat) * scale))

    def _cmul(self, x, w):
        return torch.einsum("bixy,ioxy->boxy", x, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1  = torch.view_as_complex(self.w1)
        w2  = torch.view_as_complex(self.w2)
        B, _, H, W = x.shape
        x_ft   = torch.fft.rfft2(x)
        out_ft = torch.zeros(B, self.out_ch, H, W // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :,  :self.wn1, :self.wn2] = self._cmul(x_ft[:, :,  :self.wn1, :self.wn2], w1)
        out_ft[:, :, -self.wn1:, :self.wn2] = self._cmul(x_ft[:, :, -self.wn1:, :self.wn2], w2)
        return torch.fft.irfft2(out_ft, s=(H, W))


# ----- 5. IO Layer -----
class _IOLayer(nn.Module):
    def __init__(self, features: int, wavenumber: int, drop: float = 0.):
        super().__init__()
        self.fconv = _FourierConv2d(features, features, wavenumber, wavenumber)
        self.skip  = nn.Conv2d(features, features, 1)
        self.act   = nn.GELU()
        self.drop  = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.drop(self.fconv(x) + self.skip(x)))


# ----- 6. IO Residual Block -----
class _IOResBlock(nn.Module):
    def __init__(self, features: int, wavenumber: int,
                 drop_path: float = 0., drop: float = 0.):
        super().__init__()
        self.io    = _IOLayer(features, wavenumber, drop)
        self.norm1 = _LayerNorm2D(features, data_format="channels_first")
        self.norm2 = _LayerNorm2D(features, data_format="channels_first")
        self.pw1   = nn.Conv2d(features, 4 * features, 1)
        self.pw2   = nn.Conv2d(4 * features, features, 1)
        self.act   = nn.GELU()
        self.dp    = _DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = x
        x = self.dp(self.io(self.norm1(x))) + skip
        skip = x
        x = self.pw2(self.act(self.pw1(self.norm2(x))))
        return self.dp(x) + skip


# =============================================================================
# 2D Meta Networks
# =============================================================================

# -----------1. Meta-T Network (2D) ------------
class MetaT2D(nn.Module):
    """
    Generates a per-sample complex 2D conv kernel [B, Cout, Cin, 3, 3].
    Adapted from Meta_T in FNS/model/FNS.py.
    """
    def __init__(self, in_channels: int, out_channels: int, act: str):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 4, 3),
            getActivationFunction(act),
            ResBlock2D(4, 4, 3, act),
            nn.Conv2d(4, 8, 3),
            getActivationFunction(act),
            ResBlock2D(8, 8, 3, act),
            nn.Conv2d(8, in_channels * out_channels, 3),
            getActivationFunction(act),
            nn.AdaptiveAvgPool2d(3),
        )
        self.fnn = nn.Sequential(
            nn.Linear(in_channels * out_channels * 9, 200),
            getActivationFunction(act),
            nn.Linear(200, 200),
            getActivationFunction(act),
            nn.Linear(200, 200),
            getActivationFunction(act),
            nn.Linear(200, 2 * in_channels * out_channels * 9),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 1, Nx, Ny] → [B, Cout, Cin, 3, 3] complex"""
        x = self.cnn(x).view(x.shape[0], -1)
        x = self.fnn(x)
        dim    = self.in_channels * self.out_channels * 9
        x_real = x[:, :dim].view(-1, self.out_channels, self.in_channels, 3, 3)
        x_imag = x[:, dim:].view(-1, self.out_channels, self.in_channels, 3, 3)
        return x_real + 1j * x_imag


# -----------2. Meta-λ: sFNO backbone (2D) ------------
class sFNO_epsilon_v2(nn.Module):
    """
    FNO-based hyper-network producing spatial predictions [B, out_channel, N, N].

    Config keys:
        dims           list[int]   channel widths per stage
        depths         list[int]   IO-ResBlock count per stage
        modes          list[int]   Fourier wavenumbers per stage
        padding        int         zero-padding added before stages, stripped after
        drop_path_rate float       max stochastic-depth rate (linearly scaled)
        drop           float       dropout inside IO layers
    """
    def __init__(self, config: dict, in_channel: int, out_channel: int):
        super().__init__()
        dims, depths, modes = config["dims"], config["depths"], config["modes"]
        self.padding = config.get("padding", 0)

        # Lifting chain: stem + (n-1) × (LayerNorm → Conv)
        self.lifting = nn.ModuleList()
        self.lifting.append(nn.Conv2d(in_channel, dims[0], 1))
        for i in range(len(dims) - 1):
            self.lifting.append(nn.Sequential(
                _LayerNorm2D(dims[i], data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], 1),
            ))

        # Stages: one per dim level.
        # stages[0] is built but intentionally skipped in forward — the stem
        # level uses only the lifting conv before interpolation.
        dp_rates = torch.linspace(0, config.get("drop_path_rate", 0.0), sum(depths)).tolist()
        cur = 0
        self.stages = nn.ModuleList()
        for i, (d, m) in enumerate(zip(dims, modes)):
            self.stages.append(nn.Sequential(*[
                _IOResBlock(d, m,
                            drop_path=dp_rates[cur + j],
                            drop=config.get("drop", 0.0))
                for j in range(depths[i])
            ]))
            cur += depths[i]

        self.head = nn.Conv2d(dims[-1], out_channel, 1)

    def forward(self, x: torch.Tensor, N: int) -> torch.Tensor:
        """
        x: [B, in_channel, H, W]
        N: target spatial resolution after interpolation
        Returns: [B, out_channel, N, N]
        """
        x = self.lifting[0](x)
        x = F.interpolate(x, size=(N, N), mode='bilinear', align_corners=True)
        if self.padding > 0:
            x = F.pad(x, [0, self.padding, 0, self.padding])
        for i in range(1, len(self.lifting)):
            x = self.lifting[i](x)
            x = self.stages[i](x)
        if self.padding > 0:
            x = x[..., :-self.padding, :-self.padding]
        return self.head(x)


# -----------3. Meta-λ: FNO wrapper (2D) ------------
class FNOMetaLambda2D(nn.Module):
    """
    Wraps sFNO_epsilon_v2 to match the MetaUNet interface:
        forward(k_norm, N) → [B, 1, N, N] cfloat

    Prepends 2D grid coordinates to k before calling sFNO, then converts
    the real spatial output to complex frequency-domain weights via fft2.

    Config keys (sub-set of fns_setting):
        dims / depths / modes / fno_padding / drop_path_rate / drop
    """
    def __init__(self, config):
        super().__init__()
        fno_cfg = {
            "dims":           config.get("dims",            [32, 64, 64, 32]),
            "depths":         config.get("depths",          [2, 2, 2, 2]),
            "modes":          config.get("modes",           [8, 8, 8, 8]),
            "padding":        config.get("fno_padding",     2),
            "drop_path_rate": config.get("drop_path_rate",  0.0),
            "drop":           config.get("drop",            0.0),
        }
        # in_channel=3: k_norm + grid_x + grid_y
        self.sfno = sFNO_epsilon_v2(fno_cfg, in_channel=3, out_channel=1)

    def forward(self, k_norm: torch.Tensor, N: int) -> torch.Tensor:
        """
        k_norm: [B, 1, Nx, Ny]
        N:      target band size passed to sFNO interpolation
        Returns: [B, 1, N, N] cfloat
        """
        B, _, Nx, Ny = k_norm.shape
        gx  = torch.linspace(0, 1, Nx, device=k_norm.device).view(1, 1, Nx, 1).expand(B, 1, Nx, Ny)
        gy  = torch.linspace(0, 1, Ny, device=k_norm.device).view(1, 1, 1, Ny).expand(B, 1, Nx, Ny)
        x   = torch.cat([k_norm, gx, gy], dim=1)   # [B, 3, Nx, Ny]
        out = self.sfno(x, N)                        # [B, 1, N, N] real
        w   = torch.fft.fft2(out, dim=(2, 3)) / (N ** 2)
        return torch.fft.fftshift(w, dim=(2, 3))


# -----------4. Meta-λ: Query-based (2D) ------------
class QueryMetaLambda2D(nn.Module):
    """
    Query-based 2D meta-λ: directly analogous to FNOMetaLambda1D.

    Pipeline:
      k_norm [B,1,Nx,Ny] → FNO backbone → global z [B, hidden]
      For each of N² positions (u,v) in [0,1]²:
          sinusoidal embedding phi(u,v) + z → MLP → one complex weight
      → [B, 1, N, N] cfloat   (no fft2, no /N² needed)

    Because positions are always normalized to [0,1]², the output size N
    can change freely at inference without going out of distribution.

    Config keys (sub-set of fns_setting):
        hidden   int         backbone + MLP width          (default 32)
        modes    int|list    FNO backbone wavenumbers       (default 8)
        Lfreq    int         sinusoidal encoding levels     (default 4)
        act      str         activation function            (default "gelu")
    """
    def __init__(self, config):
        super().__init__()
        hidden     = config.get("hidden", 32)
        modes_raw  = config.get("modes",  8)
        modes      = modes_raw[0] if isinstance(modes_raw, (list, tuple)) else int(modes_raw)
        self.Lfreq = config.get("Lfreq",  4)
        act        = config.get("act",   "gelu")

        # Backbone: k_norm [B, 1, Nx, Ny] → z [B, hidden]
        # Two FNO layers (spectral conv + pointwise skip), then global avg pool.
        # Directly mirrors FNOMetaLambda1D's SpectralConv + w structure in 2D.
        self.lifting = nn.Conv2d(1, hidden, 1)
        self.fconv1  = _FourierConv2d(hidden, hidden, modes, modes)
        self.skip1   = nn.Conv2d(hidden, hidden, 1)
        self.fconv2  = _FourierConv2d(hidden, hidden, modes, modes)
        self.skip2   = nn.Conv2d(hidden, hidden, 1)
        self.act_fn  = getActivationFunction(act)

        # Query head: (z, phi) → (real, imag) for one spectral weight
        # D_phi = 2 raw coords + 4 sinusoids-per-level-per-axis
        D_phi = 4 * self.Lfreq + 2
        self.query_head = nn.Sequential(
            nn.Linear(hidden + D_phi, hidden),
            getActivationFunction(act),
            nn.Linear(hidden, hidden),
            getActivationFunction(act),
            nn.Linear(hidden, 2),
        )

    def _freq_embed_2d(self, N: int, device) -> torch.Tensor:
        """Sinusoidal 2D positional encoding. Returns [N², 4*Lfreq+2]."""
        t        = torch.linspace(0., 1., N, device=device)
        tu, tv   = torch.meshgrid(t, t, indexing='ij')
        tu, tv   = tu.reshape(-1, 1), tv.reshape(-1, 1)   # [N², 1]
        feats    = [tu, tv]
        for k in range(self.Lfreq):
            w = (2. ** k) * 2. * torch.pi
            feats += [torch.sin(w * tu), torch.cos(w * tu),
                      torch.sin(w * tv), torch.cos(w * tv)]
        return torch.cat(feats, dim=-1)   # [N², 4*Lfreq+2]

    def forward(self, k_norm: torch.Tensor, N: int) -> torch.Tensor:
        """
        k_norm: [B, 1, Nx, Ny]
        N:      band size
        Returns [B, 1, N, N] cfloat — directly complex spectral weights.
        """
        # backbone: extract global condition
        x = self.lifting(k_norm)
        x = self.act_fn(self.fconv1(x) + self.skip1(x))
        x = self.act_fn(self.fconv2(x) + self.skip2(x))
        z = x.mean(dim=(-2, -1))                           # [B, hidden]

        # query at N² positions
        B   = z.shape[0]
        phi = self._freq_embed_2d(N, k_norm.device)        # [N², D_phi]
        inp = torch.cat([
            z.unsqueeze(1).expand(-1, N * N, -1),
            phi.unsqueeze(0).expand(B,  -1, -1),
        ], dim=-1)                                          # [B, N², hidden+D_phi]
        out = self.query_head(inp)                          # [B, N², 2]
        w   = torch.complex(out[..., 0], out[..., 1])
        return w.view(B, 1, N, N)
