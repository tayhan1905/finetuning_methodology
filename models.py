import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.svd_utils import svd_head_tail

# ----------------------------
# SALT
# ----------------------------
class SALT(nn.Module):
    def __init__(self, base_linear: nn.Linear, r_top: int, tail_rank: int = 16):
        super().__init__()
        assert isinstance(base_linear, nn.Linear)
        self.base = base_linear
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.has_bias = base_linear.bias is not None

        self.base.weight.requires_grad_(False)
        if self.has_bias:
            self.base.bias.requires_grad_(False)

        W = self.base.weight.detach().to(torch.float32)
        (U_top, S_top, Vh_top), (U_tail, S_tail, Vh_tail) = svd_head_tail(W, r_top)

        dtype, device = self.base.weight.dtype, self.base.weight.device
        self.register_buffer("U_top",  U_top.to(dtype).to(device))
        self.register_buffer("S_top",  S_top.to(dtype).to(device))
        self.register_buffer("Vh_top", Vh_top.to(dtype).to(device))
        self.register_buffer("U_tail",  U_tail.to(dtype).to(device))
        self.register_buffer("S_tail",  S_tail.to(dtype).to(device))
        self.register_buffer("Vh_tail", Vh_tail.to(dtype).to(device))

        self.r = S_top.numel()
        self.r_tail = S_tail.numel()

        if self.r > 0:
            self.alpha = nn.Parameter(torch.zeros(self.r, dtype=dtype, device=device))
            self.beta  = nn.Parameter(torch.zeros(self.r, dtype=dtype, device=device))
        else:
            self.alpha = None
            self.beta  = None

        if self.r_tail > 0 and tail_rank > 0:
            self.tail_rank = min(tail_rank, self.r_tail)
            # LoRA in Σ_tail
            self.X = nn.Parameter(torch.zeros(self.r_tail, self.tail_rank, dtype=dtype, device=device))
            self.Y = nn.Parameter(torch.zeros(self.tail_rank, self.r_tail, dtype=dtype, device=device))
            with torch.no_grad():
                # LoRA-style init: A random, B zero
                self.X.add_(0.01 * torch.randn_like(self.X))
        else:
            self.tail_rank = 0
            self.X = None
            self.Y = None

    def forward(self, x):
        parts = []

        if self.r > 0:
            delta_sigma_top = self.S_top * self.alpha + self.beta
            Sigma_top = torch.diag(F.relu(delta_sigma_top))
            W_top = self.U_top @ Sigma_top @ self.Vh_top
            parts.append(W_top)

        if self.r_tail > 0:
            Sigma_tail = torch.diag(self.S_tail)
            if self.tail_rank > 0:
                Sigma_tail = Sigma_tail + (self.X @ self.Y)
            Sigma_tail = F.relu(Sigma_tail)
            W_tail = self.U_tail @ Sigma_tail @ self.Vh_tail
            parts.append(W_tail)

        W_tilde = parts[0] if len(parts) == 1 else parts[0] + parts[1]
        return F.linear(x, W_tilde, self.base.bias)

# ----------------------------
# SALT + EDoRA (tail = BRA)
# ----------------------------
class SALTEdoraLinear(nn.Module):
    """
    SALT top; EDoRA tail: ΔW_tail = B R A
    """
    def __init__(self, base_linear: nn.Linear, r: int):
        super().__init__()
        self.base = base_linear
        self.in_features, self.out_features = base_linear.in_features, base_linear.out_features
        self.has_bias = base_linear.bias is not None

        self.base.weight.requires_grad_(False)
        if self.has_bias:
            self.base.bias.requires_grad_(False)

        W = self.base.weight.detach().to(torch.float32)
        (U_top, S_top, Vh_top), (U_tail, S_tail, Vh_tail) = svd_head_tail(W, r)
        dtype, device = self.base.weight.dtype, self.base.weight.device

        self.register_buffer("U_top",  U_top.to(dtype).to(device))
        self.register_buffer("S_top",  S_top.to(dtype).to(device))
        self.register_buffer("Vh_top", Vh_top.to(dtype).to(device))
        self.register_buffer("U_tail",  U_tail.to(dtype).to(device))
        self.register_buffer("S_tail",  S_tail.to(dtype).to(device))
        self.register_buffer("Vh_tail", Vh_tail.to(dtype).to(device))

        self.r_top = S_top.numel()
        self.r_tail = S_tail.numel()

        self.alpha = nn.Parameter(torch.zeros(self.r_top, dtype=dtype, device=device))
        self.beta  = nn.Parameter(torch.zeros(self.r_top, dtype=dtype, device=device))

        if self.r_tail > 0:
            self.R = nn.Parameter(torch.randn(self.r_tail, self.r_tail, dtype=dtype, device=device) * 0.01)

    def forward(self, x):
        parts = []

        if self.r_top > 0:
            delta_sigma_top = self.S_top * self.alpha + self.beta
            Sigma_top = torch.diag(F.relu(delta_sigma_top))
            W_top = self.U_top @ Sigma_top @ self.Vh_top
            parts.append(W_top)

        if self.r_tail > 0:
            B = self.U_tail @ torch.diag(self.S_tail)
            A = self.Vh_tail
            delta_tail = B @ self.R @ A
            W_tail = (self.U_tail @ torch.diag(self.S_tail) @ self.Vh_tail) + delta_tail
            parts.append(W_tail)

        W_tilde = parts[0] if len(parts) == 1 else parts[0] + parts[1]
        return F.linear(x, W_tilde, self.base.bias)

# ----------------------------
# SALT + EDoRA V2 (tail = BDRA)
# ----------------------------
class SALTEdoraLinearV2(nn.Module):
    """
    SALT–EDoRA–V2 (BDRA):
    Top: SALT (α, β). Tail: B D R A (D diagonal magnitude, R rotation).
    """
    def __init__(self, base_linear: nn.Linear, r: int):
        super().__init__()
        assert isinstance(base_linear, nn.Linear)
        self.base = base_linear
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.has_bias = base_linear.bias is not None

        self.base.weight.requires_grad_(False)
        if self.has_bias:
            self.base.bias.requires_grad_(False)

        W = self.base.weight.detach().to(torch.float32)
        (U_top, S_top, Vh_top), (U_tail, S_tail, Vh_tail) = svd_head_tail(W, r)
        dtype, device = self.base.weight.dtype, self.base.weight.device

        self.register_buffer("U_top",  U_top.to(dtype).to(device))
        self.register_buffer("S_top",  S_top.to(dtype).to(device))
        self.register_buffer("Vh_top", Vh_top.to(dtype).to(device))
        self.register_buffer("U_tail",  U_tail.to(dtype).to(device))
        self.register_buffer("S_tail",  S_tail.to(dtype).to(device))
        self.register_buffer("Vh_tail", Vh_tail.to(dtype).to(device))

        self.r_top = S_top.numel()
        self.r_tail = S_tail.numel()

        self.alpha = nn.Parameter(torch.zeros(self.r_top, dtype=dtype, device=device))
        self.beta  = nn.Parameter(torch.zeros(self.r_top, dtype=dtype, device=device))

        if self.r_tail > 0:
            self.D = nn.Parameter(torch.ones(self.r_tail, dtype=dtype, device=device))
            self.R = nn.Parameter(torch.eye(self.r_tail, dtype=dtype, device=device))
            with torch.no_grad():
                self.R.add_(0.01 * torch.randn_like(self.R))

    def forward(self, x):
        parts = []

        if self.r_top > 0:
            delta_sigma_top = self.S_top * self.alpha + self.beta
            Sigma_top = torch.diag(F.relu(delta_sigma_top))
            W_top = self.U_top @ Sigma_top @ self.Vh_top
            parts.append(W_top)

        if self.r_tail > 0:
            W_tail_base = self.U_tail @ torch.diag(self.S_tail) @ self.Vh_tail
            B = self.U_tail @ torch.diag(self.S_tail)
            A = self.Vh_tail
            Dm = torch.diag(self.D)
            delta_tail = B @ Dm @ self.R @ A
            W_tail = W_tail_base + delta_tail
            parts.append(W_tail)

        W_tilde = parts[0] if len(parts) == 1 else parts[0] + parts[1]
        return F.linear(x, W_tilde, self.base.bias)