import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.svd_utils import svd_head_tail

# ----------------------------
# SALT
# ----------------------------
class SALT(nn.Module):

    def __init__(self, base_linear: nn.Linear, r: int, lora_rank: int = 8):
        super().__init__()
        assert isinstance(base_linear, nn.Linear)
        self.base = base_linear
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.has_bias = base_linear.bias is not None

        # Freeze pretrained base weights
        self.base.weight.requires_grad_(False)
        if self.has_bias:
            self.base.bias.requires_grad_(False)

        # Compute SVD
        W = self.base.weight.detach().to(torch.float32)
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)

        p = S.numel()
        r_top = min(r, p)
        r_tail = p - r_top

        # Split head/tail singular subspaces
        self.register_buffer("U_top",  U[:, :r_top])
        self.register_buffer("S_top",  S[:r_top])
        self.register_buffer("Vh_top", Vh[:r_top, :])
        self.register_buffer("U_tail",  U[:, r_top:])
        self.register_buffer("S_tail",  S[r_top:])
        self.register_buffer("Vh_tail", Vh[r_top:, :])

        dtype, device = self.base.weight.dtype, self.base.weight.device

        # --- Top-r: spectral scale & shift ---
        self.alpha = nn.Parameter(torch.ones(r_top, dtype=dtype, device=device))
        self.beta  = nn.Parameter(torch.zeros(r_top, dtype=dtype, device=device))

        # --- Tail: low-rank LoRA update ---
        self.X = nn.Parameter(torch.zeros(r_tail, lora_rank, dtype=dtype, device=device))
        self.Y = nn.Parameter(torch.zeros(lora_rank, r_tail, dtype=dtype, device=device))
        nn.init.normal_(self.X, std=0.01)
        nn.init.zeros_(self.Y)

    def forward(self, x):
        # --- Top part: scale and shift dominant spectrum ---
        Sigma_top = torch.diag(F.relu(self.S_top * self.alpha + self.beta))
        W_top = self.U_top @ Sigma_top @ self.Vh_top

        # --- Tail part: LoRA-style low-rank update ---
        if self.S_tail.numel() > 0:
            Sigma_tail = torch.diag(self.S_tail) + (self.X @ self.Y)
            Sigma_tail = F.relu(Sigma_tail)
            W_tail = self.U_tail @ Sigma_tail @ self.Vh_tail
        else:
            W_tail = 0.0

        # Combine
        W_tilde = W_top + W_tail
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
        # Start from the frozen pretrained weights
        W_tilde = self.base.weight

        # SALT top adaptation
        if self.r_top > 0:
            delta_sigma_top = self.S_top * self.alpha + self.beta
            Sigma_top = torch.diag(F.relu(delta_sigma_top))
            W_top_delta = self.U_top @ Sigma_top @ self.Vh_top
            W_tilde = W_tilde + W_top_delta

        # EDoRA tail adaptation (BDRA-style)
        if self.r_tail > 0:
            B = self.U_tail @ torch.diag(self.S_tail)
            A = self.Vh_tail
            delta_tail = B @ self.R @ A
            W_tilde = W_tilde + delta_tail

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
        # Start from frozen pretrained base weights
        W_tilde = self.base.weight

        # Apply SALT top adaptation
        if self.r_top > 0:
            delta_sigma_top = self.S_top * self.alpha + self.beta
            Sigma_top = torch.diag(F.relu(delta_sigma_top))
            W_top_delta = self.U_top @ Sigma_top @ self.Vh_top
            W_tilde = W_tilde + W_top_delta

        # Apply EDoRA tail adaptation
        if self.r_tail > 0:
            B = self.U_tail @ torch.diag(self.S_tail)
            A = self.Vh_tail
            Dm = torch.diag(self.D)
            delta_tail = B @ Dm @ self.R @ A
            W_tilde = W_tilde + delta_tail

        return F.linear(x, W_tilde, self.base.bias)
