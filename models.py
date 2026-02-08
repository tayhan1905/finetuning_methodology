import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.svd_utils import svd_head_tail

# This is for adaptive eigen dispersion
def choose_head_rank_by_eigen_dispersion(S: torch.Tensor,
                                         min_r: int = 1,
                                         max_r: int | None = None,
                                         energy_threshold: float = 0.90) -> int:
    S = S.detach().float().clamp_min(1e-12)
    n = S.numel()  
    if n <= 2:
        return max(min_r, min(n, n - 1))
    if max_r is None:
        max_r = n - 1
    log_s = torch.log(S)
    d1 = log_s[:-1] - log_s[1:]
    d2 = d1[:-1] - d1[1:]
    k_rel = torch.argmax(d2).item()
    r_top = k_rel + 1
    if d2[k_rel] < 1e-3:
        cum = torch.cumsum(S**2, 0)
        r_top = int(torch.searchsorted(cum, energy_threshold * cum[-1]).item() + 1)
    return max(min_r, min(r_top, max_r))

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

        # Freeze pretrained base weights + bias >> to be reused
        self.base.weight.requires_grad_(False)
        if self.has_bias:
            self.base.bias.requires_grad_(False)

        # Compute SVD
        W = self.base.weight.detach().to(torch.float32)
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)

        p = S.numel()
        r_top = min(r, p)
        r_tail = p - r_top

        # Split head/tail singular subspaces + save them as buffers so that they are not trained
        self.register_buffer("U_top",  U[:, :r_top])
        self.register_buffer("S_top",  S[:r_top])
        self.register_buffer("Vh_top", Vh[:r_top, :])
        self.register_buffer("U_tail",  U[:, r_top:])
        self.register_buffer("S_tail",  S[r_top:])
        self.register_buffer("Vh_tail", Vh[r_top:, :])

        dtype, device = self.base.weight.dtype, self.base.weight.device

        # scale and shift
        self.alpha = nn.Parameter(torch.ones(r_top, dtype=dtype, device=device))
        self.beta  = nn.Parameter(torch.zeros(r_top, dtype=dtype, device=device))

        # low-rank LoRA update
        self.X = nn.Parameter(torch.zeros(r_tail, lora_rank, dtype=dtype, device=device))
        self.Y = nn.Parameter(torch.zeros(lora_rank, r_tail, dtype=dtype, device=device))
        nn.init.normal_(self.X, std=0.01)
        nn.init.zeros_(self.Y)

    def forward(self, x):
        # scale and shift with relu
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

        self.alpha = nn.Parameter(torch.ones(self.r_top, dtype=dtype, device=device))
        self.beta  = nn.Parameter(torch.zeros(self.r_top, dtype=dtype, device=device))

        if self.r_tail > 0:
            self.D = nn.Parameter(torch.ones(self.r_tail, dtype=dtype, device=device))
            self.R = nn.Parameter(torch.eye(self.r_tail, dtype=dtype, device=device))
            with torch.no_grad():
                self.R.add_(0.01 * torch.randn_like(self.R))

    def forward(self, x):
        # Start from frozen pretrained base weights
        W_tail = self.U_tail @ torch.diag(self.S_tail) @ self.Vh_tail

        # Apply SALT top adaptation
        if self.r_top > 0:
            delta_sigma_top = self.S_top * self.alpha + self.beta
            Sigma_top = torch.diag(F.relu(delta_sigma_top))
            W_top_delta = self.U_top @ Sigma_top @ self.Vh_top

        # Apply EDoRA tail adaptation
        if self.r_tail > 0:
            B = self.U_tail @ torch.diag(self.S_tail)
            A = self.Vh_tail
            Dm = torch.diag(F.relu(self.D))
            delta_tail = B @ Dm @ self.R @ A
            W_tail = W_tail + delta_tail

        W_tilde = W_tail + W_top_delta

        return F.linear(x, W_tilde, self.base.bias)


# ======================================================
# Version 3 — Static split by eigen dispersion (with intrinsic tail rank)
# ======================================================
class SALTEdoraLinearV3(nn.Module):
    """
    SALT–EDoRA–V3 (Intrinsic Rank Version):
      * One-time SVD of frozen base weight.
      * Auto-split head/tail by eigen dispersion (e.g., 600/400).
      * Head (top): SALT (α, β) — scale & shift singular values.
      * Tail (bottom): EDoRA (BDRA) — modeled using only the top-r singulars
        from within the tail subspace (intrinsic rank compression).
    """

    def __init__(self, base_linear: nn.Linear,
                 r_intrinsic: int = 4,
                 min_r_top: int = 1,
                 max_r_top: int | None = None,
                 r_top_override: int | None = None,
                 energy_threshold: float = 0.9):
        super().__init__()
        assert isinstance(base_linear, nn.Linear)
        self.base = base_linear
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.has_bias = base_linear.bias is not None

        # Freeze pretrained parameters
        self.base.weight.requires_grad_(False)
        if self.has_bias:
            self.base.bias.requires_grad_(False)

        # Take pretrained weights and run SVD
        W = self.base.weight.detach().to(torch.float32)
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        dtype, device = W.dtype, W.device

        # 1️⃣ Split by eigen dispersion (e.g., head=600, tail=400)
        if r_top_override is None:
            r_top = choose_head_rank_by_eigen_dispersion(S, min_r_top, max_r_top, energy_threshold=energy_threshold)
        else:
            r_top = int(r_top_override)
        r_top = max(0, min(r_top, S.numel()))
        r_tail = S.numel() - r_top

        # 2️⃣ Slice SVD into head and tail
        U_top,  S_top,  Vh_top  = U[:, :r_top],  S[:r_top],  Vh[:r_top, :]
        U_tail, S_tail, Vh_tail = U[:, r_top:], S[r_top:], Vh[r_top:, :]

        # 3️⃣ Further compress the tail: only keep intrinsic rank r_intrinsic
        # e.g. tail has 400 singulars, but we only use top 4 of them.
        self.r_intrinsic = min(r_intrinsic, r_tail)
        U_tail_r  = U_tail[:, :self.r_intrinsic]
        S_tail_r  = S_tail[:self.r_intrinsic]
        Vh_tail_r = Vh_tail[:self.r_intrinsic, :]

        # 4️⃣ Register buffers (frozen decomposition)
        self.register_buffer("U_top",  U_top.to(dtype).to(device))
        self.register_buffer("S_top",  S_top.to(dtype).to(device))
        self.register_buffer("Vh_top", Vh_top.to(dtype).to(device))
        self.register_buffer("U_tail",  U_tail.to(dtype).to(device))
        self.register_buffer("S_tail",  S_tail.to(dtype).to(device))
        self.register_buffer("Vh_tail", Vh_tail.to(dtype).to(device))
        self.register_buffer("U_tail_r",  U_tail_r.to(dtype).to(device))
        self.register_buffer("S_tail_r",  S_tail_r.to(dtype).to(device))
        self.register_buffer("Vh_tail_r", Vh_tail_r.to(dtype).to(device))

        self.r_top = int(S_top.numel())
        self.r_tail = int(S_tail.numel())

        # 5️⃣ SALT params for head (α, β)
        if self.r_top > 0:
            self.alpha = nn.Parameter(torch.ones(self.r_top, dtype=dtype, device=device))
            self.beta  = nn.Parameter(torch.zeros(self.r_top, dtype=dtype, device=device))
        else:
            self.register_buffer("alpha", torch.zeros(0, dtype=dtype, device=device))
            self.register_buffer("beta",  torch.zeros(0, dtype=dtype, device=device))

        # 6️⃣ BDRA params for intrinsic tail (D, R)
        if self.r_intrinsic > 0:
            self.D = nn.Parameter(torch.ones(self.r_intrinsic, dtype=dtype, device=device))
            R0 = torch.eye(self.r_intrinsic, dtype=dtype, device=device)
            R0 = R0 + 0.01 * torch.randn_like(R0)
            self.R = nn.Parameter(R0)
        else:
            self.register_buffer("D", torch.zeros(0, dtype=dtype, device=device))
            self.register_buffer("R", torch.zeros(0, 0, dtype=dtype, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # --- Head reconstruction (SALT) ---
        if self.r_top > 0:
            delta_sigma_top = self.S_top * self.alpha + self.beta
            Sigma_top = torch.diag(F.relu(delta_sigma_top))
            W_head = self.U_top @ Sigma_top @ self.Vh_top
        else:
            W_head = torch.zeros_like(self.base.weight, dtype=self.base.weight.dtype, device=self.base.weight.device)

        # --- Tail reconstruction (low-energy base) ---
        if self.r_tail > 0:
            W_tail_base = self.U_tail @ torch.diag(self.S_tail) @ self.Vh_tail
        else:
            W_tail_base = torch.zeros_like(W_head)

        # --- EDoRA (BDRA) adaptation on intrinsic tail rank ---
        if self.r_intrinsic > 0:
            B = self.U_tail_r @ torch.diag(self.S_tail_r)
            A = self.Vh_tail_r
            Dm = torch.diag(F.relu(self.D))
            delta_tail = B @ Dm @ self.R @ A
        else:
            delta_tail = torch.zeros_like(W_tail_base)

        # --- Combine adapted head + low-rank tail ---
        W_eff = W_head + W_tail_base + delta_tail

        return F.linear(x, W_eff, self.base.bias)

# ======================================================
# Version 4 — Adaptive eigen dispersion (dynamic split)
# ======================================================
class SALTEdoraLinearV4(nn.Module):
    """
    SALT–EDoRA–V4 (Adaptive Dispersion):
      * Starts from SVD of frozen base weight.
      * Periodically re-evaluates head/tail split by eigen dispersion
        using the CURRENT adapted singulars (SALT-updated head + frozen tail).
      * Head: SALT (α, β) on top singulars.
      * Tail: BDRA with user-chosen intrinsic rank r_intrinsic inside current tail.
    """

    def __init__(self, base_linear: nn.Linear,
                 r_intrinsic: int = 4,
                 min_r_top: int = 1,
                 max_r_top: int | None = None,
                 update_interval: int = 100,
                 ema_momentum: float = 0.9):
        super().__init__()
        assert isinstance(base_linear, nn.Linear)
        self.base = base_linear
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.has_bias = base_linear.bias is not None

        # Freeze pretrained backbone
        self.base.weight.requires_grad_(False)
        if self.has_bias:
            self.base.bias.requires_grad_(False)

        # Base SVD
        W = self.base.weight.detach().to(torch.float32)
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        dtype, device = W.dtype, W.device
        self.register_buffer("U_full", U.to(dtype).to(device))
        self.register_buffer("S_full", S.to(dtype).to(device))
        self.register_buffer("Vh_full", Vh.to(dtype).to(device))

        self.min_r_top = min_r_top
        self.max_r_top = max_r_top or S.numel()
        self.update_interval = update_interval
        self.ema_momentum = float(ema_momentum)
        self.register_buffer("step", torch.zeros(1, dtype=torch.long))

        # Initial split (static heuristic on base S just once at init)
        self.r_top = choose_head_rank_by_eigen_dispersion(S, min_r_top, max_r_top)
        self.r_intrinsic = min(r_intrinsic, S.numel() - self.r_top)

        # Initialize subspaces & params
        self._initialize_subspaces(dtype, device)

        # Initialize EMA spectrum buffer with the current adapted spectrum
        with torch.no_grad():
            S_head_adapted = (self.S_top * torch.ones_like(self.S_top) + torch.zeros_like(self.S_top)).clamp_min(0)
            S_current = torch.cat([S_head_adapted, self.S_tail])
            self.register_buffer("S_est", S_current.clone())

    # --------------------------
    # Helper to initialize / resize parameters
    # --------------------------
    def _initialize_subspaces(self, dtype, device):
        k = self.S_full.numel()
        r_top, r_tail = self.r_top, k - self.r_top
        U, S, Vh = self.U_full, self.S_full, self.Vh_full

        # SVD slices
        self.register_buffer("U_top",  U[:, :r_top].to(dtype).to(device))
        self.register_buffer("S_top",  S[:r_top].to(dtype).to(device))
        self.register_buffer("Vh_top", Vh[:r_top, :].to(dtype).to(device))
        self.register_buffer("U_tail",  U[:, r_top:].to(dtype).to(device))
        self.register_buffer("S_tail",  S[r_top:].to(dtype).to(device))
        self.register_buffer("Vh_tail", Vh[r_top:, :].to(dtype).to(device))

        # Intrinsic tail rank compression
        self.r_intrinsic = min(self.r_intrinsic, r_tail)
        self.register_buffer("U_tail_r",  self.U_tail[:, :self.r_intrinsic].to(dtype).to(device))
        self.register_buffer("S_tail_r",  self.S_tail[:self.r_intrinsic].to(dtype).to(device))
        self.register_buffer("Vh_tail_r", self.Vh_tail[:self.r_intrinsic, :].to(dtype).to(device))

        # SALT params
        self.alpha = nn.Parameter(torch.ones(r_top, dtype=dtype, device=device))
        self.beta  = nn.Parameter(torch.zeros(r_top, dtype=dtype, device=device))

        # BDRA params
        self.D = nn.Parameter(torch.ones(self.r_intrinsic, dtype=dtype, device=device))
        R0 = torch.eye(self.r_intrinsic, dtype=dtype, device=device)
        R0 = R0 + 0.01 * torch.randn_like(R0)
        self.R = nn.Parameter(R0)

    # --------------------------
    # Periodic adaptive update (uses LIVE adapted singulars + EMA)
    # --------------------------
    def maybe_update_split(self):
        self.step += 1
        if self.step % self.update_interval != 0:
            return

        with torch.no_grad():
            # === 1️⃣ Compute current spectrum ===
            S_head_adapted = (self.S_top * self.alpha + self.beta).clamp_min(0)

            # Tail: combine BDRA-adapted subspace and untouched remainder
            S_tail_adapted = torch.cat([
                (self.S_tail_r * F.relu(self.D)),  # adapted intrinsic tail
                self.S_tail[self.r_intrinsic:]     # remaining unmodified tail singulars
            ])

            S_current = torch.cat([S_head_adapted, S_tail_adapted])

            # === 2️⃣ EMA smoothing ===
            m = self.ema_momentum
            self.S_est.mul_(m).add_((1.0 - m) * S_current)

            # === 3️⃣ Re-evaluate eigen dispersion ===
            new_r_top = choose_head_rank_by_eigen_dispersion(
                self.S_est, self.min_r_top, self.max_r_top
            )

            # === 4️⃣ Rebuild if boundary moves ===
            if int(new_r_top) != int(self.r_top):
                print(f"[Adaptive Split] r_top {self.r_top} → {int(new_r_top)}")
                self.r_top = int(new_r_top)
                dtype, device = self.S_full.dtype, self.S_full.device
                self._initialize_subspaces(dtype, device)

                # Reset EMA with updated shapes
                S_head_adapted = (self.S_top * self.alpha + self.beta).clamp_min(0)
                S_tail_adapted = torch.cat([
                    (self.S_tail_r * F.relu(self.D)),
                    self.S_tail[self.r_intrinsic:]
                ])
                self.S_est = torch.cat([S_head_adapted, S_tail_adapted]).detach()


    # --------------------------
    # Forward
    # --------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.maybe_update_split()

        # === SALT head ===
        delta_sigma_top = self.S_top * self.alpha + self.beta
        Sigma_top = torch.diag(F.relu(delta_sigma_top))
        W_head = self.U_top @ Sigma_top @ self.Vh_top

        # === Base tail ===
        W_tail_base = self.U_tail @ torch.diag(self.S_tail) @ self.Vh_tail

        # === BDRA intrinsic tail ===
        B = self.U_tail_r @ torch.diag(self.S_tail_r)
        A = self.Vh_tail_r
        Dm = torch.diag(F.relu(self.D))
        delta_tail = B @ Dm @ self.R @ A

        W_eff = W_head + W_tail_base + delta_tail
        return F.linear(x, W_eff, self.base.bias)
