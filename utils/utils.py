"""
utils/utils.py
==============
Shared mathematical utilities for weight-space analysis.
"""

import torch
import numpy as np


# ---------------------------------------------------------------------------
# Principal Angles
# ---------------------------------------------------------------------------

def compute_principal_angles(
    W1: torch.Tensor,
    W2: torch.Tensor,
    top_k: int = 32,
) -> dict:
    """
    Compute the principal angles between the column spaces of two weight matrices.

    Given W1, W2 ∈ ℝ^{m × n}, this function measures how "aligned" the two
    linear subspaces (spanned by the columns of W1 and W2) are.

    Algorithm (Björck & Golub, 1973)
    ---------------------------------
    1. Compute economy SVD:  W1 = U1 Σ1 V1ᵀ,  W2 = U2 Σ2 V2ᵀ
       → U1 ∈ ℝ^{m × r1},  U2 ∈ ℝ^{m × r2}   (orthonormal bases)
    2. Keep only the top min(top_k, r1, r2) columns of each basis.
    3. Cross-Gram matrix:   M = U1ᵀ U2  ∈ ℝ^{k × k}
    4. SVD of M:            M = P Σ Qᵀ
       The singular values σᵢ are the cosines of the principal angles θᵢ.
    5. θᵢ = arccos(σᵢ)  (clamped to [0°, 90°] for numerical safety).

    Interpretation
    --------------
    • θᵢ = 0°  → perfectly aligned direction
    • θᵢ = 90° → orthogonal (completely misaligned) direction
    • mean_angle_deg gives an overall alignment score.

    Args:
        W1, W2  : (out_features × in_features) weight tensors.
        top_k   : number of principal angles to compute.

    Returns:
        dict with keys:
            cosines        – ndarray (k,): cosines of principal angles
            angles_deg     – ndarray (k,): principal angles in degrees
            mean_angle_deg – float: average angle (lower = more aligned)
            max_angle_deg  – float: largest angle (most misaligned direction)
    """
    U1, _, _ = torch.linalg.svd(W1.float(), full_matrices=False)
    U2, _, _ = torch.linalg.svd(W2.float(), full_matrices=False)

    k  = min(top_k, U1.shape[1], U2.shape[1])
    U1 = U1[:, :k]
    U2 = U2[:, :k]

    M       = U1.T @ U2                           # (k × k)
    cosines = torch.linalg.svdvals(M).clamp(-1.0, 1.0)
    angles  = torch.acos(cosines) * (180.0 / torch.pi)

    return {
        "cosines":        cosines.numpy(),
        "angles_deg":     angles.numpy(),
        "mean_angle_deg": float(angles.mean()),
        "max_angle_deg":  float(angles.max()),
    }


def compute_principal_angle_cosines(
    W1: torch.Tensor,
    W2: torch.Tensor,
    top_k: int = 32,
) -> torch.Tensor:
    """
    Convenience wrapper: returns only the cosine values as a 1-D tensor.

    These are the diagonal of U1ᵀ U2 after alignment (full SVD version),
    which equals the singular values of the cross-Gram matrix U1ᵀ U2.

    Note: the original utils.py used torch.diag(U1ᵀ U2) which gives the
    *diagonal* of the cross-product, not the true principal-angle cosines.
    The correct approach is torch.linalg.svdvals(U1ᵀ U2), implemented here.
    """
    result = compute_principal_angles(W1, W2, top_k=top_k)
    return torch.tensor(result["cosines"])


# ---------------------------------------------------------------------------
# Legacy alias  (keeps backward compatibility with any existing call sites)
# ---------------------------------------------------------------------------

def compute_principal_angle(W1: torch.Tensor, W2: torch.Tensor) -> torch.Tensor:
    """
    Deprecated.  Use compute_principal_angles() instead.

    Returns cosines of the top-min(rank) principal angles between the
    column spaces of W1 and W2 (corrected from the original diagonal-only
    implementation which did not compute true principal angles).
    """
    return compute_principal_angle_cosines(W1, W2, top_k=min(W1.shape[1], W2.shape[1]))
