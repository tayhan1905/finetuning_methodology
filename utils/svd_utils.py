# utils/svd_utils.py
import torch

def truncated_svd(W, rank):
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    r = min(rank, S.numel())
    return U[:, :r].contiguous(), S[:r].contiguous(), Vh[:r, :].contiguous()

def svd_head_tail(W, r):
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    p = S.numel()

    # Ensure we don't exceed the available rank
    r_tail = min(r, p)
    r_top = max(p - r_tail, 0)

    # Split head/tail
    U_top, S_top, Vh_top = U[:, :r_top], S[:r_top], Vh[:r_top, :]
    U_tail, S_tail, Vh_tail = U[:, r_top:], S[r_top:], Vh[r_top:, :]

    return (U_top, S_top, Vh_top), (U_tail, S_tail, Vh_tail)

