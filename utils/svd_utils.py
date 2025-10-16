# utils/svd_utils.py
import torch

def truncated_svd(W, rank):
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    r = min(rank, S.numel())
    return U[:, :r].contiguous(), S[:r].contiguous(), Vh[:r, :].contiguous()

def svd_head_tail(W, r):
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    p = S.numel()
    r_top = min(r, p)
    r_bot = min(r, p - r_top) if (2*r <= p) else min(r, max(0, p - r_top))
    U_top, S_top, Vh_top = U[:, :r_top], S[:r_top], Vh[:r_top, :]
    U_bot, S_bot, Vh_bot = U[:, p-r_bot:], S[p-r_bot:], Vh[p-r_bot:, :]
    return (U_top, S_top, Vh_top), (U_bot, S_bot, Vh_bot)
