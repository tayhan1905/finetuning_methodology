import torch
import torch.nn.functional as F
import numpy as np

def compute_principal_angle(W1, W2):
    U1, S1, V1 = torch.svd(W1)
    U2, S2, V2 = torch.svd(W2)

    principal_angle_cos = torch.diag(U1.T @ U2)

    return principal_angle_cos

