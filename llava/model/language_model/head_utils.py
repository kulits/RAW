from torch import nn
from transformers.activations import GELUActivation
import torch

POS_SCALE = 0.1
W_ROTATION_MSE = 1
W_APPEARANCE_NORM = 0.001
W_APPEARANCE_COSINE = 10


def get_rotation_head(config):
    return nn.Sequential(
        nn.Tanh(),
        nn.Linear(config.hidden_size, config.hidden_size, bias=True),
        GELUActivation(),
        nn.Linear(config.hidden_size, 9, bias=True),
    )


def get_appearance_head(config):
    return nn.Sequential(
        nn.Tanh(),
        nn.Linear(config.hidden_size, config.hidden_size, bias=True),
        GELUActivation(),
        nn.Linear(config.hidden_size, config.appearance_dim, bias=True),
    )


def symmetric_orthogonalization(x):
    """Maps 9D input vectors onto SO(3) via symmetric orthogonalization.

    x: should have size [batch_size, 9]

    Output has size [batch_size, 3, 3], where each inner 3x3 matrix is in SO(3).
    """
    m = x.view(-1, 3, 3)
    u, s, v = torch.svd(m)
    vt = torch.transpose(v, 1, 2)
    det = torch.det(torch.matmul(u, vt))
    det = det.view(-1, 1, 1)
    vt = torch.cat((vt[:, :2, :], vt[:, -1:, :] * det), 1)
    r = torch.matmul(u, vt)
    return r
