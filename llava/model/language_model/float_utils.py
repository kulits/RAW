from torch import nn
from transformers.activations import GELUActivation
import torch


def get_float_head(config):
    if config.float_head_type == "linear":
        return nn.Linear(config.hidden_size, 1, bias=True)
    elif config.float_head_type == "tanh_mlp_gelu":
        return nn.Sequential(
            nn.Tanh(),
            nn.Linear(config.hidden_size, config.hidden_size, bias=True),
            GELUActivation(),
            nn.Linear(config.hidden_size, 1, bias=True),
        )
    else:
        print("Not using a float head:", config.float_head_type)
