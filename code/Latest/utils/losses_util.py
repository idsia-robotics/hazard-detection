import torch
from torch.nn.functional import mse_loss, l1_loss

losses_list = ["mse", "mae"]


def compute_losses(y_true: torch.Tensor, y_pred: torch.Tensor):

    mse = mse_loss(y_true, y_pred)
    mae = l1_loss(y_true, y_pred)
    return [mse, mae]

