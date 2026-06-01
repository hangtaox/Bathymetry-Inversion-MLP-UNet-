import torch
import torch.nn as nn
import torch.nn.functional as F


def gradient(x):
    """
    计算空间梯度，用于抑制过平滑
    """
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    return dx, dy


class BathymetryPointLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        # pred, target: [B,1]
        return self.mse(pred, target)

