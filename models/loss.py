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


class BathymetryLoss(nn.Module):
    def __init__(self, lambda_grad=0.2):
        super().__init__()
        self.lambda_grad = lambda_grad
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        # 深度误差
        loss_depth = self.mse(pred, target)

        # 梯度一致性
        pred_dx, pred_dy = gradient(pred)
        tgt_dx, tgt_dy = gradient(target)

        loss_grad = self.mse(pred_dx, tgt_dx) + self.mse(pred_dy, tgt_dy)

        total_loss = loss_depth + self.lambda_grad * loss_grad
        return total_loss
