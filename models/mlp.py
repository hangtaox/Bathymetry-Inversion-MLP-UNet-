# models/mlp.py
import torch
import torch.nn as nn

class ResidualMLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 6,
        out_dim: int = 1
    ):
        super().__init__()

        # 动态接收输入维度
        self.input = nn.Linear(in_dim, hidden_dim)

        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(0.15), # 添加一点 Dropout 防止过拟合，与双流对齐
                nn.Linear(hidden_dim, hidden_dim)
            )
            for _ in range(num_layers)
        ])

        self.output = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.input(x)
        for block in self.blocks:
            x = x + block(x)
        return self.output(x)