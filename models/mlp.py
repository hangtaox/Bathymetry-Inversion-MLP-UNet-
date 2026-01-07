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

        self.input = nn.Linear(in_dim, hidden_dim)

        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
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

# TODO: 用 gMLP block 替换 self.blocks
# TODO: 在 input 前加入 Fourier Features
# TODO: 改成 multi-branch 输入（grav / curv 分支）