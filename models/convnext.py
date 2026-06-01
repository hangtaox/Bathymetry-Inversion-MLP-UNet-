import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm2d(nn.Module):
    """
    专门为二维特征图写的 LayerNorm。
    PyTorch 自带的 nn.LayerNorm 默认作用于最后一个维度。
    对于 [N, C, H, W] 的图像数据，我们需要在 C 维度上做归一化。
    """
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        # x shape: [N, C, H, W]
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class ConvNeXtBlock(nn.Module):
    """
    适配数值回归的 ConvNeXt 核心模块
    """
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        # 1. 深度可分离卷积 (Depthwise Conv)
        # kernel_size=7, padding=3 保证特征图长宽完全不变
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        
        # 2. 局部通道 LayerNorm
        self.norm = LayerNorm2d(dim)
        
        # 3. 倒置瓶颈的升维 (Pointwise Conv: 1x1)
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1)
        
        # 4. GELU 激活
        self.act = nn.GELU()
        
        # 5. 倒置瓶颈的降维 (Pointwise Conv: 1x1)
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1)
        
        # 6. 现代 DropPath (取代传统 Dropout)
        # 如果 drop_path 为 0，就相当于 nn.Identity()，原样输出
        from torchvision.ops import StochasticDepth
        self.drop_path = StochasticDepth(drop_path, mode="row") if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input_x = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        # 残差连接：原始输入 + 经过网络处理并可能被 drop_path 的残差
        x = input_x + self.drop_path(x)
        return x

class BathymetryPatchConvNeXt(nn.Module):
    """
    专为 BathymetryShipPointCNNDataset 设计的等分辨率数值回归网络
    """
    def __init__(self, in_channels, hidden_dim=64, num_blocks=4, drop_path_rate=0.1):
        super().__init__()
        
        # ==========================================
        # 模块 1: Stem (特征映射层)
        # ==========================================
        # 作用：将你的物理通道 (重力+lon+lat) 平滑映射到高维特征空间
        # 注意这里不再像原版那样用 stride=4 降采样，而是用 3x3 (或者 1x1) 保持分辨率
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1, stride=1),
            LayerNorm2d(hidden_dim)
        )
        
        # ==========================================
        # 模块 2: 主体特征提取 (等分辨率 ConvNeXt Blocks)
        # ==========================================
        # 构建一个递增的 DropPath 概率列表，越深的层越容易被随机丢弃，防过拟合神技
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, num_blocks)]
        
        blocks = []
        for i in range(num_blocks):
            blocks.append(ConvNeXtBlock(dim=hidden_dim, drop_path=dp_rates[i]))
        self.stages = nn.Sequential(*blocks)
        
        # ==========================================
        # 模块 3: 数值回归头 (Regression Head)
        # ==========================================
        self.head = nn.Sequential(
            # 汇聚空间信息，无论输入是 5x5, 11x11 还是 15x15，统统压成 1x1
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.LayerNorm(hidden_dim), # 回归前最后稳一次分布
            nn.Linear(hidden_dim, 1)  # 输出单一水深预测值
        )

    def forward(self, x):
        # x shape: [Batch, in_channels, patch_size, patch_size]
        x = self.stem(x)
        x = self.stages(x)
        x = self.head(x)
        return x
