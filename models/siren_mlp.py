# models/mlp.py
import torch
import torch.nn as nn
import numpy as np

class SineLayer(nn.Module):
    """
    SIREN 的核心层：带有特殊初始化权重的正弦激活层
    """
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30.0):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                # 第一层需要更大的均匀分布来覆盖整个频域
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                # 后续层使用特殊的频率初始化
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega_0, 
                     np.sqrt(6 / self.in_features) / self.omega_0
                )
        
    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))

class SIREN_MLP(nn.Module):
    """
    纯粹的 SIREN 网络专家
    """
    def __init__(self, in_dim: int, hidden_dim: int = 128, num_layers: int = 5, out_dim: int = 1, omega_0: float = 30.0):
        super().__init__()
        self.net = nn.ModuleList()
        
        # 第一层 (必须标记为 is_first=True)
        self.net.append(SineLayer(in_dim, hidden_dim, is_first=True, omega_0=omega_0))
        
        # 中间隐藏层
        for _ in range(num_layers - 1):
            self.net.append(SineLayer(hidden_dim, hidden_dim, is_first=False, omega_0=omega_0))
            
        # 最后一层：不带正弦激活，直接输出深度
        final_linear = nn.Linear(hidden_dim, out_dim)
        with torch.no_grad():
            final_linear.weight.uniform_(
                -np.sqrt(6 / hidden_dim) / omega_0, 
                 np.sqrt(6 / hidden_dim) / omega_0
            )
        self.net.append(final_linear)
        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        return self.net(x)

class DualHeadSirenNet(nn.Module):
    """
    自适应双专家路由网络 (Dynamic Routing Two-Headed SIREN)
    """
    def __init__(self, in_channels: int, hidden_dim: int = 128, num_layers: int = 5, omega_0: float = 30.0):
        super().__init__()
        
        # 专家 1：深海专家 (吃 3x3 展平数据)
        self.siren_3x3 = SIREN_MLP(
            in_dim=in_channels * 9, 
            hidden_dim=hidden_dim, 
            num_layers=num_layers, 
            out_dim=1, 
            omega_0=omega_0
        )
        
        # 专家 2：海岸线专家 (只吃中心 1x1 单点数据，防止 NaN 崩溃)
        self.siren_1x1 = SIREN_MLP(
            in_dim=in_channels * 1, 
            hidden_dim=hidden_dim, 
            num_layers=num_layers, 
            out_dim=1, 
            omega_0=omega_0
        )

    def forward(self, x_patch, mask_patch):
        """
        x_patch: [B, C, 3, 3] 物理特征 + 经纬度
        mask_patch: [B, 1, 3, 3] 有效掩码 (1 为海洋，0 为陆地 NaN)
        """
        # 提取中心单点特征 [B, C]
        x_center = x_patch[:, :, 1, 1] 
        
        # 统计每个 3x3 patch 里的有效点数量
        valid_counts = mask_patch.sum(dim=(1, 2, 3)) # Shape: [B]
        
        # 物理切片条件
        is_pure_ocean = (valid_counts == 9)  # 完美的深海，无 NaN
        is_coast = ~is_pure_ocean            # 边缘海岸线，包含 NaN
        
        # 初始化结果张量
        pred_siren = torch.zeros(x_patch.size(0), device=x_patch.device)

        # ==========================================================
        # 路径 A：纯海域 (零 NaN，物理切片送入 3x3 高频专家)
        # ==========================================================
        if is_pure_ocean.any():
            ocean_patches = x_patch[is_pure_ocean] 
            ocean_flat = ocean_patches.reshape(ocean_patches.size(0), -1) # 展平 [N, C*9]
            pred_siren[is_pure_ocean] = self.siren_3x3(ocean_flat).squeeze(-1)

        # ==========================================================
        # 路径 B：海岸带 (含有 NaN，物理切片提取单点送入 1x1 专家)
        # ==========================================================
        if is_coast.any():
            coast_centers = x_center[is_coast] # 只取中心有效点 [M, C]
            pred_siren[is_coast] = self.siren_1x1(coast_centers).squeeze(-1)

        return pred_siren