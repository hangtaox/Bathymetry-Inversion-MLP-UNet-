import torch
import torch.nn as nn

class DualStreamFusionNet(nn.Module):
    def __init__(self, in_channels=8, patch_size=11, cnn_feat_dim=128, mlp_feat_dim=128, mlp_layers=4):
        super().__init__()
        self.patch_size = patch_size
        
        # 假设最后两个通道是经纬度，计算纯物理通道的数量
        self.physics_channels = in_channels - 2 
        
        # ==========================================================
        # 分支 1：空间形态特征提取流 (Spatial-CNN Branch) - 经典收缩架构
        # ==========================================================
        self.spatial_cnn = nn.Sequential(
            # 第1层：3x3卷积，无填充 (Padding=0)。
            # 如果输入是 11x11，输出变为 9x9；如果是 15x15，输出变为 13x13
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 第2层：3x3卷积，无填充。
            # 如果输入是 9x9，输出变为 7x7；如果是 13x13，输出变为 11x11
            nn.Conv2d(64, 128, kernel_size=3, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # 第3层：3x3卷积，无填充。
            # 如果输入是 7x7，输出变为 5x5；如果是 11x11，输出变为 9x9
            nn.Conv2d(128, 256, kernel_size=3, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # 【终极统一层】：自适应全局平均池化
            # 不管上面出来的特征图是 5x5 还是 9x9，统统一把浓缩成 1x1 的中心特征！
            nn.AdaptiveAvgPool2d(1) 
        )
        
        self.cnn_fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, cnn_feat_dim), 
            nn.ReLU(inplace=True)
        )

        # ==========================================================
        # 分支 2：物理中心特征映射流 (Physical-MLP Branch)
        # ==========================================================
        self.mlp_input = nn.Linear(self.physics_channels, mlp_feat_dim)
        self.mlp_blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(mlp_feat_dim),
                nn.Linear(mlp_feat_dim, mlp_feat_dim),
                nn.GELU(),
                nn.Dropout(0.1), 
                nn.Linear(mlp_feat_dim, mlp_feat_dim)
            ) for _ in range(mlp_layers)
        ])

        # ==========================================================
        # 分支 3：特征融合与联合回归层
        # ==========================================================
        self.regressor = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(cnn_feat_dim + mlp_feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # --- 1. 提取空间形态特征 ---
        cnn_out = self.spatial_cnn(x)           
        cnn_feat = self.cnn_fc(cnn_out)         

        # --- 2. 提取物理中心特征 ---
        # 自动计算中心点索引 (如 patch=11，中心索引为 5；patch=15，中心索引为 7)
        # 这个设计保证了无论你怎么改 patch_size，MLP 永远精准抓取最中间的像素！
        center_idx = self.patch_size // 2
        x_center = x[:, :-2, center_idx, center_idx]  

        # 残差 MLP 传播
        mlp_feat = self.mlp_input(x_center)
        for block in self.mlp_blocks:
            mlp_feat = mlp_feat + block(mlp_feat)   

        # --- 3. 融合与回归 ---
        fused_feat = torch.cat([cnn_feat, mlp_feat], dim=1) 
        pred_depth = self.regressor(fused_feat) 

        return pred_depth.squeeze(1)