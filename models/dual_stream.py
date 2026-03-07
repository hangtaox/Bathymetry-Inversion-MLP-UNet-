import torch
import torch.nn as nn

class DualStreamFusionNet(nn.Module):
    def __init__(self, in_channels=8, patch_size=5, cnn_feat_dim=128, mlp_feat_dim=128, mlp_layers=4):
        super().__init__()
        self.patch_size = patch_size
        
        # 假设最后两个通道是经纬度，这里计算纯物理通道的数量
        # 如果你的数据集有变化，请确保 physics_channels 等于去除坐标后的通道数
        self.physics_channels = in_channels - 2 
        
        # ==========================================================
        # 分支 1：空间形态特征提取流 (Spatial-CNN Branch)
        # ==========================================================
        self.spatial_cnn = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 使用膨胀卷积扩大感受野，但不降低分辨率
            nn.Conv2d(64, 128, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # 【关键修复】：引入全局平均池化，浓缩空间形态，彻底解决参数爆炸与过拟合
            nn.AdaptiveAvgPool2d(1) 
        )
        
        self.cnn_fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, cnn_feat_dim), # 输入维度从 6400 降回 256
            nn.ReLU(inplace=True)
        )

        # ==========================================================
        # 分支 2：物理中心特征映射流 (Physical-MLP Branch)
        # ==========================================================
        # 【关键修复】：输入维度仅为纯物理通道，严禁传入坐标数据
        self.mlp_input = nn.Linear(self.physics_channels, mlp_feat_dim)
        self.mlp_blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(mlp_feat_dim),
                nn.Linear(mlp_feat_dim, mlp_feat_dim),
                nn.GELU(),
                nn.Dropout(0.1), # 增加轻量级 Dropout 防止内部过拟合
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
        # CNN 分支可以看到所有信息（包含坐标），用于建立空间方位感
        cnn_out = self.spatial_cnn(x)           
        cnn_feat = self.cnn_fc(cnn_out)         

        # --- 2. 提取物理中心特征 ---
        # 精确截取张量正中心点，且【切片剥离最后两个坐标通道】
        center_idx = self.patch_size // 2
        x_center = x[:, :-2, center_idx, center_idx]  # 输出: [B, physics_channels]

        # 残差 MLP 传播
        mlp_feat = self.mlp_input(x_center)
        for block in self.mlp_blocks:
            mlp_feat = mlp_feat + block(mlp_feat)   

        # --- 3. 融合与回归 ---
        fused_feat = torch.cat([cnn_feat, mlp_feat], dim=1) 
        pred_depth = self.regressor(fused_feat) 

        return pred_depth.squeeze(1)