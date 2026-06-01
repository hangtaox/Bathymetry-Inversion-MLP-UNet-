import torch
import torch.nn as nn

class DualStreamFusionNet(nn.Module):
    def __init__(self, in_channels=12, patch_size=11, cnn_feat_dim=512, mlp_feat_dim=128, mlp_layers=6):
        super().__init__()
        self.patch_size = patch_size
        self.physics_channels = in_channels - 2 
        
        # ==========================================================
        # 分支 1：主导分支 (CNN) - 负责预测“基础水深”
        # ==========================================================
        self.spatial_cnn = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, patch_size, patch_size)
            flattened_size = self.spatial_cnn(dummy_input).view(1, -1).size(1) 
            
        self.cnn_fc = nn.Sequential(
            nn.Dropout(0.45),
            nn.Flatten(),
            nn.Linear(flattened_size, cnn_feat_dim), 
            nn.ReLU(inplace=True)
        )
        
        # CNN 独立回归器 -> 直接输出预测深度
        self.cnn_regressor = nn.Sequential(
            nn.Linear(cnn_feat_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )

        # ==========================================================
        # 分支 2：辅助分支 (MLP) - 负责预测“物理修正残差”
        # ==========================================================
        self.mlp_input = nn.Linear(self.physics_channels, mlp_feat_dim)
        self.mlp_blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(mlp_feat_dim),
                nn.Linear(mlp_feat_dim, mlp_feat_dim),
                nn.GELU(),
                nn.Dropout(0.15), 
                nn.Linear(mlp_feat_dim, mlp_feat_dim)
            ) for _ in range(mlp_layers)
        ])
        
        # MLP 独立回归器 -> 直接输出修正残差 (Offset)
        self.mlp_regressor = nn.Sequential(
            nn.Linear(mlp_feat_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # --- 1. CNN 主分支前向传播 ---
        cnn_out = self.spatial_cnn(x)           
        cnn_feat = self.cnn_fc(cnn_out)         
        pred_base = self.cnn_regressor(cnn_feat) # 输出: 基础水深 [B, 1]

        # --- 2. MLP 辅助分支前向传播 ---
        center_idx = self.patch_size // 2
        x_center = x[:, :-2, center_idx, center_idx]  
        mlp_feat = self.mlp_input(x_center)
        for block in self.mlp_blocks:
            mlp_feat = mlp_feat + block(mlp_feat)   
        pred_res = self.mlp_regressor(mlp_feat) # 输出: 物理修正残差 [B, 1]

        # --- 3. 决策级残差融合 (极简、高效、无冲突) ---
        # 最终水深 = CNN基础预测 + MLP细节修正
        pred_depth = pred_base + pred_res 

        return pred_depth.squeeze(1)