import torch
import torch.nn as nn

class CNNEncoderFC(nn.Module):
    def __init__(self, in_channels=12, patch_size=11):
        super().__init__()

        # 1. 卷积特征提取部分 (保持原样)
        self.features = nn.Sequential(
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

        # 2. 动态计算展平后的维度 (PyTorch 标准做法)
        # 我们用一个形状为 [1, in_channels, patch_size, patch_size] 的全零张量过一遍卷积层
        # 看看它出来之后展平是多大
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, patch_size, patch_size)
            dummy_output = self.features(dummy_input)
            # flatten 之后的维度大小 (例如：patch=11 时，这里算出来会自动是 2048)
            flattened_size = dummy_output.view(1, -1).size(1) 

        # 3. 回归器部分：第一层的输入大小使用上面自动计算出的 flattened_size
        self.regressor = nn.Sequential(
            nn.Dropout(0.4),
            nn.Flatten(),
            nn.Linear(flattened_size, 512), # 动态接收 2048 或是其他维度的输入
            nn.ReLU(inplace=True),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        x = self.features(x)   
        return self.regressor(x).squeeze(1)