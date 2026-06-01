import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. 掩码卷积层 MaskedConv2d
# ============================================================
# 功能：
# 对输入中的 NaN 区域进行数学隔离。
#
# 输入：
#   x    : 特征图 [B, C, H, W]
#   mask : 有效掩码 [B, 1, H, W]
#          有效海洋区域=1，陆地/NaN区域=0
#
# 输出：
#   out      : 卷积结果
#   new_mask : 更新后的掩码
#
# 原理：
# 普通卷积：
#   y = Σ(w_i * x_i)
#
# 掩码卷积：
#   y = Σ(w_i * x_i * m_i) / Σ(m_i)
#
# 即：
#   无效位置完全不参与卷积计算
# ============================================================

class MaskedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, padding=1):
        super().__init__()

        self.kernel_size = kernel_size
        self.padding = padding

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True
        )

    def forward(self, x, mask):

        # ----------------------------------------------------
        # Step 1：屏蔽无效区域
        # ----------------------------------------------------
        x = x * mask

        # ----------------------------------------------------
        # Step 2：普通卷积
        # ----------------------------------------------------
        out = self.conv(x)

        # ----------------------------------------------------
        # Step 3：统计每个卷积窗口内有效像素数量
        # ----------------------------------------------------
        with torch.no_grad():
            ones_kernel = torch.ones(
                1, 1,
                self.kernel_size,
                self.kernel_size,
                device=x.device
            )

            valid_count = F.conv2d(
                mask,
                ones_kernel,
                padding=self.padding
            )

            # 只要窗口内存在有效值，则输出有效
            new_mask = (valid_count > 0).float()

        # ----------------------------------------------------
        # Step 4：归一化补偿
        # 防止边界处因有效值减少导致响应变小
        # ----------------------------------------------------
        out = out / (valid_count + 1e-8)

        # 再次屏蔽
        out = out * new_mask

        return out, new_mask


# ============================================================
# 2. 掩码最大池化 MaskedMaxPool2d
# ============================================================
#
# 功能：
# 对无效区域赋值 -inf，使其永远不会被选为最大值
#
# 为什么用 MaxPool：
# 保留海底地形突变（海沟、断裂、陡坡）
#
# 不用 AvgPool：
# 会进一步平滑高梯度区域
# ============================================================

class MaskedMaxPool2d(nn.Module):
    def __init__(self, kernel_size=2, stride=2):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x, mask):

        # 无效区域置为负无穷
        x = x.masked_fill(mask == 0, float('-inf'))

        # 最大池化
        pooled_x = F.max_pool2d(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride
        )

        # mask同步池化
        pooled_mask = F.max_pool2d(
            mask,
            kernel_size=self.kernel_size,
            stride=self.stride
        )

        pooled_mask = (pooled_mask > 0).float()

        # 若全窗口无效，则恢复为0
        pooled_x = torch.where(
            pooled_mask.bool(),
            pooled_x,
            torch.zeros_like(pooled_x)
        )

        return pooled_x, pooled_mask


# ============================================================
# 3. 掩码卷积块
# ============================================================
#
# 结构：
#   MaskedConv
#   GroupNorm
#   ReLU
#
# 双流传播：
#   (x, mask) -> (x', mask')
# ============================================================

class MaskedConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, groups):
        super().__init__()

        self.conv = MaskedConv2d(in_ch, out_ch)

        # GroupNorm
        self.norm = nn.GroupNorm(groups, out_ch)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, mask):

        x, mask = self.conv(x, mask)

        x = self.norm(x)

        x = self.relu(x)

        return x, mask


# ============================================================
# 4. 主网络
# ============================================================

class CNNEncoderFC(nn.Module):
    def __init__(self, in_channels=12, patch_size=11):
        super().__init__()

        # ----------------------------------------------------
        # Feature extractor
        # ----------------------------------------------------

        self.block1 = MaskedConvBlock(in_channels, 64, groups=8)
        self.pool1 = MaskedMaxPool2d()

        self.block2 = MaskedConvBlock(64, 128, groups=8)
        self.pool2 = MaskedMaxPool2d()

        self.block3 = MaskedConvBlock(128, 256, groups=16)

        self.block4 = MaskedConvBlock(256, 256, groups=16)

        # ----------------------------------------------------
        # 自动计算flatten维度
        # ----------------------------------------------------
        with torch.no_grad():
            dummy_x = torch.zeros(
                1, in_channels, patch_size, patch_size
            )

            dummy_mask = torch.ones(
                1, 1, patch_size, patch_size
            )

            x, m = self.block1(dummy_x, dummy_mask)
            x, m = self.pool1(x, m)

            x, m = self.block2(x, m)
            x, m = self.pool2(x, m)

            x, m = self.block3(x, m)
            x, m = self.block4(x, m)

            flattened_size = x.view(1, -1).size(1)

        # ----------------------------------------------------
        # Regression head
        # ----------------------------------------------------
        self.regressor = nn.Sequential(
            nn.Dropout(0.35),

            nn.Flatten(),

            nn.Linear(flattened_size, 256),
            nn.ReLU(inplace=True),

            nn.Linear(256, 1)
        )

    # ========================================================
    # forward
    #
    # 输入：
    #   x    : [B, C, H, W]
    #   mask : [B, 1, H, W]
    #
    # 双流传播：
    #   每层同步更新 feature 与 mask
    # ========================================================

    def forward(self, x, mask):

        x, mask = self.block1(x, mask)
        x, mask = self.pool1(x, mask)

        x, mask = self.block2(x, mask)
        x, mask = self.pool2(x, mask)

        x, mask = self.block3(x, mask)

        x, mask = self.block4(x, mask)

        out = self.regressor(x)

        return out.squeeze(1)