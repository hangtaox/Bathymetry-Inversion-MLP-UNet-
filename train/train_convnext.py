import sys
import os
import time
import json
from pathlib import Path

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision.ops import StochasticDepth
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 导入你的 Dataset
from data.dataset_cnn import BathymetryShipPointCNNDataset
from models.convnext import BathymetryPatchConvNeXt

# =========================================================
# 【全局控制】统一控制感受野大小
# =========================================================
PATCH_SIZE = 11  

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    # 动态创建 ConvNeXt 专属文件夹 (避免与 CNN 冲突)
    CKPT_DIR = f"./checkpoints/convnext/{PATCH_SIZE}"
    RESULT_DIR = f"./result/convnext/{PATCH_SIZE}"
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)

    # =========================
    # 数据路径 (保持原样)
    # =========================
    PATHS = {
        "grav": "./data/processed/G_ocean_raw.nc",
        "curv": "./data/processed/VGG_ocean_raw.nc",
        "b_lp": "./data/processed/B_LP.nc",
        "g_lp": "./data/processed/G_LP.nc",
        "g_bp": "./data/processed/G_BP.nc",
        "vgg_bp": "./data/processed/VGG_BP.nc",
        "g_east": "./data/processed/G_East.nc",
        "g_north": "./data/processed/G_North.nc",
        "g_east_bp": "./data/processed/G_East_BP.nc",
        "g_north_bp": "./data/processed/G_North_BP.nc",
    }
    SHIP_NC = "./data/ship_combined_calibrated.nc"
    ALIGNED_SHIP_NC = "./data/ship_bathymetry_points_aligned.nc"
    LON_RANGE = (105, 125)
    LAT_RANGE = (0, 30)

    # =========================
    # 构建 Dataset
    # =========================
    dataset = BathymetryShipPointCNNDataset(
        feature_paths=PATHS,
        ship_nc_path=SHIP_NC,
        aligned_ship_nc_path=ALIGNED_SHIP_NC,
        lon_range=LON_RANGE,
        lat_range=LAT_RANGE,
        patch_size=PATCH_SIZE,      
        normalize=True,
    )
    print(f"[INFO] Total samples: {len(dataset)}")

    # 保存归一化参数
    norm_params = {
        "y_mean": float(dataset.y_mean),
        "y_std": float(dataset.y_std),
        "x_means": dataset.X_mean.squeeze().tolist(),
        "x_stds": dataset.X_std.squeeze().tolist(),
    }
    with open(f"{CKPT_DIR}/normalization_params.json", "w") as f:
        json.dump(norm_params, f, indent=4)

    # =========================
    # 数据集划分 8:1:1
    # =========================
    N = len(dataset)
    n_train = int(0.8 * N)
    n_val = int(0.1 * N)
    n_test = N - n_train - n_val

    train_set, val_set, test_set = random_split(
        dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"[INFO] Train/Val/Test = {len(train_set)}/{len(val_set)}/{len(test_set)}")

    # ConvNeXt 对 Batch Size 的宽容度很高，64 和 128 都是极佳选择
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_set, batch_size=128, shuffle=False)
    test_loader  = DataLoader(test_set, batch_size=128, shuffle=False)

    # =========================
    # 模型 & 优化器
    # =========================
    # in_channels 自动适配你的特征字典数量 + 2 (lon, lat)
    model = BathymetryPatchConvNeXt(
        in_channels=dataset.X.shape[1], 
        hidden_dim=64,     # 隐藏层通道数，控制模型体量
        num_blocks=4,      # 核心块的数量
        drop_path_rate=0.1 # 抗空间自相关过拟合的概率
    ).to(device)
    
    num_epochs = 100
    criterion = nn.SmoothL1Loss()
    
    # ⚠️ 注意：现代架构推荐使用 AdamW 而不是标准 Adam，weight_decay 稍微调大
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

    patience = 15  
    best_val = np.inf
    min_delta = 5e-6 
    best_epoch = -1
    no_improve = 0
    train_hist, val_hist = [], []

    # =========================
    # 训练循环
    # =========================
    print("\n[INFO] Start training Patch-to-Point ConvNeXt...\n")

    for epoch in range(num_epochs):
        t0 = time.time()
        model.train()
        train_loss = 0.0

        for X, y in tqdm(train_loader, desc=f"Epoch {epoch+1:03d} [Train]", leave=False):
            X = X.to(device)
            y = y.to(device).squeeze(1)

            optimizer.zero_grad()
            pred = model(X).squeeze(1) # 模型输出 [Batch, 1], 压缩为 [Batch]
            loss = criterion(pred, y)
            loss.backward()
            
            # 梯度裁剪依然保留，防止重力异常场中的极端尖刺数据
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * len(y)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        y_true_list, y_pred_list = [], []

        with torch.no_grad():
            for X, y in tqdm(val_loader, desc=f"Epoch {epoch+1:03d} [Val]", leave=False):
                X = X.to(device)
                y = y.to(device).squeeze(1)

                pred = model(X).squeeze(1)
                val_loss += criterion(pred, y).item() * len(y)

                y_true_list.append(y.cpu().numpy())
                y_pred_list.append(pred.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        scheduler.step()

        # 指标计算
        y_true = dataset.inverse_transform_y(np.concatenate(y_true_list))
        y_pred = dataset.inverse_transform_y(np.concatenate(y_pred_list))
        
        val_mae = mean_absolute_error(y_true, y_pred)
        val_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        val_std = np.std(y_pred - y_true)          
        val_corr = pearsonr(y_true, y_pred)[0]     

        lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - t0

        train_hist.append(train_loss)
        val_hist.append(val_loss)

        print(
            f"Epoch {epoch+1:03d} | "
            f"Time {epoch_time:.1f}s | "
            f"Train {train_loss:.5f} | "
            f"Val {val_loss:.5f} | "
            f"MAE {val_mae:.1f} m | "
            f"RMSE {val_rmse:.1f} m | "
            f"STD {val_std:.1f} m | "
            f"Corr {val_corr:.3f} | "
            f"LR {lr:.1e}"
        )

        if val_loss < (best_val - min_delta):
            best_val = val_loss
            best_epoch = epoch
            no_improve = 0
            torch.save(model.state_dict(), f"{CKPT_DIR}/best_model.pt")
            print("  ↳ Best model updated")
        else:
            no_improve += 1

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"{CKPT_DIR}/epoch_{epoch+1:03d}.pt")

        if no_improve >= patience:
            print(f"\n[Early Stop] Stop at epoch {epoch+1}")
            break

    # =========================
    # 数据提取与保存
    # =========================
    print("\n[INFO] Extracting Test Set predictions for analysis...")
    model.load_state_dict(torch.load(f"{CKPT_DIR}/best_model.pt", weights_only=True))
    model.eval()

    y_true_list, y_pred_list = [], []
    lon_norm_list, lat_norm_list = [], [] 
    center_idx = PATCH_SIZE // 2 

    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            y = y.to(device).squeeze(1)
            pred = model(X).squeeze(1)
            
            y_true_list.append(y.cpu().numpy())
            y_pred_list.append(pred.cpu().numpy())
            
            # 由于 dataset 里最后两个通道是 lon 和 lat，这里提取依然完美成立
            lon_norm_list.append(X[:, -2, center_idx, center_idx].cpu().numpy())
            lat_norm_list.append(X[:, -1, center_idx, center_idx].cpu().numpy())

    y_true_norm = np.concatenate(y_true_list)
    y_pred_norm = np.concatenate(y_pred_list)
    y_true_real = dataset.inverse_transform_y(y_true_norm)
    y_pred_real = dataset.inverse_transform_y(y_pred_norm)
    
    lon_norm = np.concatenate(lon_norm_list)
    lat_norm = np.concatenate(lat_norm_list)
    lon_mean, lon_std = norm_params["x_means"][-2], norm_params["x_stds"][-2]
    lat_mean, lat_std = norm_params["x_means"][-1], norm_params["x_stds"][-1]
    
    lon_real = lon_norm * lon_std + lon_mean
    lat_real = lat_norm * lat_std + lat_mean

    test_df = pd.DataFrame({
        "lon": lon_real,
        "lat": lat_real,
        "y_true": y_true_real,
        "y_pred": y_pred_real
    })
    
    test_csv_path = f"{CKPT_DIR}/test_results.csv"
    test_df.to_csv(test_csv_path, index=False)
    print(f"[INFO] Test set predictions saved to {test_csv_path}")

    # =========================
    # 训练历史曲线
    # =========================
    plt.figure(figsize=(8, 5))
    plt.plot(train_hist, label="Train")
    plt.plot(val_hist, label="Val")
    plt.axvline(best_epoch, linestyle="--", color="r", label="Best")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"ConvNeXt Training History (Patch: {PATCH_SIZE})")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{RESULT_DIR}/convnext_train_val_curve.png", dpi=300)
    plt.close()
    
    print("\n[INFO] ConvNeXt Training complete!")

if __name__ == "__main__":
    train()