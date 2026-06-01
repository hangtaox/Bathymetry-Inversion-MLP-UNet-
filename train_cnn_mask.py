import os
import time
import json
from pathlib import Path

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
from tqdm import tqdm
import matplotlib.pyplot as plt

from models.mask_cnn import CNNEncoderFC
from data.dataset_maskcnn import BathymetryShipPointCNNDataset

# =========================================================
# 【全局控制】统一控制感受野大小，方便随时修改
# =========================================================
PATCH_SIZE = 11  

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    # 动态创建包含 patch_size 的专属文件夹
    CKPT_DIR = f"./checkpoints/mask_cnn/{PATCH_SIZE}"
    RESULT_DIR = f"./result/_mask_cnn/{PATCH_SIZE}"
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)

    # =========================
    # 数据路径 & 研究区 (已与双流网络 100% 对齐)
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

    LON_RANGE = (104, 122)
    LAT_RANGE = (0, 26)

    # =========================
    # 构建 Dataset
    # =========================
    dataset = BathymetryShipPointCNNDataset(
        feature_paths=PATHS,
        ship_nc_path=SHIP_NC,
        aligned_ship_nc_path=ALIGNED_SHIP_NC,
        lon_range=LON_RANGE,
        lat_range=LAT_RANGE,
        patch_size=PATCH_SIZE,  # 动态传入       
        normalize=True,
    )

    print(f"[INFO] Total samples: {len(dataset)}")

    # =========================
    # 保存归一化参数
    # =========================
    norm_params = {
        "y_mean": float(dataset.y_mean),
        "y_std": float(dataset.y_std),
        "x_means": dataset.X_mean.squeeze().tolist(),
        "x_stds": dataset.X_std.squeeze().tolist(),
    }

    with open(f"{CKPT_DIR}/normalization_params.json", "w") as f:
        json.dump(norm_params, f, indent=4)

    print(f"[INFO] Normalization parameters saved to {CKPT_DIR}")

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

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_set, batch_size=128, shuffle=False)
    test_loader  = DataLoader(test_set, batch_size=128, shuffle=False)

    # =========================
    # 模型 & 优化器
    # =========================
    # 动态适应输入通道数
    model = CNNEncoderFC(
            in_channels=dataset.X.shape[1], 
            patch_size=PATCH_SIZE
        ).to(device)
    num_epochs = 100

    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=2e-2)
    
    # 替换为余弦退火调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-5
    )

    patience = 15  # 稍微调大，配合余弦退火的平滑下降
    best_val = np.inf
    min_delta = 5e-6 # 新增：验证集实质性进步的最小阈值
    best_epoch = -1
    no_improve = 0
    train_hist, val_hist = [], []

    # =========================
    # 训练循环
    # =========================
    print("\n[INFO] Start training Baseline CNN...\n")

    for epoch in range(num_epochs):
        t0 = time.time()
        model.train()
        train_loss = 0.0

        for X, mask, y in tqdm(train_loader, desc=f"Epoch {epoch+1:03d} [Train]", leave=False):
            X = X.to(device)
            mask = mask.to(device)
            y = y.to(device).squeeze(1)

            optimizer.zero_grad()
            pred = model(X, mask)

            loss = criterion(pred, y)
            loss.backward()
            # 梯度裁剪，限制最大梯度范数为 1.0，防止梯度爆炸和异常尖刺
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * len(y)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        y_true_list, y_pred_list = [], []

        with torch.no_grad():
            for X, mask, y in tqdm(val_loader, desc=f"Epoch {epoch+1:03d} [Val]", leave=False):
                X = X.to(device)
                mask = mask.to(device)
                y = y.to(device).squeeze(1)

                pred = model(X, mask)
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

        # 模型保存
        if val_loss < (best_val - min_delta):
            best_val = val_loss
            best_epoch = epoch
            no_improve = 0
            torch.save(model.state_dict(), f"{CKPT_DIR}/best_model.pt")
            print("  ↳ Best model updated")
        else:
            no_improve += 1

        # 每 10 轮保险保存
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"{CKPT_DIR}/epoch_{epoch+1:03d}.pt")
            print(f"  ↳ Periodic save at epoch {epoch+1}")

        if no_improve >= patience:
            print(f"\n[Early Stop] Stop at epoch {epoch+1}")
            break

    # =========================
    # 数据提取与保存 (供推理代码画图)
    # =========================
    print("\n[INFO] Extracting Test Set predictions for analysis...")
    
    # 消除安全警告
    model.load_state_dict(torch.load(f"{CKPT_DIR}/best_model.pt", weights_only=True))
    model.eval()

    y_true_list, y_pred_list = [], []
    lon_norm_list, lat_norm_list = [], [] 
    
    # 动态计算中心点索引
    center_idx = PATCH_SIZE // 2 

    with torch.no_grad():
        for X, mask, y in test_loader:
            X = X.to(device)
            mask = mask.to(device)
            y = y.to(device)
            
            pred = model(X, mask)
            
            y_true_list.append(y.cpu().numpy())
            y_pred_list.append(pred.cpu().numpy())
            
            # 从 CNN 输入的 2D 特征中提取中心点的经纬度
            lon_norm_list.append(X[:, -2, center_idx, center_idx].cpu().numpy())
            lat_norm_list.append(X[:, -1, center_idx, center_idx].cpu().numpy())

    # 反归一化水深
    y_true_norm = np.concatenate(y_true_list).squeeze()
    y_pred_norm = np.concatenate(y_pred_list).squeeze()
    y_true_real = dataset.inverse_transform_y(y_true_norm)
    y_pred_real = dataset.inverse_transform_y(y_pred_norm)
    
    # 反归一化坐标
    lon_norm = np.concatenate(lon_norm_list).squeeze()
    lat_norm = np.concatenate(lat_norm_list).squeeze()
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
    plt.title(f"Baseline CNN Training History (Patch: {PATCH_SIZE})")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{RESULT_DIR}/cnn_train_val_curve.png", dpi=300)
    plt.close()
    
    print("\n[INFO] CNN Training complete! You can now analyze it using your evaluation scripts.")

if __name__ == "__main__":
    train()