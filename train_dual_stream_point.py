import os
import time
import json
from pathlib import Path

import pandas as pd
import numpy as np
import xarray as xr
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from scipy import ndimage
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
from tqdm import tqdm

import matplotlib.pyplot as plt

# 【修改1】引入新的双流网络模型
# 确保你已经将上一次生成的双流网络代码保存为 models/dual_stream.py
from models.dual_stream import DualStreamFusionNet 
from data.dataset_cnn import BathymetryShipPointCNNDataset

# ============================================================
#  深度分箱统计函数, 按真实水深分箱，计算各区间统计量
# ============================================================
def depth_binned_statistics(y_true, y_pred, bin_size=200, min_points=30):
    bins = np.arange(y_true.min(), y_true.max() + bin_size, bin_size)

    stats = {
        "depth_min": [],
        "depth_max": [],
        "depth_center": [],
        "Number": [],
        "MAE": [],
        "RMSE": [],
        "STD": [],
        "Bias": [],
    }

    residual = y_pred - y_true

    for i in range(len(bins) - 1):
        mask = (y_true >= bins[i]) & (y_true < bins[i + 1])
        n = mask.sum()

        depth_center = 0.5 * (bins[i] + bins[i + 1])
        stats["depth_min"].append(bins[i])
        stats["depth_max"].append(bins[i + 1])
        stats["depth_center"].append(depth_center)
        stats["Number"].append(n)

        if n < min_points:
            stats["MAE"].append(np.nan)
            stats["RMSE"].append(np.nan)
            stats["STD"].append(np.nan)
            stats["Bias"].append(np.nan)
        else:
            r = residual[mask]
            stats["MAE"].append(np.mean(np.abs(r)))
            stats["RMSE"].append(np.sqrt(np.mean(r**2)))
            stats["STD"].append(np.std(r))
            stats["Bias"].append(np.mean(r))

    return stats

# ============================================================
#  训练主程序
# ============================================================

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    # 【修改2】修改 checkpoint 保存文件夹，避免覆盖 cnn 的模型
    os.makedirs("./checkpoints/dual_stream", exist_ok=True)
    os.makedirs("./tmp_img", exist_ok=True)
    os.makedirs("./result", exist_ok=True)

    # =========================
    # 数据路径 & 研究区
    # =========================
    PATHS = {
        "grav": "./data/processed/G_ocean_raw.nc",
        "curv": "./data/processed/VGG_ocean_raw.nc",
        "b_lp": "./data/processed/B_LP.nc",
        "g_lp": "./data/processed/G_LP.nc",
        "g_bp": "./data/processed/G_BP.nc",
        "vgg_bp": "./data/processed/VGG_BP.nc",
    }

    SHIP_NC = "./ship_bathymetry_points.nc"
    ALIGNED_SHIP_NC = "./ship_bathymetry_points_aligned.nc"

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
        patch_size=5,          # ★ 局部感受野，与双流网络默认的 5x5 一致
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

    # 【修改3】重命名归一化参数文件路径
    with open("./checkpoints/dual_stream/normalization_params.json", "w") as f:
        json.dump(norm_params, f, indent=4)

    print("[INFO] Normalization parameters saved")
    print(f"       y_mean={norm_params['y_mean']:.2f}, y_std={norm_params['y_std']:.2f}")

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
    # 【修改4】实例化新的双流融合网络
    # 注意这里传入 in_channels 以及对应的 patch_size
    model = DualStreamFusionNet(
        in_channels=dataset.X.shape[1], 
        patch_size=5
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # =========================
    # 训练参数
    # =========================
    num_epochs = 120
    patience = 15

    best_val = np.inf
    best_epoch = -1
    no_improve = 0

    train_hist, val_hist = [], []

    # =========================
    # 训练循环
    # =========================
    print("\n[INFO] Start training Dual-Stream Network...\n")

    for epoch in range(num_epochs):
        t0 = time.time()

        # ---------- Train ----------
        model.train()
        train_loss = 0.0

        for X, y in tqdm(
            train_loader,
            desc=f"Epoch {epoch+1:03d} [Train]",
            leave=False
        ):
            X = X.to(device)
            y = y.to(device).squeeze(1)

            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(y)

        train_loss /= len(train_loader.dataset)

        # ---------- Validation ----------
        model.eval()
        val_loss = 0.0
        y_true_list, y_pred_list = [], []

        with torch.no_grad():
            for X, y in tqdm(
                val_loader,
                desc=f"Epoch {epoch+1:03d} [Val]",
                leave=False
            ):
                X = X.to(device)
                y = y.to(device).squeeze(1)

                pred = model(X)
                val_loss += criterion(pred, y).item() * len(y)

                y_true_list.append(y.cpu().numpy())
                y_pred_list.append(pred.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)

        # ---------- 反归一化指标 ----------
        y_true = dataset.inverse_transform_y(np.concatenate(y_true_list))
        y_pred = dataset.inverse_transform_y(np.concatenate(y_pred_list))

        val_mae = mean_absolute_error(y_true, y_pred)
        val_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
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
            f"Corr {val_corr:.3f} | "
            f"LR {lr:.1e}"
        )

        # ---------- 保存模型 ----------
        # 【修改5】保存模型路径修改
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            no_improve = 0
            torch.save(model.state_dict(), "./checkpoints/dual_stream/best_model.pt")
            print("  ↳ Best model updated")
        else:
            no_improve += 1

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"./checkpoints/dual_stream/epoch_{epoch+1:03d}.pt")

        if no_improve >= patience:
            print(f"\n[Early Stop] Stop at epoch {epoch+1}")
            break

    # =========================
    # 测试集评估 & 可视化
    # =========================
    print("\n[INFO] Evaluating on test set...")

    model.load_state_dict(torch.load("./checkpoints/dual_stream/best_model.pt"))
    model.eval()

    y_true_list, y_pred_list = [], []

    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            y = y.to(device)

            pred = model(X)
            y_true_list.append(y.cpu().numpy())
            y_pred_list.append(pred.cpu().numpy())

    y_true_norm = np.concatenate(y_true_list).squeeze()
    y_pred_norm = np.concatenate(y_pred_list).squeeze()

    y_true = dataset.inverse_transform_y(y_true_norm)
    y_pred = dataset.inverse_transform_y(y_pred_norm)

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    bias = np.mean(y_pred - y_true)
    r2   = r2_score(y_true, y_pred)
    r, _ = pearsonr(y_true, y_pred)
    std = np.std(y_pred - y_true)

    print("\n===== Test Metrics (Dual-Stream) =====")
    print(f"MAE   : {mae:.3f} m")
    print(f"RMSE  : {rmse:.3f} m")
    print(f"STD   : {std:.3f} m")
    print(f"Bias  : {bias:.3f} m")
    print(f"R²    : {r2:.4f}")
    print(f"CorrR : {r:.4f}")

    depth_stats = depth_binned_statistics(
        y_true,
        y_pred,
        bin_size=200,
        min_points=50
    )

    df_depth = pd.DataFrame(depth_stats)
    # 【修改6】输出CSV文件名修改
    df_depth.to_csv(
        "./result/dual_depth_binned_statistics.csv",
        index=False
    )

    print("\n[INFO] Depth-binned statistics saved.")

    # =========================
    # 散点图
    # =========================
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=3, alpha=0.4)

    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, "r--", lw=1)

    plt.xlabel("Ship depth (m)")
    plt.ylabel("Predicted depth (m)")
    # 【修改】在 title 中加入了 STD={std:.2f} m
    plt.title(f"Dual-Stream Pred vs GT\nRMSE={rmse:.2f} m, STD={std:.2f} m, R={r:.3f}")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    # 【修改7】所有图片名字修改
    plt.savefig("./result/dual_scatter_pred_vs_gt.png", dpi=300)
    plt.close()

    # =========================
    # 残差分布
    # =========================
    residual = y_pred - y_true

    plt.figure(figsize=(6, 4))
    plt.hist(residual, bins=100, density=True, alpha=0.7)
    plt.axvline(0, color="r", linestyle="--", label="Zero")
    plt.xlabel("Residual (Pred - GT) [m]")
    plt.ylabel("Density")
    plt.title(f"Dual-Stream Residual Distribution\nBias={bias:.2f} m")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("./result/dual_residual_hist.png", dpi=300)
    plt.close()

    # =========================
    # RMSE vs Depth
    # =========================
    plt.figure(figsize=(8, 6))

    plt.plot(df_depth["depth_center"], df_depth["RMSE"], label="RMSE", marker="o")
    plt.plot(df_depth["depth_center"], df_depth["MAE"],  label="MAE",  marker="s")
    plt.plot(df_depth["depth_center"], df_depth["STD"],  label="STD",  marker="^")
    plt.plot(df_depth["depth_center"], df_depth["Bias"], label="Bias", marker="x")

    plt.xlabel("Depth (m)")
    plt.ylabel("Error (m)")
    plt.title("Dual-Stream Error Statistics vs Depth")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("./result/dual_error_vs_depth.png", dpi=300)
    plt.close()

    # =========================
    # 训练曲线
    # =========================
    plt.figure(figsize=(8, 5))
    plt.plot(train_hist, label="Train")
    plt.plot(val_hist, label="Val")
    plt.axvline(best_epoch, linestyle="--", color="r", label="Best")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Dual-Stream Training History")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("./tmp_img/dual_train_val_curve.png", dpi=300)
    plt.close()

    print("\n[INFO] Done. All Dual-Stream results saved successfully.")


if __name__ == "__main__":
    train()