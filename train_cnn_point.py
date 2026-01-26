import os
import time
import json
from pathlib import Path

import numpy as np
import xarray as xr
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from scipy import ndimage
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

from models.cnn import CNNEncoderFC
from data.dataset_cnn import BathymetryShipPointCNNDataset

# ============================================================
#  训练主程序
# ============================================================

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    os.makedirs("./checkpoints/cnn", exist_ok=True)
    os.makedirs("./tmp_img", exist_ok=True)

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
        patch_size=5,          # ★ 局部感受野
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

    with open("./checkpoints/cnn/normalization_params.json", "w") as f:
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
    model = CNNEncoderFC(in_channels=6).to(device)
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
    print("\n[INFO] Start training...\n")

    for epoch in range(num_epochs):
        t0 = time.time()

        # ---------- Train ----------
        model.train()
        train_loss = 0.0

        for X, y in train_loader:
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
            for X, y in val_loader:
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
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            no_improve = 0
            torch.save(model.state_dict(), "./checkpoints/cnn/best_model.pt")
            print("  ↳ Best model updated")
        else:
            no_improve += 1

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"./checkpoints/cnn/epoch_{epoch+1:03d}.pt")

        if no_improve >= patience:
            print(f"\n[Early Stop] Stop at epoch {epoch+1}")
            break

    # =========================
    # 测试集评估 & 可视化
    # =========================
    print("\n[INFO] Evaluating on test set...")

    model.load_state_dict(torch.load("./checkpoints/cnn/best_model.pt"))
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

    print("\n===== Test Metrics =====")
    print(f"MAE   : {mae:.3f} m")
    print(f"RMSE  : {rmse:.3f} m")
    print(f"Bias  : {bias:.3f} m")
    print(f"R²    : {r2:.4f}")
    print(f"CorrR : {r:.4f}")

    # =========================
    # 散点图
    # =========================
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=3, alpha=0.4)

    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, "r--", lw=1)

    plt.xlabel("Ship depth (m)")
    plt.ylabel("Predicted depth (m)")
    plt.title(f"Prediction vs GT\nRMSE={rmse:.2f} m, R={r:.3f}")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("./tmp_img/cnn_scatter_pred_vs_gt.png", dpi=300)
    plt.show()

    # =========================
    # 残差分布
    # =========================
    residual = y_pred - y_true

    plt.figure(figsize=(6, 4))
    plt.hist(residual, bins=100, density=True, alpha=0.7)
    plt.axvline(0, color="r", linestyle="--", label="Zero")
    plt.xlabel("Residual (Pred - GT) [m]")
    plt.ylabel("Density")
    plt.title(f"Residual Distribution\nBias={bias:.2f} m")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("./tmp_img/cnn_residual_hist.png", dpi=300)
    plt.show()

    # =========================
    # RMSE vs Depth
    # =========================
    bins = np.arange(y_true.min(), y_true.max(), 200)
    rmse_bins = []

    for i in range(len(bins) - 1):
        mask = (y_true >= bins[i]) & (y_true < bins[i + 1])
        if mask.sum() < 50:
            rmse_bins.append(np.nan)
        else:
            rmse_bins.append(
                np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))
            )

    plt.figure(figsize=(7, 4))
    plt.plot(bins[:-1], rmse_bins, marker="o")
    plt.xlabel("Depth bin (m)")
    plt.ylabel("RMSE (m)")
    plt.title("RMSE vs Depth")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("./tmp_img/cnn_rmse_vs_depth.png", dpi=300)
    plt.show()

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
    plt.title("Training History")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("./tmp_img/cnn_train_val_curve.png", dpi=300)
    plt.show()

    print("\n[INFO] Done.")


if __name__ == "__main__":
    train()
