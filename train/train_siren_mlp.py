import torch
from torch.utils.data import DataLoader, random_split
import numpy as np
import time
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import os
import sys
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr

# -------------------------
# 路径设置
# -------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# 引入我们刚才重构好的双专家 SIREN 网络
from models.siren_mlp import DualHeadSirenNet
from data.dataset_siren_mlp import BathymetryShipPointDataset

def depth_binned_statistics(y_true, y_pred, bin_size=200, min_points=30):
    bins = np.arange(y_true.min(), y_true.max() + bin_size, bin_size)
    stats = {
        "depth_min": [], "depth_max": [], "depth_center": [],
        "Number": [], "MAE": [], "RMSE": [], "STD": [], "Bias": [],
    }
    residual = y_pred - y_true
    for i in range(len(bins) - 1):
        mask = (y_true >= bins[i]) & (y_true < bins[i + 1])
        n = mask.sum()
        stats["depth_center"].append(0.5 * (bins[i] + bins[i + 1]))
        stats["depth_min"].append(bins[i])
        stats["depth_max"].append(bins[i + 1])
        stats["Number"].append(n)
        if n < min_points:
            for k in ["MAE", "RMSE", "STD", "Bias"]: stats[k].append(np.nan)
        else:
            r = residual[mask]
            stats["MAE"].append(np.mean(np.abs(r)))
            stats["RMSE"].append(np.sqrt(np.mean(r**2)))
            stats["STD"].append(np.std(r))
            stats["Bias"].append(np.mean(r))
    return stats

# 数据路径
PATHS = {
    "grav": Path("./data/processed/G_ocean_raw.nc"),
    "curv": Path("./data/processed/VGG_ocean_raw.nc"),
    "b_lp": Path("./data/processed/B_LP.nc"),
    "g_lp": Path("./data/processed/G_LP.nc"),
    "g_bp": Path("./data/processed/G_BP.nc"),
    "vgg_bp": Path("./data/processed/VGG_BP.nc"),
    "g_east": Path("./data/processed/G_East.nc"),
    "g_north": Path("./data/processed/G_North.nc"),
    "g_east_bp": Path("./data/processed/G_East_BP.nc"),
    "g_north_bp": Path("./data/processed/G_North_BP.nc"),
}

SHIP_NC = "./data/ship_combined_calibrated.nc"
ALIGNED_SHIP_NC = "./data/ship_bathymetry_points_aligned.nc"

LON_RANGE = (104, 122)
LAT_RANGE = (0, 26)

def train():
    os.makedirs("./checkpoints/siren_mlp", exist_ok=True)
    os.makedirs("./result/siren_mlp", exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # =========================
    # Dataset
    # =========================
    dataset = BathymetryShipPointDataset(
        feature_paths=PATHS,
        ship_nc_path=SHIP_NC,
        aligned_ship_nc_path=ALIGNED_SHIP_NC,
        lon_range=LON_RANGE,
        lat_range=LAT_RANGE,
        agg_method="median",
        normalize=True
    )

    # 展开 [1, C+2, 1, 1] 为一维列表以适配 JSON 保存
    x_means_flat = dataset.X_mean.flatten().tolist()
    x_stds_flat = dataset.X_std.flatten().tolist()

    norm_params = {
        "y_mean": float(dataset.y_mean),
        "y_std": float(dataset.y_std),
        "x_means": x_means_flat,
        "x_stds": x_stds_flat
    }
    with open("./checkpoints/siren_mlp/normalization_params.json", "w") as f:
        json.dump(norm_params, f, indent=4)
    print(f"Normalization parameters saved to ./checkpoints/siren_mlp/normalization_params.json")

    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    val_size   = int(0.1 * dataset_size)
    test_size  = dataset_size - train_size - val_size

    train_set, val_set, test_set = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    batch_size = 512
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # =========================
    # Model (双分支 SIREN)
    # =========================
    # dataset.X 的 shape 为 [N, C+2, 3, 3]
    input_dim = dataset.X.shape[1] 
    print(f"SIREN Channels: {input_dim}")
    
    model = DualHeadSirenNet(
        in_channels=input_dim,
        hidden_dim=128,
        num_layers=5,
        omega_0=30.0 # 可视训练震荡情况在此处调参
    ).to(device)

    max_epochs = 100
    criterion = torch.nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs, eta_min=1e-6
    )

    best_val_loss = float("inf")
    best_epoch = 0
    patience = 15
    no_improve = 0
    min_delta = 5e-6 

    train_hist, val_hist = [], []
    
    # =========================
    # Training loop
    # =========================
    # 训练循环内部修改
    print("\n[INFO] Start training Dual-Headed SIREN with Debug info...")
    for epoch in range(max_epochs):
        start_time = time.time()
        model.train()
        train_loss = 0.0

        for batch_idx, (X, mask, y) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1:03d} [Train]", leave=False)):
            X, mask, y = X.to(device), mask.to(device), y.to(device)

            optimizer.zero_grad()
            
            # 物理掩码判定：9 个点全有效为纯海域，否则为海岸带
            valid_counts = mask.sum(dim=(1, 2, 3))
            is_pure_ocean = (valid_counts == 9)
            is_coast = ~is_pure_ocean
            
            pred = model(X, mask)
            y_target = y.squeeze(-1)
            loss = criterion(pred, y_target)
            
            # 2. 加入防崩策略：如果单步 Loss 突然爆炸，直接丢弃该 Batch 的梯度
            if torch.isnan(loss) or loss.item() > 2.0:
                print(f"\n[Warning] 检测到梯度爆炸 (Loss: {loss.item():.4f})，跳过此步更新！")
                continue

            loss.backward()
            
            # 3. 严格的梯度裁剪，压制 SIREN 的灾难性发散
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()

            train_loss += loss.item() * len(y)

            # ==========================================
            # 4. 诊断输出：随机抽取 1% 的 Batch 打印具体表现
            # ==========================================
            if torch.rand(1).item() < 0.01:
                # 注意：此处 pred 和 y_target 还是归一化状态，为了直观我们需要还原
                pred_real = dataset.inverse_transform_y(pred.detach().cpu().numpy())
                y_real = dataset.inverse_transform_y(y_target.detach().cpu().numpy())
                error_real = np.abs(pred_real - y_real)
                
                print(f"\n--- Epoch {epoch+1} Batch {batch_idx} 诊断 ---")
                
                # 打印一个海域样本的信息
                if is_pure_ocean.any():
                    idx_ocean = is_pure_ocean.nonzero(as_tuple=True)[0][0].item()
                    print(f"🌊 [海域分支] 预测值: {pred_real[idx_ocean]:.1f}m | 真实标签: {y_real[idx_ocean]:.1f}m | 误差: {error_real[idx_ocean]:.1f}m")
                
                # 打印一个海岸样本的信息
                if is_coast.any():
                    idx_coast = is_coast.nonzero(as_tuple=True)[0][0].item()
                    print(f"🏖️ [海岸分支] 预测值: {pred_real[idx_coast]:.1f}m | 真实标签: {y_real[idx_coast]:.1f}m | 误差: {error_real[idx_coast]:.1f}m")

        train_loss /= len(train_loader.dataset)
        train_hist.append(train_loss)

        model.eval()
        val_loss = 0.0
        y_true_list, y_pred_list = [], []

        with torch.no_grad():
            for X, mask, y in val_loader:
                X, mask, y = X.to(device), mask.to(device), y.to(device)
                pred = model(X, mask)
                val_loss += criterion(pred, y.squeeze(-1)).item() * len(y)

                y_true_list.append(y.cpu().numpy())
                y_pred_list.append(pred.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        val_hist.append(val_loss)
        scheduler.step()

        y_true_norm = np.concatenate(y_true_list).squeeze()
        y_pred_norm = np.concatenate(y_pred_list).squeeze()

        y_true = dataset.inverse_transform_y(y_true_norm)
        y_pred = dataset.inverse_transform_y(y_pred_norm)

        val_mae = mean_absolute_error(y_true, y_pred)
        val_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        val_corr = pearsonr(y_true, y_pred)[0]

        epoch_time = time.time() - start_time
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch+1:03d} | Time {epoch_time:.1f}s | "
            f"Train {train_loss:.5f} | Val {val_loss:.5f} | "
            f"MAE {val_mae:.1f} m | RMSE {val_rmse:.1f} m | "
            f"Corr {val_corr:.3f} | LR {lr:.1e}"
        )

        if val_loss < (best_val_loss - min_delta):
            best_val_loss = val_loss
            best_epoch = epoch
            no_improve = 0
            torch.save(model.state_dict(), "./checkpoints/siren_mlp/best_model.pt")
            print("  ↳ Best model updated")
        else:
            no_improve += 1

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"./checkpoints/siren_mlp/epoch_{epoch+1:03d}.pt")

        if no_improve >= patience:
            print(f"\n[Early Stop] Stop at epoch {epoch+1}")
            break

    # =========================
    # Evaluation
    # =========================
    print("\nEvaluating on test set...")
    model.load_state_dict(torch.load("./checkpoints/siren_mlp/best_model.pt", weights_only=True))
    model.eval()

    y_true_list, y_pred_list = [], []
    lon_norm_list, lat_norm_list = [], []

    with torch.no_grad():
        for X, mask, y in test_loader:
            X, mask, y = X.to(device), mask.to(device), y.to(device)
            pred = model(X, mask)
            y_true_list.append(y.cpu().numpy())
            y_pred_list.append(pred.cpu().numpy())
            
            # X 现在是 4D: [B, C+2, 3, 3] 提取中心点的经纬度
            lon_norm_list.append(X[:, -2, 1, 1].cpu().numpy())
            lat_norm_list.append(X[:, -1, 1, 1].cpu().numpy())

    y_true_norm = np.concatenate(y_true_list).squeeze()
    y_pred_norm = np.concatenate(y_pred_list).squeeze()
    lon_norm = np.concatenate(lon_norm_list).squeeze()
    lat_norm = np.concatenate(lat_norm_list).squeeze()

    y_true = dataset.inverse_transform_y(y_true_norm)
    y_pred = dataset.inverse_transform_y(y_pred_norm)
    
    lon_mean, lon_std = x_means_flat[-2], x_stds_flat[-2]
    lat_mean, lat_std = x_means_flat[-1], x_stds_flat[-1]
    lon_real = lon_norm * lon_std + lon_mean
    lat_real = lat_norm * lat_std + lat_mean

    test_df = pd.DataFrame({
        "lon": lon_real,
        "lat": lat_real,
        "y_true": y_true,
        "y_pred": y_pred
    })
    test_csv_path = "./checkpoints/siren_mlp/test_results.csv"
    test_df.to_csv(test_csv_path, index=False)
    print(f"[INFO] Test set predictions saved to {test_csv_path}")

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    bias = np.mean(y_pred - y_true)
    r2   = r2_score(y_true, y_pred)
    r, _ = pearsonr(y_true, y_pred)

    print("\n===== Test Metrics (Dual-Head SIREN) =====")
    print(f"MAE   : {mae:.3f} m")
    print(f"RMSE  : {rmse:.3f} m")
    print(f"Bias  : {bias:.3f} m")
    print(f"R²    : {r2:.4f}")
    print(f"CorrR : {r:.4f}")

    plt.figure(figsize=(8, 5))
    plt.plot(train_hist, label="Train")
    plt.plot(val_hist, label="Val")
    plt.axvline(best_epoch, linestyle="--", color="r", label="Best")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training History")
    plt.grid(alpha=0.3)
    plt.savefig("./result/siren_mlp/train_val_curve.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    train()