import torch
from torch.utils.data import DataLoader, random_split
import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import os
import sys

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr

# -------------------------
# 路径设置
# -------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from models.mlp import ResidualMLP
from dataset_mlp import BathymetryShipPointDataset

# -------------------------
# 数据路径
# -------------------------
PATHS = {
    "grav": Path("./data/SWOT/grav_SWOT_02.nc"),
    "curv": Path("./data/SWOT/curv_SWOT_02.nc"),
    "b_lp": Path("./data/processed/B_LP.nc"),
    "g_lp": Path("./data/processed/G_LP.nc"),
    "g_bp": Path("./data/processed/G_BP.nc"),
    "vgg_bp": Path("./data/processed/VGG_BP.nc"),
}

SHIP_NC = "./ship_bathymetry_points.nc"
ALIGNED_SHIP_NC = "./ship_bathymetry_points_aligned.nc"

LON_RANGE = (105, 125)
LAT_RANGE = (0, 30)


# =========================
# 物理指标（UNet 同款）
# =========================
def compute_physical_metrics_1d(y_pred, y_true):
    diff = y_pred - y_true
    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(np.mean(diff ** 2))
    corr = np.corrcoef(y_pred, y_true)[0, 1]
    return mae, rmse, corr


def train():

    os.makedirs("./checkpoints/mlp", exist_ok=True)
    os.makedirs("./tmp_img", exist_ok=True)

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

    dataset_size = len(dataset)
    print(f"Total samples: {dataset_size}")

    train_size = int(0.8 * dataset_size)
    val_size   = int(0.1 * dataset_size)
    test_size  = dataset_size - train_size - val_size

    train_set, val_set, test_set = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"Split -> Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")

    batch_size = min(2048, train_size)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # =========================
    # Model
    # =========================
    model = ResidualMLP(
        in_dim=8,
        hidden_dim=128,
        num_layers=6,
        out_dim=1
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = torch.nn.SmoothL1Loss(beta=50.0)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=10,
        min_lr=1e-6,
        verbose=True
    )

    best_val_loss = float("inf")
    best_epoch = 0
    early_stop_patience = 20

    train_hist, val_hist = [], []

    max_epochs = 100

    # =========================
    # Training loop
    # =========================
    for epoch in range(max_epochs):
        start_time = time.time()

        # ---------- Train ----------
        model.train()
        train_loss = 0.0

        for X, y in tqdm(train_loader, desc=f"Epoch {epoch+1:03d} [Train]", leave=False):
            X, y = X.to(device), y.to(device)

            pred = model(X)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_hist.append(train_loss)

        # ---------- Val ----------
        model.eval()
        val_loss = 0.0

        y_true_list, y_pred_list = [], []

        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)

                pred = model(X)
                loss = criterion(pred, y)

                val_loss += loss.item()

                y_true_list.append(y.cpu().numpy())
                y_pred_list.append(pred.cpu().numpy())

        val_loss /= len(val_loader)
        val_hist.append(val_loss)

        scheduler.step(val_loss)

        # ---------- 物理指标 ----------
        y_true_norm = np.concatenate(y_true_list).squeeze()
        y_pred_norm = np.concatenate(y_pred_list).squeeze()

        y_true = dataset.inverse_transform_y(y_true_norm)
        y_pred = dataset.inverse_transform_y(y_pred_norm)

        val_mae, val_rmse, val_corr = compute_physical_metrics_1d(
            y_pred, y_true
        )

        # ---------- 保存最优 ----------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), "./checkpoints/mlp/best_model.pt")

        # ---------- 每 10 轮存一次 ----------
        if (epoch + 1) % 10 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "train_hist": train_hist,
                    "val_hist": val_hist,
                },
                f"./checkpoints/mlp/checkpoint_epoch_{epoch+1}.pt"
            )

        # ---------- 打印（UNet 风格） ----------
        epoch_time = time.time() - start_time
        lr = optimizer.param_groups[0]["lr"]

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

        if epoch - best_epoch > early_stop_patience:
            print("Early stopping triggered.")
            break

    # =========================
    # Test（完全保留你的原逻辑）
    # =========================
    print("\nEvaluating on test set...")
    model.load_state_dict(torch.load("./checkpoints/mlp/best_model.pt"))
    model.eval()

    y_true_list, y_pred_list = [], []

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
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

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=3, alpha=0.4)

    lims = [
        min(y_true.min(), y_pred.min()),
        max(y_true.max(), y_pred.max())
    ]
    plt.plot(lims, lims, "r--", lw=1)

    plt.xlabel("Ship depth (m)")
    plt.ylabel("Predicted depth (m)")
    plt.title(f"Prediction vs GT\nRMSE={rmse:.2f} m, R={r:.3f}")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("./tmp_img/scatter_pred_vs_gt.png", dpi=300)
    plt.show()

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
    plt.savefig("./tmp_img/residual_hist.png", dpi=300)
    plt.show()

    bins = np.arange(y_true.min(), y_true.max(), 200)
    rmse_bins = []

    for i in range(len(bins)-1):
        mask = (y_true >= bins[i]) & (y_true < bins[i+1])
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
    plt.savefig("./tmp_img/rmse_vs_depth.png", dpi=300)
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(train_hist, label="Train")
    plt.plot(val_hist, label="Val")
    plt.axvline(best_epoch, linestyle="--", color="r", label="Best")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training History")
    plt.grid(alpha=0.3)
    plt.savefig("./tmp_img/train_val_curve.png", dpi=300)
    plt.show()

    print("\nRunning full-region inference...")

if __name__ == "__main__":
    train()
