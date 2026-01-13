import torch
from torch.utils.data import DataLoader
import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path

# 获取当前脚本所在目录（train/）
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录（当前脚本的父目录）
project_root = os.path.dirname(current_dir)
# 将项目根目录添加到模块搜索路径
sys.path.insert(0, project_root)

from data.dataset import BathymetryPatchDataset
from models.UNet import BathymetryUNet
from models.loss import BathymetryLoss

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

PATHS = {
    "grav_path": Path("./data/SWOT/grav_SWOT_02.nc"),
    "gebco_path": Path("./data\GEBCO_2024\gebco_2024\GEBCO_2024.nc"),
    "curv_path": Path("./data/SWOT/curv_SWOT_02.nc"),
    "h_mlp_path": Path("./tmp_img/mlp_prediction.nc")
}
LON_RANGE = (112, 114)
LAT_RANGE = (15, 18) 

def compute_physical_metrics(h_pred, h_true):
    diff = h_pred - h_true
    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(np.mean(diff**2))
    corr = np.corrcoef(h_pred.flatten(), h_true.flatten())[0, 1]
    return mae, rmse, corr

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n" + "="*50)
    print("1. 构建 Dataset")

    train_set = BathymetryPatchDataset(
        grav_path=PATHS["grav_path"],
        curv_path=PATHS["curv_path"],
        gebco_path=PATHS["gebco_path"],
        h_mlp_path=PATHS["h_mlp_path"],
        lon_range=LON_RANGE,
        lat_range=LAT_RANGE,
        patch_size=64,
        stride=32,
        normalize=True,
        split="train"
    )
    res_mean = train_set.residual_mean
    res_std  = train_set.residual_std

    val_set = BathymetryPatchDataset(
        grav_path=PATHS["grav_path"],
        curv_path=PATHS["curv_path"],
        gebco_path=PATHS["gebco_path"],
        h_mlp_path=PATHS["h_mlp_path"],
        lon_range=LON_RANGE,
        lat_range=LAT_RANGE,
        patch_size=64,
        stride=32,
        normalize=True,
        split="val"
    )

    print(f"Train patches: {len(train_set)}")
    print(f"Val patches:   {len(val_set)}")

    # ----------------------------
    # DataLoader（UNet 显存友好）
    # ----------------------------
    batch_size = 16   # ★ UNet 的 batch size 是瓶颈
    num_workers = 0

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # ----------------------------
    # Model
    # ----------------------------
    print("\n" + "="*50)
    print("2. 初始化模型")

    model = BathymetryUNet(
        in_channels=5,
        out_channels=1,
        base_ch=32
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"UNet 参数量: {total_params:,}")

    # ----------------------------
    # Loss / Optimizer / Scheduler
    # ----------------------------
    criterion = BathymetryLoss(lambda_grad = 0.2,lambda_l2 = 0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4,weight_decay=0.01)
    
    total_epochs = 100
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_epochs,
        eta_min=3e-5
    )

    # ----------------------------
    # Training config
    # ----------------------------
    
    best_val_loss = float("inf")
    best_epoch = 0

    train_history = []
    val_history = []

    print("\n" + "="*50)
    print("3. 开始训练")

    for epoch in range(total_epochs):
        epoch_start = time.time()

        # ========= Train =========
        model.train()
        train_loss = 0.0

        train_pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1:03d}",
            bar_format='{l_bar}{bar:30}{r_bar}',
            ncols=80,
            leave=False
        )

        for x, y in train_pbar:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            train_loss += loss.item()
            train_pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss /= len(train_loader)
        train_history.append(train_loss)

        # ========= Validation =========
        model.eval()
        val_loss = 0.0

        mae_list = []
        rmse_list = []
        corr_list = []

        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(val_loader):
                x = x.to(device)
                y = y.to(device)

                pred = model(x)
                loss = criterion(pred, y)
                val_loss += loss.item()

                # -------------------------------
                # 反归一化残差（物理量）
                # -------------------------------
                delta_pred = pred.cpu().numpy()          # (B,1,H,W)
                delta_pred = delta_pred.squeeze(1)       # (B,H,W)
                delta_pred_phys = delta_pred * res_std + res_mean

                # -------------------------------
                # 取对应 patch 的 h_mlp / h_true
                # -------------------------------
                B = delta_pred_phys.shape[0]

                for b in range(B):
                    global_idx = batch_idx * val_loader.batch_size + b
                    if global_idx >= len(val_set):
                        break

                    i, j = val_set.patch_coords[global_idx]
                    ps = val_set.patch_size

                    h_mlp_patch_norm = val_set.h_base[i:i+ps, j:j+ps]
                    h_mlp_patch_phys = (
                        h_mlp_patch_norm * val_set.h_base_std
                        + val_set.h_base_mean
                    )

                    h_true_patch = val_set.gebco[i:i+ps, j:j+ps]

                    h_pred_patch = h_mlp_patch_phys + delta_pred_phys[b]

                    mae, rmse, corr = compute_physical_metrics(
                        h_pred_patch, h_true_patch
                    )

                    mae_list.append(mae)
                    rmse_list.append(rmse)
                    corr_list.append(corr)

        val_loss /= len(val_loader)

        val_mae  = np.mean(mae_list)
        val_rmse = np.mean(rmse_list)
        val_corr = np.mean(corr_list)

        val_history.append(val_loss)


        # ========= Save best =========
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch

            torch.save(
                model.state_dict(),
                "./checkpoints./unet/bathymetry_unet_best.pt"
            )

        # ========= Logging =========
        epoch_time = time.time() - epoch_start
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


        # ========= Checkpoint =========
        if (epoch + 1) % 10 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "train_history": train_history,
                    "val_history": val_history,
                },
                f"./checkpoints./unet/checkpoint_epoch_{epoch+1}.pt"
            )

        # ========= Early stopping =========
        if epoch - best_epoch > 30:
            print("⚠️ Early stopping triggered")
            break

    # ----------------------------
    # Save final & plot
    # ----------------------------
    torch.save(model.state_dict(), "./checkpoints./unet/bathymetry_unet_final.pt")

    np.savez(
        "./tmp_img/unet_training_history.npz",
        train_loss=train_history,
        val_loss=val_history,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss
    )
    np.savez(
        "./checkpoints./unet/residual_norm_stats.npz",
        residual_mean=train_set.residual_mean,
        residual_std=train_set.residual_std,
        grav_mean=train_set.grav_mean,
        grav_std=train_set.grav_std,
        curv_mean=train_set.curv_mean, 
        curv_std=train_set.curv_std,
        h_base_mean=train_set.h_base_mean,
        h_base_std=train_set.h_base_std,
    )

    print("\n训练完成")
    print(f"Best Epoch: {best_epoch+1}")
    print(f"Best Val Loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    train()
