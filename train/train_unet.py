import torch
from torch.utils.data import DataLoader
import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

from data.dataset import BathymetryPatchDataset
from models.UNet import BathymetryUNet
from models.loss import BathymetryLoss

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

PATHS = {
    "grav_path": r"D:\project\filtered_gravity_bandpass.nc",
    "gebco_path": r"D:\project\data\GEBCO_2024\gebco_2024\GEBCO_2024.nc",
    "curv_path": r"D:\project\filtered_curvature_bandpass.nc"
}

LON_RANGE = (112, 114)
LAT_RANGE = (15, 18)


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n" + "="*50)
    print("1. 构建 Dataset")

    train_set = BathymetryPatchDataset(
        grav_path=PATHS["grav_path"],
        curv_path=PATHS["curv_path"],
        gebco_path=PATHS["gebco_path"],
        lon_range=LON_RANGE,
        lat_range=LAT_RANGE,
        patch_size=64,
        stride=32,
        normalize=True,
        split="train"
    )

    val_set = BathymetryPatchDataset(
        grav_path=PATHS["grav_path"],
        curv_path=PATHS["curv_path"],
        gebco_path=PATHS["gebco_path"],
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
    batch_size = 4   # ★ UNet 的 batch size 是瓶颈
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
        in_channels=2,
        out_channels=1,
        base_ch=32
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"UNet 参数量: {total_params:,}")

    # ----------------------------
    # Loss / Optimizer / Scheduler
    # ----------------------------
    criterion = BathymetryLoss(lambda_grad=0.2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=8,
        verbose=True
    )

    # ----------------------------
    # Training config
    # ----------------------------
    total_epochs = 100
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
            desc=f"Epoch {epoch+1:03d}/{total_epochs} [Train]",
            ncols=100,
            leave=False
        )

        for x, y in train_pbar:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            train_pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss /= len(train_loader)
        train_history.append(train_loss)

        # ========= Validation =========
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)

                pred = model(x)
                loss = criterion(pred, y)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_history.append(val_loss)

        scheduler.step(val_loss)

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

    print("\n训练完成")
    print(f"Best Epoch: {best_epoch+1}")
    print(f"Best Val Loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    train()
