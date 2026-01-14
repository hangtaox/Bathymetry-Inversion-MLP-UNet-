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

# 使用新的数据集类
from data.dataset import BathymetryDataset
from models.UNet import BathymetryUNet
from models.loss import BathymetryLoss

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

PATHS = {
    "grav_path": Path("./data/SWOT/grav_SWOT_02.nc"),
    "gebco_path": Path("./data/GEBCO_2024/gebco_2024/GEBCO_2024.nc"),
    "curv_path": Path("./data/SWOT/curv_SWOT_02.nc"),
    "h_mlp_path": Path("./tmp_img/mlp_prediction.nc")
}
LON_RANGE = (112, 114)
LAT_RANGE = (15, 18) 

def compute_physical_metrics(h_pred, h_true):
    """计算物理指标"""
    diff = h_pred - h_true
    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(np.mean(diff**2))
    corr = np.corrcoef(h_pred.flatten(), h_true.flatten())[0, 1]
    return mae, rmse, corr

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    print("\n" + "="*50)
    print("1. 构建数据集")
    
    # 创建训练集
    train_set = BathymetryDataset(
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
    
    # 创建验证集
    val_set = BathymetryDataset(
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
    
    # 保存归一化统计信息
    train_set.save_stats("./checkpoints/unet/normalization_stats.npz")
    
    print(f"训练集样本数: {len(train_set)}")
    print(f"验证集样本数: {len(val_set)}")
    
    # 创建DataLoader
    batch_size = 16
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True
    )

    print("\n" + "="*50)
    print("2. 初始化模型")
    
    # 创建模型
    model = BathymetryUNet(
        in_channels=5,  # 重力、梯度、经度、纬度、h_mlp
        out_channels=1,  # 预测绝对深度
        base_ch=32
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")
    
    # 损失函数、优化器、学习率调度器
    criterion = BathymetryLoss(lambda_grad=0.2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    
    total_epochs = 100
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_epochs, eta_min=3e-5
    )
    
    # 训练记录
    best_val_loss = float("inf")
    best_epoch = 0
    train_history = []
    val_history = []
    
    print("\n" + "="*50)
    print("3. 开始训练")
    
    for epoch in range(total_epochs):
        epoch_start = time.time()
        
        # ---------- 训练阶段 ----------
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
        
        # ---------- 验证阶段 ----------
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
                
                # 将预测的归一化深度反归一化为物理值
                pred_norm = pred.cpu().numpy().squeeze(1)  # (B, H, W)
                pred_phys = pred_norm * train_set.stats['depth_std'] + train_set.stats['depth_mean']
                
                # 将标签的归一化深度反归一化为物理值
                y_norm = y.cpu().numpy().squeeze(1)  # (B, H, W)
                y_phys = y_norm * train_set.stats['depth_std'] + train_set.stats['depth_mean']
                
                # 计算每个batch的指标
                for b in range(pred_phys.shape[0]):
                    mae, rmse, corr = compute_physical_metrics(
                        pred_phys[b], y_phys[b]
                    )
                    mae_list.append(mae)
                    rmse_list.append(rmse)
                    corr_list.append(corr)
        
        val_loss /= len(val_loader)
        val_mae = np.mean(mae_list)
        val_rmse = np.mean(rmse_list)
        val_corr = np.mean(corr_list)
        val_history.append(val_loss)
        
        # ---------- 保存最佳模型 ----------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            
            torch.save(
                model.state_dict(),
                "./checkpoints/unet/bathymetry_unet_best.pt"
            )
        
        # ---------- 输出训练信息 ----------
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
        
        # ---------- 定期保存检查点 ----------
        if (epoch + 1) % 10 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "train_history": train_history,
                    "val_history": val_history,
                    "stats": train_set.stats
                },
                f"./checkpoints/unet/checkpoint_epoch_{epoch+1}.pt"
            )
        
        # ---------- 早停 ----------
        if epoch - best_epoch > 30:
            print("⚠️ 早停触发")
            break
        
        # 更新学习率
        scheduler.step()
    
    # ---------- 保存最终模型和训练记录 ----------
    torch.save(
        model.state_dict(),
        "./checkpoints/unet/bathymetry_unet_final.pt"
    )
    
    np.savez(
        "./tmp_img/unet_training_history.npz",
        train_loss=train_history,
        val_loss=val_history,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss
    )
    
    print("\n训练完成")
    print(f"最佳轮次: {best_epoch+1}")
    print(f"最佳验证损失: {best_val_loss:.6f}")


if __name__ == "__main__":
    train()