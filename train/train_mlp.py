# train.py - 修改数据集划分部分
import torch
from torch.utils.data import DataLoader, random_split
import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

from models.mlp import ResidualMLP
from data.dataset import BathymetryPointDataset


PATHS = {
    "grav_path": Path("./data/SWOT/grav_SWOT_02.nc"),
    "gebco_path": Path("./tmp_img/bathymetry_long_wave_65km.nc"),
    "curv_path": Path("./data/SWOT/curv_SWOT_02.nc")
}
LON_RANGE = (112, 114)
LAT_RANGE = (15, 18)

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # ------------------------------------------------
    # 1. 构建完整 Dataset
    # ------------------------------------------------
    print("\n" + "="*50)
    print("1. 加载数据集...")
    start_time = time.time()
    
    dataset = BathymetryPointDataset(
        grav_path=PATHS["grav_path"],
        curv_path=PATHS["curv_path"],
        gebco_path=PATHS["gebco_path"],
        lon_range=LON_RANGE,
        lat_range=LAT_RANGE,
        normalize=True,
        downsample_method='median'
    )

    load_time = time.time() - start_time
    print(f"数据集加载完成，耗时: {load_time:.2f}秒")
    print(f"总样本数: {len(dataset):,}")
    
    # 验证数据分布
    print(f"数据分布统计:")
    print(f"  水深范围: {dataset.y.min():.1f} ~ {dataset.y.max():.1f}")
    print(f"  重力范围: {dataset.X[:, 0].min():.2f} ~ {dataset.X[:, 0].max():.2f}")
    print(f"  梯度范围: {dataset.X[:, 1].min():.2f} ~ {dataset.X[:, 1].max():.2f}")
    
    # 检查数据
    X_sample, y_sample = dataset[0]
    print(f"单个样本检查:")
    print(f"  输入X形状: {X_sample.shape} (应该是[4])")
    print(f"  标签y形状: {y_sample.shape} (应该是[1])")
    print(f"  输入特征: 重力={X_sample[0]:.3f}, 梯度={X_sample[1]:.3f}")
    print(f"           经度={X_sample[2]:.3f}, 纬度={X_sample[3]:.3f}")
    print(f"  标签水深: {y_sample[0]:.3f} (归一化后)")
    # ------------------------------------------------
    # 2. 随机划分训练/测试集（关键修正！）
    # ------------------------------------------------
    print("\n" + "="*50)
    print("2. 随机划分训练/测试集...")
    
    # 设置随机种子保证可重复性
    torch.manual_seed(42)
    
    # 计算划分大小
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)  # 80%训练
    test_size = dataset_size - train_size  # 20%测试
    
    print(f"总样本数: {dataset_size:,}")
    print(f"训练集大小: {train_size:,} (80%)")
    print(f"测试集大小: {test_size:,} (20%)")
    
    # 使用random_split进行随机划分
    train_set, test_set = random_split(
        dataset, 
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # 检查划分是否正确
    print(f"验证划分结果:")
    print(f"  训练集实际大小: {len(train_set):,}")
    print(f"  测试集实际大小: {len(test_set):,}")

    # ------------------------------------------------
    # 3. DataLoader
    # ------------------------------------------------
    print("\n" + "="*50)
    print("3. 创建DataLoader...")
    
    # 根据数据量调整批次大小
    if train_size < 10000:
        batch_size = min(256, train_size)  # 小数据用小批次
    else:
        batch_size = 8192
    
    train_loader = DataLoader(
        train_set, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_set, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0
    )
    
    print(f"批次大小: {batch_size}")
    print(f"训练批次: {len(train_loader)}")
    print(f"测试批次: {len(test_loader)}")
    
    # 检查第一批数据
    sample_batch = next(iter(train_loader))
    print(f"批次数据形状: X={sample_batch[0].shape}, y={sample_batch[1].shape}")

    # ------------------------------------------------
    # 4. 模型 / 优化器 / 损失
    # ------------------------------------------------
    print("\n" + "="*50)
    print("4. 初始化模型和优化器...")
    
    # 根据数据量调整模型大小
    if dataset_size < 10000:
        hidden_dim = 64
        num_layers = 4
        print("小数据集，使用较小模型")
    else:
        hidden_dim = 256
        num_layers = 8
        print("大数据集，使用标准模型")
    
    model = ResidualMLP(
        in_dim=4,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        out_dim=1
    ).to(device)
    
    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    # 学习率根据数据量调整
    if dataset_size < 10000:
        lr = 1e-4  # 小数据用小学习率
    else:
        lr = 1e-3
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )

    # ------------------------------------------------
    # 5. 训练循环
    # ------------------------------------------------
    print("\n" + "="*50)
    print("5. 开始训练...")
    
    # 记录训练历史
    train_history = []
    test_history = []
    best_test_loss = float('inf')
    best_epoch = 0
    
    total_epochs = 100 if dataset_size < 10000 else 60
    
    for epoch in range(total_epochs):
        epoch_start = time.time()
        
        # ---- 训练阶段 ----
        model.train()
        train_loss = 0.0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1:03d}/{total_epochs} [训练]', 
                         leave=False, ncols=100)
        
        for batch_idx, (X, y) in enumerate(train_pbar):
            X, y = X.to(device), y.to(device)

            # 前向传播
            pred = model(X)
            loss = criterion(pred, y)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪防止爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            # 统计损失
            train_loss += loss.item()
            
            # 更新进度条
            train_pbar.set_postfix({
                'loss': f'{loss.item():.6f}',
                'avg_loss': f'{train_loss/(batch_idx+1):.6f}'
            })
        
        train_loss /= len(train_loader)
        train_history.append(train_loss)
        
        # ---- 测试阶段 ----
        model.eval()
        test_loss = 0.0
        
        with torch.no_grad():
            test_pbar = tqdm(test_loader, desc=f'Epoch {epoch+1:03d}/{total_epochs} [测试]', 
                           leave=False, ncols=100)
            
            for X, y in test_pbar:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                loss = criterion(pred, y)
                test_loss += loss.item()
        
        test_loss /= len(test_loader)
        test_history.append(test_loss)
        
        # 更新学习率
        scheduler.step(test_loss)
        
        # 保存最佳模型
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_epoch = epoch
            torch.save(model.state_dict(), "./checkpoints/mlp/bathymetry_mlp_best.pt")
        
        # 计算epoch耗时
        epoch_time = time.time() - epoch_start
        
        # 打印epoch总结
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch+1:03d}/{total_epochs} | "
              f"耗时: {epoch_time:.1f}s | "
              f"Train Loss: {train_loss:.6f} | "
              f"Test Loss: {test_loss:.6f} | "
              f"LR: {current_lr:.1e} | "
              f"Best: {best_test_loss:.6f} (Epoch {best_epoch+1})")
        
        # 每10个epoch保存一次检查点
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'test_loss': test_loss,
                'train_history': train_history,
                'test_history': test_history
            }
            torch.save(checkpoint, f"./checkpoints/mlp/checkpoint_epoch_{epoch+1}.pt")
        
        # 早停机制
        if epoch - best_epoch > 30:
            print(f"\n⚠️  Early stopping! No improvement for {epoch - best_epoch} epochs.")
            break
    
    # ------------------------------------------------
    # 6. 训练完成
    # ------------------------------------------------
    print("\n" + "="*50)
    print("6. 训练完成，保存模型...")
    
    torch.save(model.state_dict(), "./checkpoints/mlp/bathymetry_mlp_final.pt")
    
    # 保存训练历史
    history = {
        'train_loss': train_history,
        'test_loss': test_history,
        'best_epoch': best_epoch,
        'best_test_loss': best_test_loss,
        'dataset_size': dataset_size
    }
    np.savez('./tmp_img/training_history.npz', **history)
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_history, label='Train Loss', linewidth=2)
    plt.plot(test_history, label='Test Loss', linewidth=2)
    plt.axvline(x=best_epoch, color='r', linestyle='--', alpha=0.5, label=f'Best Epoch: {best_epoch+1}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title(f'Training History (Dataset: {dataset_size:,} samples)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('./tmp_img/training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n训练总结:")
    print(f"  最佳模型: bathymetry_mlp_best.pt (Epoch {best_epoch+1}, Loss: {best_test_loss:.6f})")
    print(f"  最终损失: Train={train_history[-1]:.6f}, Test={test_history[-1]:.6f}")
    print(f"  数据集大小: {dataset_size:,} 个样本")
    
    if device == "cuda":
        print(f"\nGPU内存统计:")
        print(f"  峰值使用: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")

if __name__ == "__main__":
    train()