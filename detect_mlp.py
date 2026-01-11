# inference_fixed.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def infer(model, X, X_mean, X_std, y_mean, y_std, device='cuda'):
    """模型推理主函数"""
    model.eval()
    
    # 归一化
    X_normalized = (X - X_mean) / X_std
    X_tensor = torch.tensor(X_normalized, dtype=torch.float32)
    
    # 批量推理避免内存溢出
    batch_size = 4096
    predictions = []
    
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i:i+batch_size].to(device)
            pred_normalized = model(batch)
            predictions.append(pred_normalized.cpu().numpy())
    
    pred_normalized = np.vstack(predictions)
    
    # 反归一化
    pred = pred_normalized * y_std + y_mean
    return pred

def calculate_errors(pred, target, region_name=""):
    """计算误差指标"""
    valid_mask = ~np.isnan(pred) & ~np.isnan(target)
    pred_valid = pred[valid_mask]
    target_valid = target[valid_mask]
    
    if len(pred_valid) == 0:
        print("警告：没有有效数据用于误差计算")
        return {}
    
    errors = {
        'MAE': mean_absolute_error(target_valid, pred_valid),
        'RMSE': np.sqrt(mean_squared_error(target_valid, pred_valid)),
        'R2': r2_score(target_valid, pred_valid),
        'Bias': np.mean(pred_valid - target_valid),
        'Std': np.std(pred_valid - target_valid)
    }
    
    if region_name:
        print(f"\n=== {region_name} 区域误差分析 ===")
        for name, value in errors.items():
            if name == 'R2':
                print(f"{name}: {value:.3f}")
            else:
                print(f"{name}: {value:.1f} m")
    
    return errors

def compare_with_topo(model, dataset, topo_data, device='cuda'):
    """
    与地形数据对比分析
    
    Args:
        model: 训练好的模型 (输入维度必须是4!)
        dataset: BathymetryPointDataset实例（包含归一化参数）
        topo_data: 对应区域的地形参考数据 (H, W)
    """
    # 获取归一化参数
    X_mean = dataset.X_mean
    X_std = dataset.X_std
    y_mean = dataset.y_mean
    y_std = dataset.y_std
    
    # 重建输入特征矩阵
    H, W = dataset.H, dataset.W
    lons = dataset.lons
    lats = dataset.lats
    
    # 创建网格
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    # 构建完整输入特征 (4个特征，不包含水深！)
    X_full = np.stack([
        dataset.grav.flatten(),    # 重力异常
        dataset.curv.flatten(),    # 重力梯度
        lon_grid.flatten(),        # 经度
        lat_grid.flatten()         # 纬度
    ], axis=1)  # 形状: (H*W, 4)
    
    print(f"输入特征形状: {X_full.shape}")
    
    # 推理预测
    pred_flat = infer(model, X_full, X_mean, X_std, y_mean, y_std, device)
    pred_grid = pred_flat.reshape(H, W)
    
    # 计算误差
    errors = calculate_errors(pred_grid, topo_data, "完整区域")
    
    # 可视化比较
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. 预测水深
    vmin = min(np.nanmin(pred_grid), np.nanmin(topo_data))
    vmax = max(np.nanmax(pred_grid), np.nanmax(topo_data))
    
    im1 = axes[0, 0].imshow(pred_grid, cmap='terrain', aspect='auto',
                           vmin=vmin, vmax=vmax)
    axes[0, 0].set_title('模型预测水深')
    plt.colorbar(im1, ax=axes[0, 0], label='水深 (m)')
    
    # 2. 参考水深
    im2 = axes[0, 1].imshow(topo_data, cmap='terrain', aspect='auto',
                           vmin=vmin, vmax=vmax)
    axes[0, 1].set_title('参考水深 (topo)')
    plt.colorbar(im2, ax=axes[0, 1], label='水深 (m)')
    
    # 3. 残差图
    residual = pred_grid - topo_data
    # 使用95%分位数避免极端值影响颜色范围
    residual_valid = residual[~np.isnan(residual)]
    if len(residual_valid) > 0:
        vmax_res = np.percentile(np.abs(residual_valid), 95)
    else:
        vmax_res = 100
    
    im3 = axes[0, 2].imshow(residual, cmap='RdBu_r', aspect='auto',
                           vmin=-vmax_res, vmax=vmax_res)
    axes[0, 2].set_title(f'残差 (预测-参考)\nMAE: {errors.get("MAE", 0):.1f}m')
    plt.colorbar(im3, ax=axes[0, 2], label='残差 (m)')
    
    # 4. 散点图
    valid_mask = ~np.isnan(pred_grid) & ~np.isnan(topo_data)
    if np.sum(valid_mask) > 0:
        axes[1, 0].scatter(topo_data[valid_mask], pred_grid[valid_mask],
                          alpha=0.3, s=1, c='blue')
        
        # 1:1线
        min_val = min(np.nanmin(pred_grid), np.nanmin(topo_data))
        max_val = max(np.nanmax(pred_grid), np.nanmax(topo_data))
        axes[1, 0].plot([min_val, max_val], [min_val, max_val],
                       'r--', linewidth=2, label='1:1线')
        
        axes[1, 0].set_xlim(min_val, max_val)
        axes[1, 0].set_ylim(min_val, max_val)
    
    axes[1, 0].set_xlabel('参考水深 (m)')
    axes[1, 0].set_ylabel('预测水深 (m)')
    axes[1, 0].set_title(f'散点图 (R²={errors.get("R2", 0):.3f})')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. 残差分布直方图
    if len(residual_valid) > 0:
        axes[1, 1].hist(residual_valid, bins=50, alpha=0.7, color='green',
                       edgecolor='black')
        axes[1, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[1, 1].axvline(x=errors.get('Bias', 0), color='b',
                          linestyle='-', linewidth=2,
                          label=f'Bias={errors.get("Bias", 0):.1f}m')
    
    axes[1, 1].set_xlabel('残差 (m)')
    axes[1, 1].set_ylabel('频数')
    axes[1, 1].set_title('残差分布')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. 误差统计表
    axes[1, 2].axis('off')
    if errors:
        error_text = "误差统计:\n"
        error_text += "="*20 + "\n"
        for name, value in errors.items():
            if name == 'R2':
                error_text += f"{name}: {value:.3f}\n"
            else:
                error_text += f"{name}: {value:.1f} m\n"
        
        # 添加平均水深信息
        true_valid = topo_data[~np.isnan(topo_data)]
        if len(true_valid) > 0:
            mean_depth = np.mean(true_valid)
            error_text += f"\n平均水深: {mean_depth:.1f} m\n"
            if 'RMSE' in errors:
                relative_rmse = errors['RMSE'] / abs(mean_depth) * 100
                error_text += f"相对RMSE: {relative_rmse:.1f}%\n"
    
        axes[1, 2].text(0.1, 0.5, error_text, fontsize=12,
                       verticalalignment='center', family='monospace',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    
    plt.tight_layout()
    plt.savefig('./tmp_img/topo_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return pred_grid, errors

def main_inference():
    """主推理函数"""
    import xarray as xr
    from models.mlp import ResidualMLP
    from data.dataset import BathymetryPointDataset # 使用修复后的数据集类
    
    # 路径配置
    PATHS = {
        "grav_path": Path("./data/SWOT/grav_SWOT_02.nc"),
        "gebco_path": Path("./data\GEBCO_2024\gebco_2024\GEBCO_2024.nc"),
        "curv_path": Path("./data/SWOT/curv_SWOT_02.nc")
    }
    LON_RANGE = (112, 114)
    LAT_RANGE = (15, 18)
    
    # 1. 加载数据集获取归一化参数
    print("加载数据集...")
    dataset = BathymetryPointDataset(
        grav_path=PATHS["grav_path"],
        curv_path=PATHS["curv_path"],
        gebco_path=PATHS["gebco_path"],
        lon_range=LON_RANGE,
        lat_range=LAT_RANGE,
        normalize=True
    )
    
    print(f"数据集加载完成，样本数: {len(dataset)}")
    print(f"归一化参数:")
    print(f"  y_mean: {dataset.y_mean:.1f}, y_std: {dataset.y_std:.1f}")
    
    # 2. 加载模型
    print("\n加载模型...")
    model_path = "./checkpoints/mlp/bathymetry_mlp_best.pt"
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # 关键：模型输入维度必须是4
    model = ResidualMLP(
        in_dim=4,           # 输入维度必须是4
        hidden_dim=256,     # 根据你的训练配置调整
        num_layers=8,
        out_dim=1
    )
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()
    print(f"模型加载成功: {model_path}")
    print(f"使用设备: {device}")
    
    # 3. 加载真实水深数据
    print("\n加载真实水深数据...")
    ds_topo = xr.open_dataset(PATHS["gebco_path"])
    topo_highres = ds_topo['elevation'].sel(
        lon=slice(*LON_RANGE),
        lat=slice(*LAT_RANGE)
    )
    ds_topo.close()
    
    # 降采样到与SWOT相同分辨率
    H, W = dataset.H, dataset.W
    factor = topo_highres.shape[0] // H
    topo_data = topo_highres.values[::factor, ::factor][:H, :W]
    
    print(f"真实水深数据形状: {topo_data.shape}")
    print(f"真实水深范围: {np.nanmin(topo_data):.1f} ~ {np.nanmax(topo_data):.1f} m")
    
    # 4. 运行推理
    print("\n开始推理...")
    pred_grid, errors = compare_with_topo(model, dataset, topo_data, device)
    
    # 5. 保存结果
    np.save('./tmp_img/predicted_bathymetry.npy', pred_grid)
    np.save('./tmp_img/true_bathymetry.npy', topo_data)
    
    print("\n" + "="*50)
    print("推理完成!")
    print(f"预测结果保存至: ./tmp_img/predicted_bathymetry.npy")
    print(f"对比图保存至: ./tmp_img/topo_comparison.png")
    
    return pred_grid, errors

if __name__ == "__main__":
    main_inference()