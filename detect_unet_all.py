import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xarray as xr
from tqdm import tqdm
from pathlib import Path

# 使用新的推理数据集类
from data.dataset import BathymetryInferenceDataset
from models.UNet import BathymetryUNet

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

PATHS = {
    "grav_path": Path("./data/SWOT/grav_SWOT_02.nc"),
    "gebco_path": Path("./data/GEBCO_2024/gebco_2024/GEBCO_2024.nc"),
    "curv_path": Path("./data/SWOT/curv_SWOT_02.nc"),
    "b_lp_path": Path("./data/processed/B_LP.nc"),
    "g_lp_path": Path("./data/processed/G_LP.nc"),
    "g_bp_path": Path("./data/processed/G_BP.nc"),
    "vgg_bp_path": Path("./data/processed/VGG_BP.nc")
}
LON_RANGE = (114, 116)
LAT_RANGE = (15, 18)

def make_gaussian_weight(patch_size, sigma_ratio=0.25):
    """生成高斯权重"""
    ax = np.linspace(-(patch_size - 1) / 2., (patch_size - 1) / 2., patch_size)
    xx, yy = np.meshgrid(ax, ax)
    sigma = patch_size * sigma_ratio
    weight = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    weight = weight / weight.max()
    return weight.astype(np.float32)

def infer_unet_full_map(model, dataset, stats, device="cuda"):
    """推理整个区域"""
    model.eval()
    
    H, W = dataset.H, dataset.W
    patch_size = dataset.patch_size
    
    # 初始化累加器
    pred_sum = np.zeros((H, W), dtype=np.float32)
    pred_count = np.zeros((H, W), dtype=np.float32)
    
    # 创建数据加载器
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    weight = make_gaussian_weight(patch_size)
    
    # 推理
    with torch.no_grad():
        for idx, x in enumerate(tqdm(loader, desc="推理中")):
            x = x.to(device)
            pred = model(x).squeeze().cpu().numpy()
            
            i, j = dataset.patch_coords[idx]
            pred_sum[i:i+patch_size, j:j+patch_size] += pred * weight
            pred_count[i:i+patch_size, j:j+patch_size] += weight
    
    # 合并patches
    pred_count[pred_count == 0] = 1
    depth_norm = pred_sum / pred_count
    
    # 反归一化得到物理深度
    depth_phys = depth_norm * stats['depth_std'] + stats['depth_mean']
    
    return depth_phys

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
        'Std': np.std(pred_valid - target_valid),
        'Correlation': np.corrcoef(target_valid, pred_valid)[0, 1]
    }
    
    if region_name:
        print(f"\n=== {region_name} 区域误差分析 ===")
        for name, value in errors.items():
            if name in ['R2', 'Correlation']:
                print(f"{name}: {value:.3f}")
            else:
                print(f"{name}: {value:.2f} m")
    
    return errors

def visualize_comparison(pred_grid, true_grid, errors):
    """可视化比较结果"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 确定统一的颜色范围
    vmin = min(np.nanmin(pred_grid), np.nanmin(true_grid))
    vmax = max(np.nanmax(pred_grid), np.nanmax(true_grid))
    
    # 1. 预测水深
    im1 = axes[0, 0].imshow(pred_grid, cmap='terrain', vmin=vmin, vmax=vmax)
    axes[0, 0].set_title('UNet预测水深')
    plt.colorbar(im1, ax=axes[0, 0], label='水深 (m)')
    
    # 2. 真实水深
    im2 = axes[0, 1].imshow(true_grid, cmap='terrain', vmin=vmin, vmax=vmax)
    axes[0, 1].set_title('真实水深 (GEBCO)')
    plt.colorbar(im2, ax=axes[0, 1], label='水深 (m)')
    
    # 3. 残差图
    residual = pred_grid - true_grid
    residual_valid = residual[~np.isnan(residual)]
    vmax_res = np.percentile(np.abs(residual_valid), 95) if len(residual_valid) > 0 else 100
    
    im3 = axes[0, 2].imshow(residual, cmap='RdBu_r', vmin=-vmax_res, vmax=vmax_res)
    axes[0, 2].set_title(f'残差 (预测-真实)\nMAE: {errors.get("MAE", 0):.2f} m')
    plt.colorbar(im3, ax=axes[0, 2], label='残差 (m)')
    
    # 4. 散点图
    valid_mask = ~np.isnan(pred_grid) & ~np.isnan(true_grid)
    if np.sum(valid_mask) > 0:
        axes[1, 0].scatter(true_grid[valid_mask], pred_grid[valid_mask],
                          alpha=0.3, s=1, c='blue')
        min_val = min(np.nanmin(pred_grid), np.nanmin(true_grid))
        max_val = max(np.nanmax(pred_grid), np.nanmax(true_grid))
        axes[1, 0].plot([min_val, max_val], [min_val, max_val],
                       'r--', linewidth=2, label='1:1线')
        axes[1, 0].set_xlim(min_val, max_val)
        axes[1, 0].set_ylim(min_val, max_val)
    axes[1, 0].set_xlabel('真实水深 (m)')
    axes[1, 0].set_ylabel('预测水深 (m)')
    axes[1, 0].set_title(f'散点图 (R²={errors.get("R2", 0):.3f})')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. 残差直方图
    if len(residual_valid) > 0:
        axes[1, 1].hist(residual_valid, bins=50, alpha=0.7, color='green',
                       edgecolor='black')
        axes[1, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[1, 1].axvline(x=errors.get('Bias', 0), color='b',
                          linestyle='-', linewidth=2,
                          label=f'Bias={errors.get("Bias", 0):.2f} m')
    axes[1, 1].set_xlabel('残差 (m)')
    axes[1, 1].set_ylabel('频数')
    axes[1, 1].set_title('残差分布')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. 误差统计表
    axes[1, 2].axis('off')
    error_text = "误差统计:\n" + "="*20 + "\n"
    for name, value in errors.items():
        if name in ['R2', 'Correlation']:
            error_text += f"{name}: {value:.3f}\n"
        else:
            error_text += f"{name}: {value:.2f} m\n"
    
    true_valid = true_grid[~np.isnan(true_grid)]
    if len(true_valid) > 0:
        mean_depth = np.mean(true_valid)
        error_text += f"\n平均水深: {mean_depth:.2f} m\n"
        if 'RMSE' in errors:
            relative_rmse = errors['RMSE'] / abs(mean_depth) * 100
            error_text += f"相对RMSE: {relative_rmse:.2f}%\n"
    
    axes[1, 2].text(0.1, 0.5, error_text, fontsize=12,
                   verticalalignment='center', family='monospace',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    
    plt.tight_layout()
    plt.savefig('./tmp_img/unet_depth_prediction_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def load_true_depth(gebco_path, lon_range, lat_range, target_shape):
    """加载并降采样真实水深数据"""
    ds_topo = xr.open_dataset(gebco_path)
    topo_highres = ds_topo['elevation'].sel(
        lon=slice(*lon_range),
        lat=slice(*lat_range)
    ).values
    ds_topo.close()
    
    # 降采样到目标尺寸
    H, W = target_shape
    factor_h = topo_highres.shape[0] // H
    factor_w = topo_highres.shape[1] // W
    topo_data = topo_highres[::factor_h, ::factor_w][:H, :W]
    
    return topo_data

def save_prediction_as_netcdf(pred_depth, dataset, output_path):
    """保存预测结果为NetCDF文件"""
    # 获取经纬度坐标
    pred_lons = dataset.lons  # 一维经度数组
    pred_lats = dataset.lats  # 一维纬度数组
    
    # 创建xarray Dataset
    ds_pred = xr.Dataset(
        {
            "predicted_depth": (["lat", "lon"], pred_depth),
        },
        coords={
            "lon": pred_lons,
            "lat": pred_lats
        },
        attrs={
            "description": "UNet模型预测的绝对水深",
            "model": "UNet",
            "input_features": "重力、梯度、经度、纬度、MLP预测值"
        }
    )
    
    # 保存结果
    ds_pred.to_netcdf(output_path, mode='w')
    print(f"预测结果已保存到: {output_path}")

def main():
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 加载归一化统计信息
    stats_path = "./checkpoints/unet/normalization_stats.npz"
    stats = np.load(stats_path)
    print("归一化统计信息已加载")
    
    # 加载推理数据集
    print("加载推理数据集...")
    dataset = BathymetryInferenceDataset(
        grav_path=PATHS["grav_path"],
        curv_path=PATHS["curv_path"],
        b_lp_path=PATHS["b_lp_path"],
        g_lp_path=PATHS["g_lp_path"],
        g_bp_path=PATHS["g_bp_path"],
        vgg_bp_path=PATHS["vgg_bp_path"],
        lon_range=LON_RANGE,
        lat_range=LAT_RANGE,
        patch_size=64,
        stride=32,
        stats_path=stats_path
    )
    
    print(f"推理区域尺寸: {dataset.H} × {dataset.W}")
    print(f"Patch数量: {len(dataset)}")
    
    # 加载模型
    print("加载UNet模型...")
    model = BathymetryUNet(in_channels=8, out_channels=1, base_ch=32)
    model.load_state_dict(
        torch.load("./checkpoints/unet/bathymetry_unet_best.pt", 
                  map_location=device)
    )
    model.to(device)
    model.eval()
    
    # 加载真实水深数据
    print("加载真实水深数据...")
    true_depth = load_true_depth(
        PATHS["gebco_path"],
        LON_RANGE,
        LAT_RANGE,
        (dataset.H, dataset.W)
    )
    print(f"真实水深数据形状: {true_depth.shape}")
    
    # 推理整个区域
    print("开始推理...")
    pred_depth = infer_unet_full_map(model, dataset, stats, device=device)
    
    # 计算误差指标
    print("\n计算误差指标...")
    errors = calculate_errors(pred_depth, true_depth, "UNet预测深度")
    
    # 可视化比较
    print("生成可视化结果...")
    visualize_comparison(pred_depth, true_depth, errors)
    
    # 保存预测结果
    print("保存预测结果...")
    save_prediction_as_netcdf(
        pred_depth, 
        dataset, 
        "./tmp_img/unet_depth_prediction.nc"
    )
    
    # 同时保存为numpy格式
    np.save('./tmp_img/unet_depth_prediction.npy', pred_depth)
    
    print("\n推理完成!")
    return pred_depth, errors

if __name__ == "__main__":
    main()