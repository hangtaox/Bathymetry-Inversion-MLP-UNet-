# inference_unet.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xarray as xr
from tqdm import tqdm
from pathlib import Path

from data.dataset import BathymetryPatchDataset
from models.UNet import BathymetryUNet

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

PATHS = {
    "grav_path": Path("./data/SWOT/grav_SWOT_02.nc"),
    "gebco_path": Path("./data\GEBCO_2024\gebco_2024\GEBCO_2024.nc"),
    "curv_path": Path("./data/SWOT/curv_SWOT_02.nc")
}
LON_RANGE = (112, 114)
LAT_RANGE = (15, 18)


def infer_unet_full_map_v2(model, dataset, device="cuda"):
    """
    优化版本：用 enumerate + dataset.patch_coords 获得 patch 坐标，
    完整实现滑窗推理及拼接（覆盖计数平均）
    """

    model.eval()

    H, W = dataset.H, dataset.W
    patch_size = dataset.patch_size
    stride = dataset.stride

    pred_sum = np.zeros((H, W), dtype=np.float32)
    pred_count = np.zeros((H, W), dtype=np.float32)

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    with torch.no_grad():
        for idx, (x, y) in enumerate(tqdm(loader, desc="UNet 推理", ncols=100)):
            x = x.to(device)
            pred = model(x).squeeze().cpu().numpy()

            i, j = dataset.patch_coords[idx]
            pred_sum[i:i + patch_size, j:j + patch_size] += pred
            pred_count[i:i + patch_size, j:j + patch_size] += 1

    # 防止除零
    pred_count[pred_count == 0] = 1
    pred_full = pred_sum / pred_count

    # 反归一化
    pred_full = pred_full * dataset.gebco_std + dataset.gebco_mean

    return pred_full


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
                print(f"{name}: {value:.2f} m")

    return errors


def visualize_comparison(pred_grid, topo_data, errors):
    """误差及结果可视化"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    vmin = min(np.nanmin(pred_grid), np.nanmin(topo_data))
    vmax = max(np.nanmax(pred_grid), np.nanmax(topo_data))

    # 预测水深
    im1 = axes[0, 0].imshow(pred_grid, cmap='terrain', vmin=vmin, vmax=vmax)
    axes[0, 0].set_title('模型预测水深')
    plt.colorbar(im1, ax=axes[0, 0], label='水深 (m)')

    # 参考水深
    im2 = axes[0, 1].imshow(topo_data, cmap='terrain', vmin=vmin, vmax=vmax)
    axes[0, 1].set_title('参考水深 (GEBCO)')
    plt.colorbar(im2, ax=axes[0, 1], label='水深 (m)')

    # 残差图
    residual = pred_grid - topo_data
    residual_valid = residual[~np.isnan(residual)]
    vmax_res = np.percentile(np.abs(residual_valid), 95) if len(residual_valid) > 0 else 100

    im3 = axes[0, 2].imshow(residual, cmap='RdBu_r', vmin=-vmax_res, vmax=vmax_res)
    axes[0, 2].set_title(f'残差 (预测-参考)\nMAE: {errors.get("MAE", 0):.2f} m')
    plt.colorbar(im3, ax=axes[0, 2], label='残差 (m)')

    # 散点图
    valid_mask = ~np.isnan(pred_grid) & ~np.isnan(topo_data)
    if np.sum(valid_mask) > 0:
        axes[1, 0].scatter(topo_data[valid_mask], pred_grid[valid_mask],
                          alpha=0.3, s=1, c='blue')
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

    # 残差分布直方图
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

    # 误差统计表
    axes[1, 2].axis('off')
    error_text = "误差统计:\n" + "="*20 + "\n"
    for name, value in errors.items():
        if name == 'R2':
            error_text += f"{name}: {value:.3f}\n"
        else:
            error_text += f"{name}: {value:.2f} m\n"

    true_valid = topo_data[~np.isnan(topo_data)]
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
    plt.savefig('./tmp_img/unet_topo_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("加载数据集...")
    dataset = BathymetryPatchDataset(
        grav_path=PATHS["grav_path"],
        curv_path=PATHS["curv_path"],
        gebco_path=PATHS["gebco_path"],
        lon_range=LON_RANGE,
        lat_range=LAT_RANGE,
        patch_size=64,
        stride=32,
        normalize=True,
        split="all",
        downsample_method='median',
    )

    print(f"样本数（patch数）: {len(dataset)}")

    print("加载模型...")
    model = BathymetryUNet(in_channels=2, out_channels=1, base_ch=32)
    model.load_state_dict(torch.load("./checkpoints./unet/bathymetry_unet_best.pt", map_location=device))
    model.to(device)
    model.eval()

    print("加载真实水深数据...")
    ds_topo = xr.open_dataset(PATHS["gebco_path"])
    topo_highres = ds_topo['elevation'].sel(
        lon=slice(*LON_RANGE),
        lat=slice(*LAT_RANGE)
    ).values
    ds_topo.close()

    # 降采样到与数据集相同尺寸
    H, W = dataset.H, dataset.W
    factor_h = topo_highres.shape[0] // H
    factor_w = topo_highres.shape[1] // W
    topo_data = topo_highres[::factor_h, ::factor_w][:H, :W]

    print(f"真实水深数据形状: {topo_data.shape}")

    print("开始推理...")
    pred_grid = infer_unet_full_map_v2(model, dataset, device)

    print("计算误差...")
    errors = calculate_errors(pred_grid, topo_data, "UNet 全区域")

    print("绘制结果...")
    visualize_comparison(pred_grid, topo_data, errors)

    print("保存预测结果...")
    np.save('./tmp_img/unet_predicted_bathymetry.npy', pred_grid)
    np.save('./tmp_img/unet_true_bathymetry.npy', topo_data)

    print("推理完成。")

    return pred_grid, errors


if __name__ == "__main__":
    main()
