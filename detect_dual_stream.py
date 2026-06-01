import numpy as np
import pandas as pd
import xarray as xr
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
import scipy.signal as signal
import seaborn as sns
from scipy.interpolate import griddata  # 保留用于其他可能需要的插值
import cartopy.crs as ccrs              # 用于地图投影
import cartopy.feature as cfeature      # 用于添加海岸线和陆地掩膜
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

from data.dataset_cnn import BathymetryInferenceDataset  
from models.dual_stream import DualStreamFusionNet

# =========================================================
# 【全局控制】当前要进行单模型分析和全域推理的感受野大小
# =========================================================
PATCH_SIZE = 13

# ============================================================
# 公共基础函数
# ============================================================
def depth_binned_statistics(y_true, y_pred, bin_size=200, min_points=30):
    """按深度分箱计算误差统计量"""
    bins = np.arange(y_true.min(), y_true.max() + bin_size, bin_size)
    stats = {"depth_center": [], "Number": [], "RMSE": [], "MAE": [], "Bias": [], "STD": []}
    residual = y_pred - y_true

    for i in range(len(bins) - 1):
        mask = (y_true >= bins[i]) & (y_true < bins[i + 1])
        n = mask.sum()
        stats["depth_center"].append(0.5 * (bins[i] + bins[i + 1]))
        stats["Number"].append(n)

        if n < min_points:
            for k in ["RMSE", "MAE", "Bias", "STD"]: stats[k].append(np.nan)
        else:
            r = residual[mask]
            stats["RMSE"].append(np.sqrt(np.mean(r**2)))
            stats["MAE"].append(np.mean(np.abs(r)))
            stats["Bias"].append(np.mean(r))
            stats["STD"].append(np.std(r))
    return pd.DataFrame(stats)

# ============================================================
# 【备用模块】分段误差统计图 (按需取消注释使用)
# ============================================================
# def plot_binned_metrics(y_true, y_pred, patch_size, output_dir):
#     """绘制按水深分段的误差统计图，用于分析模型在不同深度的反演数值差异"""
#     stats_df = depth_binned_statistics(y_true, y_pred)
#     
#     fig, ax1 = plt.subplots(figsize=(8, 5))
# 
#     # 绘制 RMSE (折线图)
#     ax1.plot(stats_df["depth_center"], stats_df["RMSE"], 'o-', color='tab:red', label='RMSE')
#     ax1.set_xlabel('Water Depth (m)')
#     ax1.set_ylabel('RMSE (m)', color='tab:red')
#     ax1.tick_params(axis='y', labelcolor='tab:red')
# 
#     # 绘制 Bias (柱状图，使用双 Y 轴)
#     ax2 = ax1.twinx()
#     ax2.bar(stats_df["depth_center"], stats_df["Bias"], width=150, alpha=0.3, color='tab:blue', label='Bias')
#     ax2.set_ylabel('Bias (m)', color='tab:blue')
#     ax2.tick_params(axis='y', labelcolor='tab:blue')
#     ax2.axhline(0, color='black', lw=1, ls='--')
# 
#     plt.title(f'Error Analysis by Depth Bin (Patch {patch_size})')
#     plt.tight_layout()
#     plt.savefig(f"{output_dir}/depth_binned_error.png", dpi=300, bbox_inches='tight')
#     plt.close()
#     print("    -> 分段误差图已保存。")

# ============================================================
# 模块一：单模型全面精度分析 (基于当前 PATCH_SIZE)
# ============================================================
def analyze_single_model_results(patch_size=PATCH_SIZE):
    """
    基于测试集 CSV 文件，绘制优化后的精度评估图
    """
    csv_path = f"./checkpoints/dual_stream/{patch_size}/test_results.csv"
    output_dir = f"./result/dual_stream/{patch_size}"
    
    print(f"\n[模块一] 正在处理单模型全面精度分析 (Patch: {patch_size}x{patch_size})...")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    if not Path(csv_path).exists():
        print(f"    [跳过] 找不到 {csv_path}。请先运行该感受野的训练脚本！")
        return

    df = pd.read_csv(csv_path)
    y_true, y_pred = df["y_true"].values, df["y_pred"].values
    lon, lat = df["lon"].values, df["lat"].values
    residual = y_pred - y_true

    # 1. 指标计算
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    bias = np.mean(residual)
    std = np.std(residual)
    r2 = r2_score(y_true, y_pred)

    # --- 图 1：Hexbin 密度散点图 ---
    plt.figure(figsize=(8, 7))
    hb = plt.hexbin(y_true, y_pred, gridsize=100, bins='log', cmap='Spectral_r', mincnt=1)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, "r--", lw=1.5, label="Perfect Match")
    cb = plt.colorbar(hb, label='Log10(数量)')
    plt.xlabel("船测真实水深 (m)")
    plt.ylabel("模型预测水深 (m)")
    plt.title(f"精度散点图")

    textstr = f"MAE: {mae:.2f} m\nRMSE: {rmse:.2f} m\nSTD: {std:.2f} m\nBias: {bias:.2f} m\n$R^2$: {r2:.3f}"
    props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
    plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
                   verticalalignment='top', bbox=props)
    plt.grid(alpha=0.3)
    plt.savefig(f"{output_dir}/scatter_hexbin_density.png", dpi=300, bbox_inches='tight')
    plt.close()

    # --- 修改 2：真实采样点空间残差分布图（散点形式，使用 rainbow 色带） ---
    print("    -> 正在进行空间残差散点分布渲染...")
    plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # 确定颜色刻度（取 95 分位数对称分布），突出误差范围
    vmax = np.percentile(np.abs(residual), 95)
    
    # 删去插值，直接绘制散点。色带改为 rainbow，s=5控制点大小
    sc = ax.scatter(lon, lat, c=residual, cmap='rainbow', s=3, 
                    vmin=-vmax, vmax=vmax, transform=ccrs.PlateCarree())

    # 添加海岸线和陆地填充
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, color='black', zorder=3)
    land_feature = cfeature.NaturalEarthFeature('physical', 'land', '10m',
                                                edgecolor='face',
                                                facecolor='lightgray')
    ax.add_feature(land_feature, zorder=2)

    # 网格线
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0.3)
    gl.top_labels = False
    gl.right_labels = False

    plt.colorbar(sc, label='残差 (m)', fraction=0.035, pad=0.04)
    plt.title(f'空间残差分布图 (Scatter Points)\nPatch: {patch_size}, RMSE: {rmse:.2f} m')
    
    plt.savefig(f"{output_dir}/spatial_scatter_residual_rainbow.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # --- 图 3：误差直方图 ---
    plt.figure(figsize=(6, 4))
    plt.hist(residual, bins=100, density=True, alpha=0.7, color='steelblue')
    plt.axvline(0, color="r", linestyle="--", label="Zero Error")
    plt.xlabel("Residual (Pred - GT) [m]")
    plt.ylabel("Density")
    plt.title(f"Error Distribution (Bias={bias:.2f} m)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(f"{output_dir}/residual_histogram.png", dpi=300, bbox_inches='tight')
    plt.close()

    # --- 调用备用分段误差图 (已注释) ---
    # plot_binned_metrics(y_true, y_pred, patch_size, output_dir)

    print("    -> 模块一分析完成，优化后的图件已保存。")

# ============================================================
# 模块二：全域网格推理 (基于当前 PATCH_SIZE)
# ============================================================
def run_full_inference(patch_size=PATCH_SIZE):
    """利用优化后的 InferenceDataset 进行全域水深推断"""
    print(f"\n[模块二] 执行全域网格滑动推理 (Patch: {patch_size}x{patch_size})...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
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
        "g_north_bp": Path("./data/processed/G_North_BP.nc")
    }
    
    model_path = f"./checkpoints/dual_stream/{patch_size}/best_model.pt"
    NORM_PARAMS = f"./checkpoints/dual_stream/{patch_size}/normalization_params.json"
    OUTPUT_DIR = Path(f"./predictions/dual_stream/{patch_size}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if not Path(model_path).exists():
        print(f"    [错误] 找不到模型权重 {model_path}，请先训练！")
        return None

    dataset = BathymetryInferenceDataset(
        feature_paths=PATHS, normalization_params_path=NORM_PARAMS,
        cache_dir=f"./cache/inference_dual_{patch_size}",
        lon_range=(104, 122), lat_range=(0, 26), patch_size=patch_size
    )
    
    with open(NORM_PARAMS, 'r') as f:
        input_dim = len(json.load(f)['x_means'])
    
    model = DualStreamFusionNet(in_channels=input_dim, patch_size=patch_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    print("    -> 开始网络前向传播...")
    with torch.no_grad():
        patches_tensor = dataset.patches_tensor
        batch_size = 4096
        predictions = []
        for i in range(0, len(patches_tensor), batch_size):
            batch = patches_tensor[i:i+batch_size].to(device)
            predictions.append(model(batch).cpu())
        predictions = torch.cat(predictions, dim=0)
    
    print("    -> 深度反归一化并重构地理网格...")
    predictions_real = dataset.inverse_transform(predictions.numpy().flatten())
    depth_grid = dataset.reconstruct_grid(predictions_real)
    
    ds = xr.Dataset(
        {"predicted_depth": (("lat", "lon"), depth_grid)}, 
        coords={"lon": dataset.lon, "lat": dataset.lat}
    )
    
    nc_out = OUTPUT_DIR / "predicted_bathymetry_dual.nc"
    ds.to_netcdf(nc_out)
    
    # 全域深度图预览 (此处地形依然使用 terrain 色带较好，如果要统一风格也可以改成 rainbow)
    plt.figure(figsize=(10, 8))
    masked_data = np.ma.array(depth_grid, mask=np.isnan(depth_grid))
    vmin, vmax = np.percentile(depth_grid[~np.isnan(depth_grid)], [2, 98])
    im = plt.pcolormesh(dataset.lon, dataset.lat, masked_data, shading='auto', cmap='terrain', vmin=vmin, vmax=vmax)
    plt.colorbar(im, label='深度 (m)')
    plt.title(f'预测水深')
    plt.savefig(OUTPUT_DIR / "predicted_bathymetry_map.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"    -> 全域推理完成！已保存 NC 文件: {nc_out}")
    return nc_out

# ============================================================
# 主控调度
# ============================================================
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    
    # ---------------------------------------------------------
    # 测试执行 1: 基于当前 PATCH_SIZE 变量的单模型精度评估
    # ---------------------------------------------------------
    analyze_single_model_results(patch_size=PATCH_SIZE)
    
    # ---------------------------------------------------------
    # 测试执行 2: 基于当前 PATCH_SIZE 的全图推理
    # ---------------------------------------------------------
    # run_full_inference(patch_size=PATCH_SIZE)
    
    print("\n===== 脚本执行结束 =====")