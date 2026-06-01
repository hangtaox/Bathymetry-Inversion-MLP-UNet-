import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import norm, pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xarray as xr
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# =========================================================
# 1. 配置文件路径
# =========================================================
MODEL_CSV_PATHS = {
    "MLP": "./checkpoints/mlp/test_results.csv",
    "CNN": "./checkpoints/cnn/9/test_results.csv",
    "CNN-MLP": "./checkpoints/dual_stream/13/test_results.csv"
}

PREDICTED_NC_PATH = "./predictions/dual_stream/13/predicted_bathymetry_dual.nc"
RAW_SHIP_NC_PATH = "./data/ship_combined_calibrated.nc" 

CACHE_DIR = Path("D:/project/data/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

GEBCO_NC_PATH = r"D:\project\data\GEBCO_2024\gebco_2024\GEBCO_2024.nc"
GEBCO_CSV_PATH = "./checkpoints/gebco/test_results.csv"

OUTPUT_DIR = "./result/comparative_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================================
# 辅助模块：智能获取船测对齐网格
# =========================================================
def get_ship_grid_with_cache(raw_ship_nc, target_lon, target_lat):
    lon_str = f"{target_lon.min():.2f}_{target_lon.max():.2f}"
    lat_str = f"{target_lat.min():.2f}_{target_lat.max():.2f}"
    cache_name = f"aligned_ship_grid_{lon_str}_{lat_str}_size{len(target_lon)}x{len(target_lat)}.nc"
    cache_path = CACHE_DIR / cache_name

    if cache_path.exists():
        ds_c = xr.open_dataset(cache_path)
        if np.array_equal(ds_c.lon.values, target_lon) and np.array_equal(ds_c.lat.values, target_lat):
            print(f"  -> [读取缓存] 找到匹配的船测对齐网格: {cache_name}")
            grid_data = ds_c['ship_depth'].load()
            ds_c.close()
            return grid_data
        ds_c.close()

    print(f"  -> [重新对齐] 正在执行散点到网格的高效对齐...")
    ds = xr.open_dataset(raw_ship_nc)
    s_lon, s_lat, s_depth = ds["lon"].values, ds["lat"].values, ds["depth"].values
    s_source = ds["source"].values if "source" in ds else np.zeros_like(s_depth)
    ds.close()

    mask = (s_lon >= target_lon.min()) & (s_lon <= target_lon.max()) & \
           (s_lat >= target_lat.min()) & (s_lat <= target_lat.max())
    s_lon, s_lat, s_depth, s_source = s_lon[mask], s_lat[mask], s_depth[mask], s_source[mask]

    H, W = len(target_lat), len(target_lon)
    grid = np.full((H, W), np.nan, dtype=np.float32)
    d_lon, d_lat = target_lon[1] - target_lon[0], target_lat[1] - target_lat[0]
    j_indices = np.round((s_lon - target_lon[0]) / d_lon).astype(int)
    i_indices = np.round((s_lat - target_lat[0]) / d_lat).astype(int)
    
    valid = (j_indices >= 0) & (j_indices < W) & (i_indices >= 0) & (i_indices < H)
    j_indices, i_indices, depth_v, source_v = j_indices[valid], i_indices[valid], s_depth[valid], s_source[valid]

    tmp_mb, tmp_sb = {}, {}
    for i_idx, j_idx, d, src in zip(i_indices, j_indices, depth_v, source_v):
        if src == 1: tmp_mb.setdefault((i_idx, j_idx), []).append(d)
        else: tmp_sb.setdefault((i_idx, j_idx), []).append(d)

    for i in range(H):
        for j in range(W):
            if (i, j) in tmp_mb: grid[i, j] = np.median(tmp_mb[(i, j)])
            elif (i, j) in tmp_sb: grid[i, j] = np.median(tmp_sb[(i, j)])

    new_da = xr.DataArray(grid, coords={"lat": target_lat, "lon": target_lon}, dims=("lat", "lon"), name="ship_depth")
    new_da.to_netcdf(cache_path)
    return new_da

# =========================================================
# 绘图模块
# =========================================================
def create_cartopy_axes(fig, lon, lat):
    ax = fig.add_axes([0.1, 0.15, 0.8, 0.8], projection=ccrs.PlateCarree())
    ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=ccrs.PlateCarree())
    ax.set_xticks(np.arange(np.floor(lon.min()), np.ceil(lon.max())+1, 5), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(np.floor(lat.min()), np.ceil(lat.max())+1, 5), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.tick_params(axis='both', labelsize=12, direction='in', pad=5)
    ax.gridlines(draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--', zorder=5)
    return ax

def add_eagle_eye(fig, lon, lat):
    center_lon, center_lat = (lon.max() + lon.min()) / 2, (lat.max() + lat.min()) / 2
    # 按照要求的参数精确定位鹰眼图
    inset_ax = fig.add_axes([0.2, 0.85, 0.13, 0.1], projection=ccrs.Orthographic(center_lon, center_lat))
    inset_ax.set_global()
    inset_ax.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='black', linewidth=0.4)
    inset_ax.add_feature(cfeature.OCEAN, facecolor='white')
    inset_ax.add_feature(cfeature.COASTLINE, linewidth=0.3)
    inset_ax.gridlines(linewidth=0.2, color='gray', alpha=0.5)
    inset_ax.plot([lon.min(), lon.max(), lon.max(), lon.min(), lon.min()],
                  [lat.min(), lat.min(), lat.max(), lat.max(), lat.min()],
                  color='red', linewidth=1.5, transform=ccrs.PlateCarree(), zorder=5)

def plot_overlay_map(base_da, res_da, lon, lat, cmap_bathy, norm_bathy, out_filename, title_res):
    fig = plt.figure(figsize=(8, 11))
    ax = create_cartopy_axes(fig, lon, lat)
    
    # 1. 绘制底层水深背景
    im_base = ax.pcolormesh(lon, lat, base_da.values, cmap=cmap_bathy, norm=norm_bathy, 
                            alpha=0.35, transform=ccrs.PlateCarree(), zorder=1, shading='nearest')
    
    # 🚀 核心修改：极高对比度、多色阶、无蓝色 自定义发散色带
    res_colors = [
        '#00FF00', # 极度低估 (亮绿)
        '#80FF00', # 中度低估 (黄绿)
        '#FFFF00', # 轻微低估 (亮黄)
        '#FFFFFF', # 准确无误 (纯白)
        '#FFA500', # 轻微高估 (亮橙)
        '#FF0000', # 中度高估 (正红)
        '#8B008B'  # 极度高估 (深紫)
    ]
    res_nodes = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
    custom_high_contrast = mcolors.LinearSegmentedColormap.from_list("custom_res", list(zip(res_nodes, res_colors)))
    
    # 取 95% 分位数计算色带上下限，避免个别异常点压缩色阶
    vmax_res = np.nanpercentile(np.abs(res_da.values), 95)
    if np.isnan(vmax_res) or vmax_res == 0: vmax_res = 50
    
    # 2. 绘制高亮残差层
    im_res = ax.pcolormesh(lon, lat, res_da.values, cmap=custom_high_contrast, vmin=-vmax_res, vmax=vmax_res, 
                           alpha=1.0, transform=ccrs.PlateCarree(), zorder=2, shading='nearest')
    
    # 3. 陆地掩膜与海岸线
    ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=3) 
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor='black', alpha=0.8, zorder=4)
    
    plt.draw() 
    for label in ax.yaxis.get_ticklabels():
        if '0' in label.get_text(): label.set_verticalalignment('bottom')

    # 单独的底部矩形残差色带
    cbar_res = plt.colorbar(im_res, ax=ax, orientation='horizontal', pad=0.06, aspect=40, shrink=0.8)
    cbar_res.set_label(title_res, fontsize=12, fontweight='bold')
    
    add_eagle_eye(fig, lon, lat)
    fig.savefig(out_filename, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_spatial_maps():
    print("\n[4/4] 正在绘制空间残差分布图...")
    ds_pred = xr.open_dataset(PREDICTED_NC_PATH)
    pred_da = ds_pred['predicted_depth']
    lon, lat = pred_da.lon.values, pred_da.lat.values
    gt_da = get_ship_grid_with_cache(RAW_SHIP_NC_PATH, lon, lat)
    
    ds_gebco = xr.open_dataset(GEBCO_NC_PATH)
    gebco_da = ds_gebco['elevation'].sel(lon=slice(lon.min()-0.5, lon.max()+0.5), lat=slice(lat.min()-0.5, lat.max()+0.5)).interp(lon=lon, lat=lat, method='linear')
    
    res_dual_da = pred_da - gt_da
    res_gebco_da = gebco_da - gt_da
    
    # 底图海洋蓝
    ocean_colors = ["#000018", "#0022B3", "#0066FF", "#00E5FF", "#E6FFFF"]
    cmap_bathy = mcolors.LinearSegmentedColormap.from_list("ocean_fade", list(zip([0, 0.35, 0.65, 0.9, 1.0], ocean_colors)))
    norm_bathy = mcolors.Normalize(vmin=-6000, vmax=0)

    # 绘制
    plot_overlay_map(pred_da, res_dual_da, lon, lat, cmap_bathy, norm_bathy, 
                     os.path.join(OUTPUT_DIR, "Map_Spatial_Residual_DualStream_vs_Ship.png"), 
                     "CNN-MLP 残差空间分布(m)")
    
    plot_overlay_map(gebco_da, res_gebco_da, lon, lat, cmap_bathy, norm_bathy, 
                     os.path.join(OUTPUT_DIR, "Map_Spatial_Residual_GEBCO_vs_Ship.png"), 
                     "GEBCO_2024 残差空间分布(m)")
    
    print("  -> 空间残差图绘制完毕（定制高对比多色阶）。")

# =========================================================
# 其他分析与制表模块
# =========================================================
def extract_gebco_predictions(reference_csv_path):
    if os.path.exists(GEBCO_CSV_PATH): return pd.read_csv(GEBCO_CSV_PATH)
    os.makedirs(os.path.dirname(GEBCO_CSV_PATH), exist_ok=True)
    ref_df = pd.read_csv(reference_csv_path)
    lons, lats, y_true = ref_df['lon'].values, ref_df['lat'].values, ref_df['y_true'].values 
    ds = xr.open_dataset(GEBCO_NC_PATH)
    ds_cropped = ds.sel(lon=slice(lons.min()-0.5, lons.max()+0.5), lat=slice(lats.min()-0.5, lats.max()+0.5))
    y_pred_gebco = ds_cropped['elevation'].interp(lon=xr.DataArray(lons, dims="points"), 
                                                  lat=xr.DataArray(lats, dims="points"), method='linear').values
    gebco_df = pd.DataFrame({"lon": lons, "lat": lats, "y_true": y_true, "y_pred": y_pred_gebco})
    gebco_df.to_csv(GEBCO_CSV_PATH, index=False)
    return gebco_df

def compute_metrics(y_true, y_pred):
    diff = y_pred - y_true
    safe_y_true = np.where(y_true == 0, 1e-5, y_true)
    return {
        "Max (m)": np.max(diff), "Min (m)": np.min(diff), "Mean (m)": np.mean(diff),             
        "STD (m)": np.std(diff), "RMS (m)": np.sqrt(mean_squared_error(y_true, y_pred)), 
        "MAE (m)": mean_absolute_error(y_true, y_pred), 
        "Correlation (%)": pearsonr(y_true, y_pred)[0] * 100, 
        "MAPE (%)": np.mean(np.abs(diff / safe_y_true)) * 100  
    }

def generate_overall_table(data_dict):
    print("\n[1/4] 正在生成全局精度总览表...")
    results = []
    for model_name, df in data_dict.items():
        metrics = compute_metrics(df['y_true'].values, df['y_pred'].values)
        metrics['Model'] = model_name
        results.append(metrics)
    pd.DataFrame(results)[['Model', 'Max (m)', 'Min (m)', 'Mean (m)', 'STD (m)', 'RMS (m)', 'MAE (m)', 'Correlation (%)', 'MAPE (%)']].to_csv(
        os.path.join(OUTPUT_DIR, "Table_1_Overall_Metrics.csv"), index=False, float_format="%.2f")

def generate_depth_binned_table(data_dict):
    print("\n[2/4] 正在生成分水深段精度对比表...")
    depth_bins = [(0, 1000), (1000, 2000), (2000, 3000), (3000, 4000), (4000, 12000)]
    all_binned_results = []
    for min_d, max_d in depth_bins:
        bin_label = f"<{max_d}m" if min_d == 0 else (f">{min_d}m" if max_d > 10000 else f"{min_d}-{max_d}m")
        for model_name, df in data_dict.items():
            y_true, y_pred = df['y_true'].values, df['y_pred'].values
            mask = (np.abs(y_true) >= min_d) & (np.abs(y_true) < max_d)
            if mask.sum() == 0: continue
            diff = y_pred[mask] - y_true[mask]
            all_binned_results.append({
                "Depth Bin": bin_label, "Number of Check Points": mask.sum(),
                "Model": model_name, "Max (m)": np.max(diff), "Min (m)": np.min(diff),
                "Mean (m)": np.mean(diff), "RMS (m)": np.sqrt(mean_squared_error(y_true[mask], y_pred[mask])),
                "STD (m)": np.std(diff), "Correlation (%)": pearsonr(y_true[mask], y_pred[mask])[0] * 100 if mask.sum() > 1 else np.nan
            })
    pd.DataFrame(all_binned_results).to_csv(os.path.join(OUTPUT_DIR, "Table_2_Depth_Binned_Metrics.csv"), index=False, float_format="%.2f")

def plot_error_histograms(data_dict):
    print("\n[3/4] 正在分别绘制误差直方图与正态分布拟合曲线...")
    for model_name, df in data_dict.items():
        diff = df['y_pred'].values - df['y_true'].values
        plt.figure(figsize=(6, 5))
        range_lim, bins_count = 500, 50
        weights = np.ones_like(diff) / len(diff) * 100
        plt.hist(diff, bins=bins_count, range=(-range_lim, range_lim), weights=weights, alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.5)
        mu, std = norm.fit(diff)
        x = np.linspace(-range_lim, range_lim, 100)
        p_percent = norm.pdf(x, mu, std) * ((2 * range_lim) / bins_count) * 100  
        plt.plot(x, p_percent, 'r', linewidth=1.5, label='正态拟合曲线.')
        plt.title(f"{model_name} 误差频率直方图")
        plt.xlabel("误差 (m)")
        plt.xlim(-range_lim, range_lim)
        plt.ylabel("频率 (%)")
        plt.ylim(0, 26) 
        plt.grid(axis='y', alpha=0.3)
        plt.axvline(0, color='black', linestyle='--', linewidth=1)
        plt.legend()
        safe_name = model_name.replace(" ", "_").replace("-", "_")
        plt.savefig(os.path.join(OUTPUT_DIR, f"Histogram_{safe_name}.png"), dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    print("====== 启动综合模型精度分析系统 ======")
    
    loaded_data = {}
    reference_csv = None
    
    for name, path in MODEL_CSV_PATHS.items():
        if os.path.exists(path):
            loaded_data[name] = pd.read_csv(path)
            print(f"成功加载模型 [{name}] 的数据 -> {len(loaded_data[name])} 个点")
            if reference_csv is None: reference_csv = path
        else:
            print(f"[警告] 找不到文件: {path}")
            
    if reference_csv:
        try:
            gebco_df = extract_gebco_predictions(reference_csv)
            loaded_data["GEBCO_2024"] = gebco_df
        except Exception as e:
            print(f"[错误] GEBCO 提取失败: {e}")
    
    if not loaded_data:
        print("[错误] 没有找到任何有效数据，退出程序。")
        exit()
        
    #generate_overall_table(loaded_data)
    #generate_depth_binned_table(loaded_data)
    plot_error_histograms(loaded_data)
    plot_spatial_maps()
    
    print("\n====== 分析完成 ======")