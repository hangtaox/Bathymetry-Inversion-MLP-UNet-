import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
import os
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# ==========================================
# 1. 路径与参数设置
# ==========================================
SWOT_NC_PATH = r"D:\project\data\SWOT\grav_SWOT_02.nc"   # 用于获取全区参考网格
SB_TXT_PATH = r"D:\project\tichu_nanhaisbes.txt"         # 单波束数据
MB_TXT_PATH = r"D:\project\tichu_nanhaimbes.txt"         # 多波束数据

# 缓存目录与缓存文件名
CACHE_DIR = Path(r"D:\project\data\cache")
SB_CACHE_PATH = CACHE_DIR / "grid_sb_cache.nc"
MB_CACHE_PATH = CACHE_DIR / "grid_mb_cache.nc"

AGG_METHOD = "median"  # 聚合方法 ("median" 或 "mean")

# ==========================================
# 2. 核心函数定义
# ==========================================
def read_ship_txt(path):
    """读取船测TXT，假设格式为: lon, lat, depth"""
    data = np.loadtxt(path, delimiter=",") # 如果是空格分隔，请去掉 delimiter=","
    lon = data[:, 0]
    lat = data[:, 1]
    depth = data[:, 2]
    return lon, lat, depth

def grid_points_to_swot(lon_pts, lat_pts, depth_pts, ref_lon, ref_lat, agg_method="median"):
    """将散点分配到全区 SWOT 网格"""
    H, W = len(ref_lat), len(ref_lon)
    grid = np.full((H, W), np.nan)
    
    def lonlat_to_ij(lo, la):
        j = np.searchsorted(ref_lon, lo) - 1
        i = np.searchsorted(ref_lat, la) - 1
        return i, j

    tmp = {}
    for lo, la, d in zip(lon_pts, lat_pts, depth_pts):
        i, j = lonlat_to_ij(lo, la)
        if 0 <= i < H and 0 <= j < W:
            tmp.setdefault((i, j), []).append(d)

    for (i, j), v in tmp.items():
        grid[i, j] = np.median(v) if agg_method == "median" else np.mean(v)

    return grid

# ==========================================
# 3. 智能缓存与网格化执行
# ==========================================
# 获取 SWOT 全区参考坐标
print("1. 获取 SWOT 参考网格坐标...")
ds_swot = xr.open_dataset(SWOT_NC_PATH)
ref_lon = ds_swot.lon.values
ref_lat = ds_swot.lat.values
ds_swot.close()

CACHE_DIR.mkdir(parents=True, exist_ok=True)

# 判断缓存是否存在
if SB_CACHE_PATH.exists() and MB_CACHE_PATH.exists():
    print(f"2. 发现缓存文件，直接加载网格化数据... (秒读)")
    ds_sb = xr.open_dataset(SB_CACHE_PATH)
    ds_mb = xr.open_dataset(MB_CACHE_PATH)
    grid_sb = ds_sb["ship_depth"].values
    grid_mb = ds_mb["ship_depth"].values
    ds_sb.close()
    ds_mb.close()
else:
    print(f"2. 未发现缓存，开始读取原始 TXT 并执行网格化 (这可能需要几分钟)...")
    lon_sb, lat_sb, depth_sb = read_ship_txt(SB_TXT_PATH)
    lon_mb, lat_mb, depth_mb = read_ship_txt(MB_TXT_PATH)
    
    print("   -> 正在网格化单波束(SB)数据...")
    grid_sb = grid_points_to_swot(lon_sb, lat_sb, depth_sb, ref_lon, ref_lat, AGG_METHOD)
    print("   -> 正在网格化多波束(MB)数据...")
    grid_mb = grid_points_to_swot(lon_mb, lat_mb, depth_mb, ref_lon, ref_lat, AGG_METHOD)
    
    print("   -> 正在保存为 NetCDF 缓存文件...")
    xr.Dataset(
        {"ship_depth": (("lat", "lon"), grid_sb)},
        coords={"lon": ref_lon, "lat": ref_lat}
    ).to_netcdf(SB_CACHE_PATH)
    
    xr.Dataset(
        {"ship_depth": (("lat", "lon"), grid_mb)},
        coords={"lon": ref_lon, "lat": ref_lat}
    ).to_netcdf(MB_CACHE_PATH)
    print("   -> 缓存保存成功！")

# ==========================================
# 4. 全域提取重合点与线性拟合
# ==========================================
print("3. 正在全域搜索单波束与多波束的网格重合点...")
# 寻找同网格点下都有数据的像素 (交集)
valid_mask = ~np.isnan(grid_sb) & ~np.isnan(grid_mb)

x_sb = grid_sb[valid_mask]  # 自变量: 单波束深度
y_mb = grid_mb[valid_mask]  # 因变量: 多波束深度 (基准真值)

num_points = len(x_sb)
print(f"-> 在全区范围内共找到 {num_points} 个 SWOT 网格重合点。")

if num_points < 10:
    print("重合点太少，无法进行有效拟合！")
else:
    # 简单的异常值剔除：去掉单/多波束相差超过 1000 米的绝对噪点 (避免个别错误点拉偏拟合线)
    error_mask = np.abs(x_sb - y_mb) < 1000
    x_sb_clean = x_sb[error_mask]
    y_mb_clean = y_mb[error_mask]
    print(f"-> 剔除绝对粗差后，参与拟合的有效点数: {len(x_sb_clean)}")

    # 线性回归: y_mb = slope * x_sb + intercept
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_sb_clean, y_mb_clean)
    r_squared = r_value ** 2
    
    print("=" * 50)
    print("🌟 全域拟合结果 (以多波束为基准修正单波束):")
    print(f"修正公式: 多波束测量水深 = {slope:.5f} * 单波束测量水深 + {intercept:.3f}")
    print(f"R² 决定系数: {r_squared:.4f}")
    print("=" * 50)

    # ==========================================
    # 5. 绘制专业的密度散点图与拟合曲线
    # ==========================================
    plt.figure(figsize=(9, 7))
    
    # 使用 hexbin 绘制密度图，解决大样本点重叠发黑的问题
    hb = plt.hexbin(x_sb_clean, y_mb_clean, gridsize=100, cmap='Spectral_r', mincnt=1, bins='log')
    cb = plt.colorbar(hb)
    cb.set_label('log10(数据点数量)', fontsize=11)
    
    # 绘制 1:1 绝对参考线
    min_val = min(x_sb_clean.min(), y_mb_clean.min())
    max_val = max(x_sb_clean.max(), y_mb_clean.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1.5, label='参考线')
    
    # 绘制经验拟合线
    x_fit = np.linspace(min_val, max_val, 100)
    y_fit = slope * x_fit + intercept
    plt.plot(x_fit, y_fit, 'r-', lw=2.5, 
             label=f'拟合曲线: y = {slope:.4f}x {intercept:.2f}\n$R^2$ = {r_squared:.4f}')
    
    plt.title('单波束与多波束散点图', fontsize=14, fontweight='bold')
    plt.xlabel('单波束水深 (m)', fontsize=12)
    plt.ylabel('多波束水深 (m)', fontsize=12)
    plt.legend(loc='upper left', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig('Global_SB_MB_Calibration.png', dpi=300)
    plt.show()
    
    print("绘图完成！请查看保存的 Global_SB_MB_Calibration.png")