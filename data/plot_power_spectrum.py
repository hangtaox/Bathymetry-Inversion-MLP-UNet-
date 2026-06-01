import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path

from pyproj import Transformer
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt, gaussian_filter
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# ======================
# 路径与范围配置
# ======================
PATHS = {
    "grav_path": r"D:\project\data\SWOT\grav_SWOT_02.nc",
    "gebco_path": r"D:\project\data\GEBCO_2024\gebco_2024\GEBCO_2024.nc",
    "curv_path": r"D:\project\data\SWOT\curv_SWOT_02.nc",
    "north_path": r"D:\project\data\SWOT\north_SWOT_04.nc",  # 新增: North 偏向数据
    "east_path": r"D:\project\data\SWOT\east_SWOT_04.nc"     # 新增: East 偏向数据
}

LON_RANGE = (104, 122)
LAT_RANGE = (0, 26)
PAD_WIDTH = 150  # 必须与你处理代码中的Pad保持一致

# ======================
# 预处理函数 (完全复用你的正式处理逻辑)
# ======================
def fill_land_gebco(data, sigma=3):
    mask = np.isnan(data)
    if not mask.any(): return data
    filled = data.copy()
    filled[mask] = 0.0
    smoothed = gaussian_filter(filled, sigma=sigma)
    final = data.copy()
    final[mask] = smoothed[mask]
    return final

def fill_land_swot(data, sigma=15):
    mask = np.isnan(data)
    if not mask.any(): return data
    _, indices = distance_transform_edt(mask, return_indices=True)
    filled = data[tuple(indices)]
    smoothed = gaussian_filter(filled, sigma=sigma)
    final = data.copy()
    final[mask] = smoothed[mask]
    return final

def pad_array(data, pad_width=150):
    return np.pad(data, pad_width, mode='symmetric')

def project_to_mercator(lon, lat, data):
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3395", always_xy=True)
    x_orig, _ = transformer.transform(lon, np.zeros_like(lon))
    _, y_orig = transformer.transform(np.zeros_like(lat), lat)
    ny, nx = data.shape
    x_uni = np.linspace(x_orig.min(), x_orig.max(), nx)
    y_uni = np.linspace(y_orig.min(), y_orig.max(), ny)
    interp = RegularGridInterpolator((y_orig, x_orig), data, bounds_error=False, fill_value=np.nan)
    X_uni, Y_uni = np.meshgrid(x_uni, y_uni)
    data_merc = interp((Y_uni, X_uni))
    dx_km = (x_uni[1] - x_uni[0]) / 1000.0
    dy_km = (y_uni[1] - y_uni[0]) / 1000.0
    return data_merc, dx_km, dy_km

def load_and_crop(path, var_name):
    ds = xr.open_dataset(path)
    da = ds[var_name]
    return da.sel(lon=slice(LON_RANGE[0], LON_RANGE[1]), lat=slice(LAT_RANGE[0], LAT_RANGE[1]))

# ======================
# 新增：二维功率谱径向平均计算核心
# ======================
def compute_radial_spectrum(data, dx_km, dy_km):
    """计算二维数据的径向平均功率谱"""
    nx, ny = data.shape
    # 1. 计算二维波数矩阵
    kx = np.fft.fftfreq(nx, d=dx_km)
    ky = np.fft.fftfreq(ny, d=dy_km)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    K_radial = np.sqrt(KX**2 + KY**2)
    
    # 2. 计算二维 FFT 及功率 (振幅的平方)
    F = np.fft.fft2(data)
    Power = np.abs(F)**2
    
    # 3. 展平并按波数大小排序
    K_flat = K_radial.ravel()
    P_flat = Power.ravel()
    
    # 排除 DC 分量 (k=0)
    valid = K_flat > 0
    K_flat = K_flat[valid]
    P_flat = P_flat[valid]
    
    sort_idx = np.argsort(K_flat)
    K_sorted = K_flat[sort_idx]
    P_sorted = P_flat[sort_idx]
    
    # 4. 离散化区间 (Binning)，对同一波数环上的能量求平均
    k_bins = np.linspace(K_sorted.min(), K_sorted.max(), 300)
    k_centers = 0.5 * (k_bins[1:] + k_bins[:-1])
    indices = np.digitize(K_sorted, k_bins)
    
    radial_p = np.zeros_like(k_centers)
    for i in range(1, len(k_bins)):
        mask = indices == i
        if np.any(mask):
            radial_p[i-1] = np.mean(P_sorted[mask])
        else:
            radial_p[i-1] = np.nan
            
    return k_centers, radial_p

# ======================
# 执行主程序
# ======================
print("加载并预处理数据中...")
bathy = load_and_crop(PATHS["gebco_path"], var_name="elevation")
grav = load_and_crop(PATHS["grav_path"], var_name="z")
curv = load_and_crop(PATHS["curv_path"], var_name="z")
north = load_and_crop(PATHS["north_path"], var_name="z")  # 新增加载
east = load_and_crop(PATHS["east_path"], var_name="z")    # 新增加载

lon, lat = grav.lon.values, grav.lat.values

bathy_ocean_mask = bathy.values < 0 
mask_da = xr.DataArray(bathy_ocean_mask.astype(np.int8), coords={"lat": bathy.lat, "lon": bathy.lon}, dims=("lat", "lon"))
mask_on_swot = mask_da.interp(lon=grav.lon, lat=grav.lat, method="nearest")
ocean_mask = mask_on_swot.values.astype(bool)

B = bathy.interp(lon=grav.lon, lat=grav.lat, method="linear").values
B[~ocean_mask] = np.nan
G, VGG = grav.values.copy(), curv.values.copy()
N, E = north.values.copy(), east.values.copy()  # 新增提取数值

print("投影与平滑填充中...")
B_merc, dx, dy = project_to_mercator(lon, lat, B)
G_merc, _, _ = project_to_mercator(lon, lat, G)
VGG_merc, _, _ = project_to_mercator(lon, lat, VGG)
N_merc, _, _ = project_to_mercator(lon, lat, N)  # 新增投影
E_merc, _, _ = project_to_mercator(lon, lat, E)  # 新增投影

B_ready = pad_array(fill_land_gebco(B_merc, sigma=3), PAD_WIDTH)
G_ready = pad_array(fill_land_swot(G_merc, sigma=15), PAD_WIDTH)
VGG_ready = pad_array(fill_land_swot(VGG_merc, sigma=15), PAD_WIDTH)
N_ready = pad_array(fill_land_swot(N_merc, sigma=15), PAD_WIDTH)  # 新增平滑填充
E_ready = pad_array(fill_land_swot(E_merc, sigma=15), PAD_WIDTH)  # 新增平滑填充

print("计算功率谱中...")
k_B, p_B = compute_radial_spectrum(B_ready, dx, dy)
k_G, p_G = compute_radial_spectrum(G_ready, dx, dy)
k_V, p_V = compute_radial_spectrum(VGG_ready, dx, dy)
k_N, p_N = compute_radial_spectrum(N_ready, dx, dy)  # 新增计算
k_E, p_E = compute_radial_spectrum(E_ready, dx, dy)  # 新增计算

# ======================
# 绘图逻辑 (合并为单张图展示)
# ======================
print("正在绘制合并功率谱图...")
fig, ax = plt.subplots(1, 1, figsize=(12, 8), constrained_layout=True)

# 定义数据集，指定颜色和图例标签。功率谱的单位通常为原单位的平方。
datasets = [
    (k_B, p_B, "GEBCO 水深", "m^2", "#1f77b4"),             # 蓝色
    (k_G, p_G, "重力异常", "mGal^2", "#2ca02c"),      # 绿色
    (k_V, p_V, "垂直重力梯度", "Eotvos^2", "#d62728"),                # 红色
    (k_N, p_N, "北向垂线偏差", "microradian^2", "#ff7f0e"), # 橙色 (新增)
    (k_E, p_E, "东向垂线偏差", "microradian^2", "#9467bd")   # 紫色 (新增)
]

for k, p, label, unit, color in datasets:
    # 过滤无效值并计算波长与log10(Power)
    valid = ~np.isnan(p) & (p > 0) & (k > 0)
    lam = 1.0 / k[valid]
    log_p = np.log10(p[valid])
    
    # 绘制以波长为横坐标的折线
    ax.plot(lam, log_p, color=color, linewidth=2.0, alpha=0.9, label=f"{label} (log10 {unit})")

# 标题与标签设置
ax.set_title("谱功率分析", fontsize=16, fontweight='bold', pad=15)
ax.set_xlabel("波长 (km)", fontsize=14, fontweight='bold')
ax.set_ylabel("log10 [ 谱功率 ]", fontsize=14, fontweight='bold')

# 将X轴设为对数刻度，并反转坐标轴（左侧长波 1000km，右侧短波 2km）
ax.set_xscale('log')
ax.invert_xaxis()

# 设置直观的波长刻度标签 (加入了我们在意的 4km 极限)
ticks_km = [1000, 500, 200, 100, 50, 20, 10, 4, 2]
ax.set_xticks(ticks_km)
ax.set_xticklabels([str(t) for t in ticks_km])

# 添加网格线，方便对齐刻度寻找拐点
ax.grid(True, which="major", ls="-", alpha=0.7)
ax.grid(True, which="minor", ls="--", alpha=0.3)

# 标注出我们讨论过的物理关键节点作为垂直参考线
# ax.axvline(x=160, color='gray', linestyle='--', linewidth=2, alpha=0.8, label='Long Cutoff (160 km)')
# ax.axvline(x=5, color='purple', linestyle='--', linewidth=2, alpha=0.8, label='Nyquist/Noise Cutoff (4 km)')

# 优化图例位置和样式
ax.legend(loc='upper right', fontsize=12, framealpha=0.9)

# 保存并显示
plt.savefig("Combined_Power_Spectra.png", dpi=300)
plt.show()
print("合并绘图完成！")