import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path

from pyproj import Transformer
from scipy.interpolate import RegularGridInterpolator
# 新增：距离变换与高斯平滑
from scipy.ndimage import distance_transform_edt, gaussian_filter

# ======================
# Paths & region
# ======================
PATHS = {
    "grav_path": r"D:\project\data\SWOT\grav_SWOT_02.nc",
    "gebco_path": r"D:\project\data\GEBCO_2024\gebco_2024\GEBCO_2024.nc",
    "curv_path": r"D:\project\data\SWOT\curv_SWOT_02.nc"
}

LON_RANGE = (105, 125)
LAT_RANGE = (0, 30)

OUT_DIR = Path(r"D:\project\data\processed")
OUT_DIR.mkdir(exist_ok=True)

# ======================
# Filter parameters
# ======================
CUTOFF_WAVELENGTH_KM = 135.0   # Smith & Sandwell W1
DEPTH_KM = 3.0                # downward continuation depth
A_W2 = 81                     # Smith & Sandwell (1994)
PAD_WIDTH = 150               # FFT边缘镜像拓展像素量 (约消除四周150个像素的边缘效应)

# ======================
# Math & Filter Functions
# ======================
def radial_wavenumber(nx, ny, dx_km, dy_km):
    kx = np.fft.fftfreq(nx, d=dx_km)
    ky = np.fft.fftfreq(ny, d=dy_km)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    return np.sqrt(KX**2 + KY**2)

def gaussian_lowpass(k, cutoff_wavelength_km):
    s = cutoff_wavelength_km / 5.4
    return np.exp(-2 * (np.pi * k * s) ** 2)

def gaussian_highpass(k, cutoff_wavelength_km):
    return 1.0 - gaussian_lowpass(k, cutoff_wavelength_km)

def w2_filter(k, depth_km, A=81):
    return 1.0 / (1.0 + A * k**4 * np.exp(4 * np.pi * k * depth_km))

def downward_continuation(k, depth_km):
    return np.exp(2 * np.pi * k * depth_km)

def lowpass_filter(data, dx_km, dy_km, cutoff_wavelength_km):
    nx, ny = data.shape
    k = radial_wavenumber(nx, ny, dx_km, dy_km)
    W1 = gaussian_lowpass(k, cutoff_wavelength_km)

    F = np.fft.fft2(data)
    return np.real(np.fft.ifft2(F * W1))

def bandpass_filter(data, dx_km, dy_km, cutoff_wavelength_km, depth_km, A):
    nx, ny = data.shape
    k = radial_wavenumber(nx, ny, dx_km, dy_km)

    HP = gaussian_highpass(k, cutoff_wavelength_km)
    W2 = w2_filter(k, depth_km, A)
    DC = downward_continuation(k, depth_km)

    F = np.fft.fft2(data)
    return np.real(np.fft.ifft2(F * HP * W2 * DC))

# ======================
# 新增：GEBCO 水深专用陆地填充法
# ======================
def fill_land_gebco(data, sigma=3):
    """
    针对地形数据的物理特性：将陆地设为0，并对沿海台阶进行轻微高斯平滑。
    海洋原始数据将被100%保留。
    """
    mask = np.isnan(data)
    if not mask.any(): return data
    
    # 1. 陆地全部强制设为 0（海平面基准）
    filled = data.copy()
    filled[mask] = 0.0
    
    # 2. 轻微平滑，消除 0 到 近海浅水 的微小台阶
    smoothed = gaussian_filter(filled, sigma=sigma)
    
    # 3. 完美缝合：原封不动保留真实海洋数据，仅改变陆地部分
    final = data.copy()
    final[mask] = smoothed[mask]
    return final

# ======================
# 新增：SWOT 重力/梯度专用陆地填充法
# ======================
def fill_land_swot(data, sigma=15):
    """
    针对重力场的连续性特性：使用最近邻海洋值强行外推填满陆地，再辅以重度平滑消除内陆接缝。
    海洋原始数据将被100%保留。
    """
    mask = np.isnan(data)
    if not mask.any(): return data
    
    # 1. 找到陆地像素对应的最近海洋像素索引
    _, indices = distance_transform_edt(mask, return_indices=True)
    
    # 2. 用最近的海洋真实数值瞬间填满陆地（消除海岸线落差）
    filled = data[tuple(indices)]
    
    # 3. 重度高斯平滑，熨平大陆内部的“拼贴缝隙”
    smoothed = gaussian_filter(filled, sigma=sigma)
    
    # 4. 完美缝合：原封不动保留真实海洋数据，仅改变陆地部分
    final = data.copy()
    final[mask] = smoothed[mask]
    return final

# ======================
# 新增：边缘镜像拓展 (替代 Tukey)
# ======================
def pad_array(data, pad_width=150):
    """边缘对称镜像拓展，满足 FFT 周期性且不改变边界真实物理特征"""
    return np.pad(data, pad_width, mode='symmetric')

def unpad_array(data, pad_width=150):
    """裁切掉拓展的边缘，恢复原始尺寸"""
    return data[pad_width:-pad_width, pad_width:-pad_width]

# ======================
# 投影相关函数
# ======================
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

    return data_merc, x_uni, y_uni, dx_km, dy_km

def project_to_lonlat(x_uni, y_uni, data_merc, lon, lat):
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3395", always_xy=True)
    lon_mesh, lat_mesh = np.meshgrid(lon, lat)
    x_target, y_target = transformer.transform(lon_mesh, lat_mesh)

    interp = RegularGridInterpolator((y_uni, x_uni), data_merc, bounds_error=False, fill_value=np.nan)
    data_lonlat = interp((y_target, x_target))

    return data_lonlat

# ======================
# Data Loading & Utility
# ======================
def load_and_crop(path, var_name):
    ds = xr.open_dataset(path)
    da = ds[var_name]
    da = da.sel(
        lon=slice(LON_RANGE[0], LON_RANGE[1]),
        lat=slice(LAT_RANGE[0], LAT_RANGE[1])
    )
    return da

def save_nc(data, name, unit, lon, lat):
    da = xr.DataArray(
        data,
        coords={"lat": lat, "lon": lon},
        dims=("lat", "lon"),
        name=name
    )
    da.attrs["units"] = unit
    da.to_netcdf(OUT_DIR / f"{name}.nc")

# ======================
# 1. 加载数据与建立原始掩膜
# ======================
print("1. 正在加载数据并提取海岸线掩膜...")
bathy = load_and_crop(PATHS["gebco_path"], var_name="elevation")
grav = load_and_crop(PATHS["grav_path"], var_name="z")
curv = load_and_crop(PATHS["curv_path"], var_name="z")

lon = grav.lon.values
lat = grav.lat.values

bathy_ocean_mask = bathy.values < 0 
mask_da = xr.DataArray(
    bathy_ocean_mask.astype(np.int8),
    coords={"lat": bathy.lat, "lon": bathy.lon},
    dims=("lat", "lon")
)
mask_on_swot = mask_da.interp(lon=grav.lon, lat=grav.lat, method="nearest")
ocean_mask = mask_on_swot.values.astype(bool)

B = bathy.interp(lon=grav.lon, lat=grav.lat, method="linear").values
B[~ocean_mask] = np.nan

G = grav.values.copy()
VGG = curv.values.copy()

# ======================
# 2. 投影到墨卡托物理空间
# ======================
print("2. 正在投影至墨卡托坐标系...")
B_merc, x_uni, y_uni, dx_km, dy_km = project_to_mercator(lon, lat, B)
G_merc, _, _, _, _ = project_to_mercator(lon, lat, G)
VGG_merc, _, _, _, _ = project_to_mercator(lon, lat, VGG)

# ======================
# 3. 针对不同物理属性定制陆地平滑填充
# ======================
print("3. 正在执行定制化陆地平滑填充...")
B_filled = fill_land_gebco(B_merc, sigma=3)       # 水深专用：赋0+轻微平滑
G_filled = fill_land_swot(G_merc, sigma=15)       # 重力专用：近邻延伸+重度平滑
VGG_filled = fill_land_swot(VGG_merc, sigma=15)   # 梯度专用：近邻延伸+重度平滑

# ======================
# 4. 外部边缘镜像拓展 (替代 Tukey)
# ======================
print(f"4. 正在执行边界镜像拓展 (Pad={PAD_WIDTH} pixels)...")
B_ready = pad_array(B_filled, pad_width=PAD_WIDTH)
G_ready = pad_array(G_filled, pad_width=PAD_WIDTH)
VGG_ready = pad_array(VGG_filled, pad_width=PAD_WIDTH)

# ======================
# 5. 在物理空间中进行 FFT Filtering
# ======================
print("5. 正在执行傅里叶频域滤波...")
# 注意：滤波使用的是扩展后的大网格，FFT内部会自动调整新的 nx, ny
B_LP_padded = lowpass_filter(B_ready, dx_km, dy_km, CUTOFF_WAVELENGTH_KM)
G_LP_padded = lowpass_filter(G_ready, dx_km, dy_km, CUTOFF_WAVELENGTH_KM)

G_BP_padded = bandpass_filter(
    G_ready, dx_km, dy_km,
    CUTOFF_WAVELENGTH_KM, DEPTH_KM, A_W2
)

VGG_BP_padded = bandpass_filter(
    VGG_ready, dx_km, dy_km,
    CUTOFF_WAVELENGTH_KM, DEPTH_KM, A_W2
)

# ======================
# 5.5 裁切镜像拓展边缘
# ======================
print("5.5 正在裁切冗余的镜像边缘...")
B_LP_merc = unpad_array(B_LP_padded, pad_width=PAD_WIDTH)
G_LP_merc = unpad_array(G_LP_padded, pad_width=PAD_WIDTH)
G_BP_merc = unpad_array(G_BP_padded, pad_width=PAD_WIDTH)
VGG_BP_merc = unpad_array(VGG_BP_padded, pad_width=PAD_WIDTH)

# ======================
# 6. 反插值回等经纬度网格
# ======================
print("6. 滤波完成，正在重投影回原始经纬度网格...")
B_LP = project_to_lonlat(x_uni, y_uni, B_LP_merc, lon, lat)
G_LP = project_to_lonlat(x_uni, y_uni, G_LP_merc, lon, lat)
G_BP = project_to_lonlat(x_uni, y_uni, G_BP_merc, lon, lat)
VGG_BP = project_to_lonlat(x_uni, y_uni, VGG_BP_merc, lon, lat)

# ======================
# 7. 重新施加掩膜并保存结果
# ======================
print("7. 正在剥离陆地及边缘冗余数据并保存...")
# 去除陆地和插值边缘的冗余计算结果
B_LP[~ocean_mask] = np.nan
G_LP[~ocean_mask] = np.nan
G_BP[~ocean_mask] = np.nan
VGG_BP[~ocean_mask] = np.nan

B_ocean_raw = B.copy()
G_ocean = G.copy()
VGG_ocean = VGG.copy()

G_ocean[~ocean_mask] = np.nan
VGG_ocean[~ocean_mask] = np.nan

save_nc(B_LP, "B_LP", "m", lon, lat)
save_nc(G_LP, "G_LP", "mGal", lon, lat)
save_nc(G_BP, "G_BP", "mGal", lon, lat)
save_nc(VGG_BP, "VGG_BP", "s^-2", lon, lat)
save_nc(G_ocean, "G_ocean_raw", "mGal", lon, lat)
save_nc(VGG_ocean, "VGG_ocean_raw", "s^-2", lon, lat)
save_nc(B_ocean_raw, "B_ocean_raw", "m", lon, lat)

print("处理完成！你可以打开NC文件查看极致纯净的对比效果。")