import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path

from pyproj import Transformer
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt, gaussian_filter

# ======================
# Paths & region
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

OUT_DIR = Path(r"D:\project\data\processed")
OUT_DIR.mkdir(exist_ok=True)

# ======================
# 逆向精确推导的滤波器参数 (为深度学习与南海水深深度定制)
# ======================
CUTOFF_LONG_KM = 160.0    # 剥离深部区域场的长波长截止
CUTOFF_SHORT_KM = 5       # 有效短波截止 (由剿灭4km处99%网格伪影反推获得)
DEPTH_KM = 1.7447         # 南海研究区真实平均水深

EXACT_S = 29.983          # 长波长高斯平滑系数
EXACT_A = 7.7895          # 短波长向下延拓镇压系数 (确保 4km 噪声被彻底剔除)
PAD_WIDTH = 150           # FFT边缘镜像拓展像素量

# ======================
# Math & Filter Functions
# ======================
def radial_wavenumber(nx, ny, dx_km, dy_km):
    kx = np.fft.fftfreq(nx, d=dx_km)
    ky = np.fft.fftfreq(ny, d=dy_km)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    return np.sqrt(KX**2 + KY**2)

def gaussian_lowpass(k, s):
    return np.exp(-2 * (np.pi * k * s) ** 2)

def gaussian_highpass(k, s):
    return 1.0 - gaussian_lowpass(k, s)

def w2_filter(k, depth_km, A):
    return 1.0 / (1.0 + A * (k**4) * np.exp(4 * np.pi * k * depth_km))

def downward_continuation(k, depth_km):
    return np.exp(2 * np.pi * k * depth_km)

def lowpass_filter(data, dx_km, dy_km, s):
    """提取波长 > 160km 的区域场长波分量"""
    nx, ny = data.shape
    k = radial_wavenumber(nx, ny, dx_km, dy_km)
    W1 = gaussian_lowpass(k, s)

    F = np.fft.fft2(data)
    return np.real(np.fft.ifft2(F * W1))

def bandpass_filter(data, dx_km, dy_km, s, depth_km, A):
    """提取波长 10.66km~160km 的带通分量，并包含严格物理向下延拓"""
    nx, ny = data.shape
    k = radial_wavenumber(nx, ny, dx_km, dy_km)

    HP = gaussian_highpass(k, s)           # 去除深部异常
    W2 = w2_filter(k, depth_km, A)         # 极限量级镇压高频噪声
    DC = downward_continuation(k, depth_km) # 物理向下延拓算子

    F = np.fft.fft2(data)
    # 组合应用: 高通 * 低通 * 向下延拓
    return np.real(np.fft.ifft2(F * HP * W2 * DC))

# ======================
# 陆地平滑与边缘拓展函数
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

def unpad_array(data, pad_width=150):
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

def load_and_crop(path, var_name):
    ds = xr.open_dataset(path)
    da = ds[var_name]
    return da.sel(lon=slice(LON_RANGE[0], LON_RANGE[1]), lat=slice(LAT_RANGE[0], LAT_RANGE[1]))

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

north = load_and_crop(PATHS["north_path"], var_name="z")
east = load_and_crop(PATHS["east_path"], var_name="z")

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

# 新增：完整保留带有陆地数据的原始插值B（用于后面保存给数据介绍用）
B_raw = bathy.interp(lon=grav.lon, lat=grav.lat, method="linear").values
B = B_raw.copy()
B[~ocean_mask] = np.nan

G = grav.values.copy()
VGG = curv.values.copy()
G_N = north.values.copy()  
G_E = east.values.copy()   

# ======================
# 2. 投影到墨卡托物理空间
# ======================
print("2. 正在投影至墨卡托坐标系...")
B_merc, x_uni, y_uni, dx_km, dy_km = project_to_mercator(lon, lat, B)
G_merc, _, _, _, _ = project_to_mercator(lon, lat, G)
VGG_merc, _, _, _, _ = project_to_mercator(lon, lat, VGG)

G_N_merc, _, _, _, _ = project_to_mercator(lon, lat, G_N) 
G_E_merc, _, _, _, _ = project_to_mercator(lon, lat, G_E) 

# ======================
# 3. 针对不同物理属性定制陆地平滑填充
# ======================
print("3. 正在执行定制化陆地平滑填充...")
B_filled = fill_land_gebco(B_merc, sigma=3)       
G_filled = fill_land_swot(G_merc, sigma=15)       
VGG_filled = fill_land_swot(VGG_merc, sigma=15)   

G_N_filled = fill_land_swot(G_N_merc, sigma=15)   
G_E_filled = fill_land_swot(G_E_merc, sigma=15)   

# ======================
# 4. 外部边缘镜像拓展
# ======================
print(f"4. 正在执行边界镜像拓展 (Pad={PAD_WIDTH} pixels)...")
B_ready = pad_array(B_filled, pad_width=PAD_WIDTH)
G_ready = pad_array(G_filled, pad_width=PAD_WIDTH)
VGG_ready = pad_array(VGG_filled, pad_width=PAD_WIDTH)

G_N_ready = pad_array(G_N_filled, pad_width=PAD_WIDTH) 
G_E_ready = pad_array(G_E_filled, pad_width=PAD_WIDTH) 

# ======================
# 5. 应用物理模型参数进行 FFT Filtering
# ======================
print(f"5. 正在执行傅里叶频域滤波 (使用物理严密推导参数 EXACT_A={EXACT_A:.1f}, EXACT_S={EXACT_S:.3f})...")
B_LP_padded = lowpass_filter(B_ready, dx_km, dy_km, EXACT_S)
G_LP_padded = lowpass_filter(G_ready, dx_km, dy_km, EXACT_S)

G_BP_padded = bandpass_filter(G_ready, dx_km, dy_km, EXACT_S, DEPTH_KM, EXACT_A)
VGG_BP_padded = bandpass_filter(VGG_ready, dx_km, dy_km, EXACT_S, DEPTH_KM, EXACT_A)

G_N_BP_padded = bandpass_filter(G_N_ready, dx_km, dy_km, EXACT_S, DEPTH_KM, EXACT_A)
G_E_BP_padded = bandpass_filter(G_E_ready, dx_km, dy_km, EXACT_S, DEPTH_KM, EXACT_A)

# ======================
# 5.5 裁切镜像拓展边缘
# ======================
print("5.5 正在裁切冗余的镜像边缘...")
B_LP_merc = unpad_array(B_LP_padded, pad_width=PAD_WIDTH)
G_LP_merc = unpad_array(G_LP_padded, pad_width=PAD_WIDTH)
G_BP_merc = unpad_array(G_BP_padded, pad_width=PAD_WIDTH)
VGG_BP_merc = unpad_array(VGG_BP_padded, pad_width=PAD_WIDTH)

G_N_BP_merc = unpad_array(G_N_BP_padded, pad_width=PAD_WIDTH) 
G_E_BP_merc = unpad_array(G_E_BP_padded, pad_width=PAD_WIDTH) 

# ======================
# 6. 反插值回等经纬度网格
# ======================
print("6. 滤波完成，正在重投影回原始经纬度网格...")
B_LP = project_to_lonlat(x_uni, y_uni, B_LP_merc, lon, lat)
G_LP = project_to_lonlat(x_uni, y_uni, G_LP_merc, lon, lat)
G_BP = project_to_lonlat(x_uni, y_uni, G_BP_merc, lon, lat)
VGG_BP = project_to_lonlat(x_uni, y_uni, VGG_BP_merc, lon, lat)

G_N_BP = project_to_lonlat(x_uni, y_uni, G_N_BP_merc, lon, lat) 
G_E_BP = project_to_lonlat(x_uni, y_uni, G_E_BP_merc, lon, lat) 

# ======================
# 7. 重新施加掩膜并保存结果
# ======================
print("7. 正在剥离陆地及边缘冗余数据并保存训练特征...")
B_LP[~ocean_mask] = np.nan
G_LP[~ocean_mask] = np.nan
G_BP[~ocean_mask] = np.nan
VGG_BP[~ocean_mask] = np.nan

G_N_BP[~ocean_mask] = np.nan 
G_E_BP[~ocean_mask] = np.nan 

B_ocean_raw = B.copy()
G_ocean_raw = G.copy()
VGG_ocean_raw = VGG.copy()

G_N_ocean_raw = G_N.copy() 
G_E_ocean_raw = G_E.copy() 

G_ocean_raw[~ocean_mask] = np.nan
VGG_ocean_raw[~ocean_mask] = np.nan
B_ocean_raw[~ocean_mask] = np.nan

G_N_ocean_raw[~ocean_mask] = np.nan 
G_E_ocean_raw[~ocean_mask] = np.nan 

# 原始海区特征保存
save_nc(B_LP, "B_LP", "m", lon, lat)
save_nc(G_LP, "G_LP", "mGal", lon, lat)
save_nc(G_BP, "G_BP", "mGal", lon, lat)
save_nc(VGG_BP, "VGG_BP", "s^-2", lon, lat)

save_nc(G_ocean_raw, "G_ocean_raw", "mGal", lon, lat)
save_nc(VGG_ocean_raw, "VGG_ocean_raw", "s^-2", lon, lat)
save_nc(B_ocean_raw, "B_ocean_raw", "m", lon, lat)

save_nc(G_N_ocean_raw, "G_North", "microradian", lon, lat)
save_nc(G_E_ocean_raw, "G_East", "microradian", lon, lat)
save_nc(G_N_BP, "G_North_BP", "microradian", lon, lat)
save_nc(G_E_BP, "G_East_BP", "microradian", lon, lat)

# ======================
# 8. 保存用于数据介绍的【带有陆地数据】的原始全区域网格
# ======================
print("8. 正在保存带有陆地数据的原始裁剪网格 (用于数据介绍)...")
save_nc(B_raw, "GEBCO_raw_with_land", "m", lon, lat)
save_nc(G, "G_raw_with_land", "mGal", lon, lat)
save_nc(VGG, "VGG_raw_with_land", "s^-2", lon, lat)
save_nc(G_N, "G_North_raw_with_land", "microradian", lon, lat)
save_nc(G_E, "G_East_raw_with_land", "microradian", lon, lat)

print("✅ 所有特征及带陆地原始数据已成功生成保存！")