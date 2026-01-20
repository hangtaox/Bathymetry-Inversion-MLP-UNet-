import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path

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

def load_and_crop(path, var_name):
    ds = xr.open_dataset(path)
    da = ds[var_name]

    da = da.sel(
        lon=slice(LON_RANGE[0], LON_RANGE[1]),
        lat=slice(LAT_RANGE[0], LAT_RANGE[1])
    )

    return da

def grid_spacing_km(lon, lat):
    dlon = np.mean(np.diff(lon))
    dlat = np.mean(np.diff(lat))

    lat0 = np.mean(lat)
    dx_km = 111.32 * np.cos(np.deg2rad(lat0)) * dlon
    dy_km = 111.32 * dlat

    return dx_km, dy_km

def plot_compare(raw, filtered, title, unit):
    vmin = np.nanpercentile(raw, 2)
    vmax = np.nanpercentile(raw, 98)
    
    # 计算残差
    residual = raw - filtered

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)  # 修改为3个子图

    im0 = axes[0].imshow(raw, origin="lower", vmin=vmin, vmax=vmax)
    axes[0].set_title("Original")
    plt.colorbar(im0, ax=axes[0], label=unit)

    im1 = axes[1].imshow(filtered, origin="lower", vmin=vmin, vmax=vmax)
    axes[1].set_title("Filtered")
    plt.colorbar(im1, ax=axes[1], label=unit)

    # 添加残差图
    im2 = axes[2].imshow(residual, origin="lower", cmap='RdBu_r')
    axes[2].set_title("Residual (Original - Filtered)")
    plt.colorbar(im2, ax=axes[2], label=unit)

    fig.suptitle(title)
    plt.show()

def save_nc(data, name, unit,lon, lat):
    da = xr.DataArray(
        data,
        coords={"lat": lat, "lon": lon},
        dims=("lat", "lon"),
        name=name
    )
    da.attrs["units"] = unit
    da.to_netcdf(OUT_DIR / f"{name}.nc")
# Load data
# GEBCO
bathy = load_and_crop(PATHS["gebco_path"], var_name="elevation")

# Gravity
grav = load_and_crop(PATHS["grav_path"], var_name="z")

# VGG / curvature
curv = load_and_crop(PATHS["curv_path"], var_name="z")
# gebco分辨率不一样
bathy_lon = bathy.lon.values
bathy_lat = bathy.lat.values

bathy_dx_km, bathy_dy_km = grid_spacing_km(bathy_lon, bathy_lat)
# swot数据
lon = grav.lon.values
lat = grav.lat.values

dx_km, dy_km = grid_spacing_km(lon, lat)

B = bathy.values
G = grav.values
VGG = curv.values

bathy_mask = (bathy.values < 0).astype(np.float32) 

mask_da = xr.DataArray(
    bathy_mask,
    coords={"lat": bathy.lat.values, "lon": bathy.lon.values},
    dims=("lat", "lon")
)

mask_on_swot = mask_da.interp(
    lat=lat,
    lon=lon,
    method="nearest"
).values.astype(np.float32)
# ======================
# Filtering
# ======================
B_LP = lowpass_filter(B, bathy_dx_km, bathy_dy_km, CUTOFF_WAVELENGTH_KM)

G_LP = lowpass_filter(G, dx_km, dy_km, CUTOFF_WAVELENGTH_KM)

G_BP = bandpass_filter(G, dx_km, dy_km,
                       CUTOFF_WAVELENGTH_KM, DEPTH_KM, A_W2)

VGG_BP = bandpass_filter(VGG, dx_km, dy_km,
                         CUTOFF_WAVELENGTH_KM, DEPTH_KM, A_W2)


save_nc(B_LP, "B_LP", "m",bathy_lon, bathy_lat)
save_nc(G_LP, "G_LP", "mGal",lon, lat)
save_nc(G_BP, "G_BP", "mGal",lon, lat)
save_nc(VGG_BP, "VGG_BP", "s^-2",lon, lat)
save_nc(mask_on_swot, "OceanMask", "1=ocean, 0=land", lon, lat)

# plot_compare(B, B_LP, "Low-pass Bathymetry (B_LP)", "m")
# plot_compare(G, G_LP, "Band-pass Gravity (G_LP)", "mGal")
# plot_compare(G, G_BP, "Band-pass Gravity (G_BP)", "mGal")
# plot_compare(VGG, VGG_BP, "Band-pass VGG (VGG_BP)", "s⁻²")
# plt.figure(figsize=(5, 6))
# plt.imshow(mask_on_swot, origin="lower", cmap="gray")
# plt.title("Ocean Mask on SWOT Grid")
# plt.colorbar(label="1=ocean, 0=land")
# plt.show()
