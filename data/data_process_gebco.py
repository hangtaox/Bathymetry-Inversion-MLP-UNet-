import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def create_gaussian_lowpass_filter(K_radial, cutoff_wavelength_km):
    """
    创建高斯低通滤波器
    Args:
        K_radial: 径向波数
        cutoff_wavelength_km: 截止波长（km）
    Returns:
        低通滤波器核
    """
    # 转换为度（1度≈111km）
    cutoff_deg = cutoff_wavelength_km / 111
    
    # 计算高斯滤波器参数
    sigma = 1 / (2 * np.pi * cutoff_deg)
    
    # 创建高斯低通滤波器
    filter_kernel = np.exp(-0.5 * (K_radial * sigma)**2)
    
    # 确保直流分量（频率为0）完全通过
    filter_kernel[K_radial == 0] = 1
    
    return filter_kernel

def split_bathymetry_by_wavelength(bathymetry_data, dlon, dlat, cutoff_wavelength_km=65):
    """
    将水深数据按波长分割为长波和短波部分
    Args:
        bathymetry_data: 水深数据
        dlon: 经度间隔（度）
        dlat: 纬度间隔（度）
        cutoff_wavelength_km: 分割波长（km）
    Returns:
        long_wave: 长波部分（>cutoff_wavelength_km）
        short_wave: 短波部分（<cutoff_wavelength_km）
    """
    # 傅里叶变换
    F_data = np.fft.fft2(bathymetry_data)
    
    # 计算波数
    kx = np.fft.fftfreq(bathymetry_data.shape[1], d=dlon)
    ky = np.fft.fftfreq(bathymetry_data.shape[0], d=dlat)
    Kx, Ky = np.meshgrid(kx, ky)
    K_radial = np.sqrt(Kx**2 + Ky**2)
    
    # 创建高斯低通滤波器
    lowpass_filter = create_gaussian_lowpass_filter(K_radial, cutoff_wavelength_km)
    
    # 获取长波部分（低频）
    F_long = F_data * lowpass_filter
    long_wave = np.real(np.fft.ifft2(F_long))
    
    # 获取短波部分（原始减去长波）
    short_wave = bathymetry_data - long_wave
    
    return long_wave, short_wave, lowpass_filter

def calculate_statistics(original, long_wave, short_wave):
    """
    计算统计数据
    """
    # 重建数据
    reconstructed = long_wave + short_wave
    
    # 计算误差
    reconstruction_error = original - reconstructed
    abs_error = np.abs(reconstruction_error)
    
    stats = {
        'original_mean': np.nanmean(original),
        'original_std': np.nanstd(original),
        'long_wave_mean': np.nanmean(long_wave),
        'long_wave_std': np.nanstd(long_wave),
        'short_wave_mean': np.nanmean(short_wave),
        'short_wave_std': np.nanstd(short_wave),
        'max_abs_error': np.nanmax(abs_error),
        'mean_abs_error': np.nanmean(abs_error),
        'rms_error': np.sqrt(np.nanmean(reconstruction_error**2))
    }
    
    return stats

def save_wave_components(long_wave, short_wave, lons, lats, prefix='bathymetry'):
    """
    保存长波和短波分量
    """
    # 保存长波分量
    ds_long = xr.Dataset(
        {
            'long_wave': xr.DataArray(
                data=long_wave,
                dims=['lat', 'lon'],
                coords={'lat': lats, 'lon': lons},
                attrs={'long_name': f'Long wavelength bathymetry (>65km)', 
                       'units': 'meters',
                       'cutoff_wavelength': '65km'}
            )
        },
        attrs={'title': 'Long wavelength bathymetry component'}
    )
    ds_long.to_netcdf(f'./tmp_img/{prefix}_long_wave_65km.nc')
    
    # 保存短波分量
    ds_short = xr.Dataset(
        {
            'short_wave': xr.DataArray(
                data=short_wave,
                dims=['lat', 'lon'],
                coords={'lat': lats, 'lon': lons},
                attrs={'long_name': f'Short wavelength bathymetry (<65km)', 
                       'units': 'meters',
                       'cutoff_wavelength': '65km'}
            )
        },
        attrs={'title': 'Short wavelength bathymetry component'}
    )
    ds_short.to_netcdf(f'./tmp_img/{prefix}_short_wave_65km.nc')
    
    print(f"✓ 长波分量已保存: {prefix}_long_wave_65km.nc")
    print(f"✓ 短波分量已保存: {prefix}_short_wave_65km.nc")
    
    return ds_long, ds_short

# 主程序
if __name__ == "__main__":
    # 参数设置
    GEBCO_PATH = r"D:\project\data\GEBCO_2024\gebco_2024\GEBCO_2024.nc"
    LON_RANGE = (112, 114)
    LAT_RANGE = (15, 18)
    CUTOFF_WAVELENGTH_KM = 65  # 分割波长
    
    print("=" * 60)
    print("水深数据频谱分割处理 - 简洁版")
    print("=" * 60)
    print(f"数据路径: {GEBCO_PATH}")
    print(f"处理区域: 经度{LON_RANGE}°, 纬度{LAT_RANGE}°")
    print(f"分割波长: {CUTOFF_WAVELENGTH_KM}km")
    print("=" * 60)
    
    # 加载水深数据
    print("加载水深数据...")
    ds_gebco = xr.open_dataset(GEBCO_PATH)
    bathymetry = ds_gebco['elevation'].sel(
        lon=slice(*LON_RANGE), 
        lat=slice(*LAT_RANGE)
    )
    ds_gebco.close()
    
    # 获取数据
    values = bathymetry.values
    lons, lats = bathymetry.lon.values, bathymetry.lat.values
    dlon, dlat = np.mean(np.diff(lons)), np.mean(np.diff(lats))
    
    print(f"数据尺寸: {values.shape}")
    print(f"网格分辨率: {dlon:.3f}° × {dlat:.3f}°")
    print(f"约合: {dlon*111:.1f}km × {dlat*111:.1f}km")
    
    # 分割水深数据
    print(f"\n按{CUTOFF_WAVELENGTH_KM}km波长分割水深数据...")
    long_wave, short_wave, filter_kernel = split_bathymetry_by_wavelength(
        values, dlon, dlat, CUTOFF_WAVELENGTH_KM
    )
    
    # 计算统计信息
    print("计算统计信息...")
    stats = calculate_statistics(values, long_wave, short_wave)
    
    print(f"\n=== 统计结果 ===")
    print(f"原始水深均值: {stats['original_mean']:.2f}m")
    print(f"长波分量均值: {stats['long_wave_mean']:.2f}m")
    print(f"短波分量均值: {stats['short_wave_mean']:.2f}m")
    print(f"短波幅度范围: [{short_wave.min():.2f}, {short_wave.max():.2f}]m")
    print(f"\n重建误差统计:")
    print(f"最大绝对误差: {stats['max_abs_error']:.6f}m")
    print(f"平均绝对误差: {stats['mean_abs_error']:.6f}m")
    print(f"RMS误差: {stats['rms_error']:.6f}m")
    
    # 保存分量数据
    print(f"\n保存分量数据...")
    ds_long, ds_short = save_wave_components(long_wave, short_wave, lons, lats)
    
    # 可视化
    print(f"\n生成可视化结果...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. 原始水深
    ax = axes[0, 0]
    im = ax.imshow(values, cmap='terrain', 
                   extent=[LON_RANGE[0], LON_RANGE[1], LAT_RANGE[0], LAT_RANGE[1]], 
                   origin='lower')
    ax.set_title('原始水深', fontsize=12)
    ax.set_xlabel('经度 (°)')
    ax.set_ylabel('纬度 (°)')
    plt.colorbar(im, ax=ax, label='高程 (m)')
    
    # 2. 长波分量 (>65km)
    ax = axes[0, 1]
    im = ax.imshow(long_wave, cmap='terrain', 
                   extent=[LON_RANGE[0], LON_RANGE[1], LAT_RANGE[0], LAT_RANGE[1]], 
                   origin='lower')
    ax.set_title(f'长波分量 (> {CUTOFF_WAVELENGTH_KM}km)', fontsize=12)
    ax.set_xlabel('经度 (°)')
    ax.set_ylabel('纬度 (°)')
    plt.colorbar(im, ax=ax, label='高程 (m)')
    
    # 3. 短波分量 (<65km)
    ax = axes[0, 2]
    vmax = max(abs(short_wave.min()), abs(short_wave.max()))
    im = ax.imshow(short_wave, cmap='RdBu_r', 
                   extent=[LON_RANGE[0], LON_RANGE[1], LAT_RANGE[0], LAT_RANGE[1]], 
                   origin='lower', vmin=-vmax, vmax=vmax)
    ax.set_title(f'短波分量 (< {CUTOFF_WAVELENGTH_KM}km)', fontsize=12)
    ax.set_xlabel('经度 (°)')
    ax.set_ylabel('纬度 (°)')
    plt.colorbar(im, ax=ax, label='高程 (m)')
    
    # 4. 重建水深 (长波+短波)
    ax = axes[1, 0]
    reconstructed = long_wave + short_wave
    im = ax.imshow(reconstructed, cmap='terrain', 
                   extent=[LON_RANGE[0], LON_RANGE[1], LAT_RANGE[0], LAT_RANGE[1]], 
                   origin='lower')
    ax.set_title('重建水深 (长波+短波)', fontsize=12)
    ax.set_xlabel('经度 (°)')
    ax.set_ylabel('纬度 (°)')
    plt.colorbar(im, ax=ax, label='高程 (m)')
    
    # 5. 重建误差
    ax = axes[1, 1]
    error = values - reconstructed
    vmax = max(abs(error.min()), abs(error.max()))
    im = ax.imshow(error, cmap='RdBu_r', 
                   extent=[LON_RANGE[0], LON_RANGE[1], LAT_RANGE[0], LAT_RANGE[1]], 
                   origin='lower', vmin=-vmax, vmax=vmax)
    ax.set_title(f'重建误差\n最大误差: {stats["max_abs_error"]:.2e}m', fontsize=12)
    ax.set_xlabel('经度 (°)')
    ax.set_ylabel('纬度 (°)')
    plt.colorbar(im, ax=ax, label='误差 (m)')
    
    # 6. 滤波器
    ax = axes[1, 2]
    im = ax.imshow(filter_kernel, cmap='viridis', origin='lower')
    ax.set_title(f'高斯低通滤波器\n截止波长: {CUTOFF_WAVELENGTH_KM}km', fontsize=12)
    ax.set_xlabel('波数 kx')
    ax.set_ylabel('波数 ky')
    plt.colorbar(im, ax=ax, label='滤波器响应')
    
    plt.suptitle(f'水深数据频谱分割 - 截止波长: {CUTOFF_WAVELENGTH_KM}km', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # 保存结果
    output_filename = f'./tmp_img/bathymetry_split_{CUTOFF_WAVELENGTH_KM}km.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"✓ 可视化结果已保存: {output_filename}")
    
    # 显示短波分量的详细统计
    print(f"\n=== 短波分量详细信息 ===")
    print(f"最小值: {short_wave.min():.4f}m")
    print(f"最大值: {short_wave.max():.4f}m")
    print(f"标准差: {stats['short_wave_std']:.4f}m")
    print(f"绝对平均值: {np.nanmean(np.abs(short_wave)):.4f}m")
    
    plt.show()
    
    print(f"\n" + "=" * 60)
    print("处理完成!")
    print("=" * 60)