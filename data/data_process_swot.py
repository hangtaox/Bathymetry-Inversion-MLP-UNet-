import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def compute_power_spectrum(data, dlon, dlat):
    """
    计算功率谱
    Args:
        data: 输入数据
        dlon: 经度间隔（度）
        dlat: 纬度间隔（度）
    Returns:
        wavelengths_km: 波长数组（km）
        psd_avg: 平均功率谱密度
        F_data: 傅里叶变换结果
        K_radial: 径向波数
    """
    # 傅里叶变换
    F_data = np.fft.fft2(data)
    
    # 计算波数
    kx = np.fft.fftfreq(data.shape[1], d=dlon)
    ky = np.fft.fftfreq(data.shape[0], d=dlat)
    Kx, Ky = np.meshgrid(kx, ky)
    K_radial = np.sqrt(Kx**2 + Ky**2)
    
    # 计算功率谱
    PSD = np.abs(F_data)**2
    
    # 径向平均
    k_bins = np.linspace(0, K_radial.max(), 50)
    k_digitized = np.digitize(K_radial.flatten(), k_bins)
    psd_avg = np.array([PSD.flatten()[k_digitized == i].mean() 
                       for i in range(1, len(k_bins)) if np.sum(k_digitized == i) > 5])
    k_centers = (k_bins[:-1] + k_bins[1:]) / 2
    wavelengths_km = 1 / (k_centers * 111)  # 转换为km
    
    return wavelengths_km[:len(psd_avg)], psd_avg, F_data, K_radial

def create_bandpass_filter(K_radial, lambda_low_km, lambda_high_km, z0=0):
    """
    创建带通滤波器（通过高斯高通和低通组合）
    Args:
        K_radial: 径向波数
        lambda_low_km: 低截止波长（km），用于高通（去除噪声）
        lambda_high_km: 高截止波长（km），用于低通（去除长波）
        z0: 平均水深（km），用于向下延拓
    Returns:
        带通滤波器核
    """
    # 转换为度
    lambda_low_deg = lambda_low_km / 111
    lambda_high_deg = lambda_high_km / 111
    
    # 高通滤波器（去除短波噪声）
    s_highpass = 1 / (2 * np.pi * lambda_low_deg)
    W_highpass = 1 - np.exp(-0.5 * (K_radial * s_highpass)**2)
    
    # 低通滤波器（去除长波）
    s_lowpass = 1 / (2 * np.pi * lambda_high_deg)
    W_lowpass = np.exp(-0.5 * (K_radial * s_lowpass)**2)
    
    # 带通滤波器 = 高通 * 低通
    W_bandpass = W_highpass * W_lowpass
    
    # 处理直流分量
    W_bandpass[K_radial == 0] = 0
    
    return W_bandpass

def apply_bandpass_filter(data, dlon, dlat, lambda_low_km, lambda_high_km, z0=0):
    """
    应用带通滤波器并向下延拓
    Args:
        data: 输入数据
        dlon: 经度间隔（度）
        dlat: 纬度间隔（度）
        lambda_low_km: 低截止波长（km）
        lambda_high_km: 高截止波长（km）
        z0: 平均水深（km）
    Returns:
        filtered: 滤波后数据
        W_filter: 滤波器核
        F_filtered: 滤波后的傅里叶变换
    """
    # 傅里叶变换
    F_data = np.fft.fft2(data)
    
    # 计算波数
    kx = np.fft.fftfreq(data.shape[1], d=dlon)
    ky = np.fft.fftfreq(data.shape[0], d=dlat)
    Kx, Ky = np.meshgrid(kx, ky)
    K_radial = np.sqrt(Kx**2 + Ky**2)
    
    # 创建滤波器
    W_filter = create_bandpass_filter(K_radial, lambda_low_km, lambda_high_km, z0)
    
    # 应用滤波器并向下延拓
    F_filtered = F_data * W_filter 
    filtered = np.real(np.fft.ifft2(F_filtered))
    
    return filtered, W_filter, F_filtered

def save_to_netcdf(data, lons, lats, filename, variable_name='filtered_data', 
                   description='Bandpass filtered data'):
    """
    保存数据为NetCDF格式
    Args:
        data: 要保存的数据数组
        lons: 经度数组
        lats: 纬度数组
        filename: 输出文件名
        variable_name: 变量名
        description: 变量描述
    """
    # 创建xarray数据集
    ds = xr.Dataset(
        {
            variable_name: xr.DataArray(
                data=data,
                dims=['lat', 'lon'],
                coords={
                    'lat': lats,
                    'lon': lons
                },
                attrs={
                    'long_name': description,
                    'units': 'mGal'
                }
            )
        },
        attrs={
            'title': 'SWOT Bandpass Filtered Data',
            'source': 'SWOT satellite gravity data',
            'processing': f'Bandpass filter with downward continuation',
            'creation_date': np.datetime64('now').astype(str)
        }
    )
    
    # 保存为NetCDF
    ds.to_netcdf(filename)
    print(f"数据已保存到: {filename}")
    return ds

class SWOTProcessor:
    """SWOT数据处理类"""
    
    def __init__(self, grav_path: str, gebco_path: str, curv_path: str):
        self.grav_path = grav_path
        self.gebco_path = gebco_path
        self.curv_path = curv_path
    
    def load_data(self, lon_range: tuple, lat_range: tuple):
        """加载数据"""
        # 加载重力异常数据
        ds_grav = xr.open_dataset(self.grav_path)
        grav = ds_grav['z'].sel(lon=slice(*lon_range), lat=slice(*lat_range))
        ds_grav.close()
        
        # 加载水深数据
        ds_gebco = xr.open_dataset(self.gebco_path)
        gebco = ds_gebco['elevation'].sel(lon=slice(*lon_range), lat=slice(*lat_range))
        ds_gebco.close()
        
        # 加载重力梯度数据
        ds_curv = xr.open_dataset(self.curv_path)
        curv = ds_curv['z'].sel(lon=slice(*lon_range), lat=slice(*lat_range))
        ds_curv.close()
        
        return grav, gebco, curv
    
    def compute_z0(self, gebco):
        """计算平均水深"""
        underwater = gebco.where(gebco < 0)
        return -underwater.mean().values / 1000  # 转换为km

# 主程序
if __name__ == "__main__":
    # 参数
    PATHS = {
        "grav_path": r"D:\project\data\SWOT\grav_SWOT_02.nc",
        "gebco_path": r"D:\project\data\GEBCO_2024\gebco_2024\GEBCO_2024.nc",
        "curv_path": r'D:\project\data\SWOT\curv_SWOT_02.nc'
    }
    LON_RANGE = (114, 116)
    LAT_RANGE = (15, 18)
    
    # 带通滤波参数
    HIGH_CUTOFF = 65  # 高截止波长（km），去除长波
    LOW_CUTOFF = 2    # 低截止波长（km），去除短波噪声
    
    # 处理数据
    processor = SWOTProcessor(**PATHS)
    grav, gebco, curv = processor.load_data(LON_RANGE, LAT_RANGE)
    z0 = processor.compute_z0(gebco)  # 单位为km
    
    # 准备数据
    values_grav = grav.values
    values_curv = curv.values
    lons, lats = grav.lon.values, grav.lat.values
    dlon, dlat = np.mean(np.diff(lons)), np.mean(np.diff(lats))
    
    print(f"数据处理开始...")
    print(f"数据范围: 经度{LON_RANGE}°, 纬度{LAT_RANGE}°")
    print(f"数据尺寸: {values_grav.shape}")
    print(f"平均水深: {z0*1000:.0f}m")
    print(f"带通滤波范围: {LOW_CUTOFF}-{HIGH_CUTOFF}km")
    
    # 处理重力异常数据
    print("\n处理重力异常数据...")
    # 计算原始数据功率谱
    wavelengths_km, psd_avg, F_data, K_radial = compute_power_spectrum(values_grav, dlon, dlat)
    
    # 应用带通滤波器
    filtered_grav, W_filter, F_filtered_grav = apply_bandpass_filter(
        values_grav, dlon, dlat, LOW_CUTOFF, HIGH_CUTOFF, z0
    )
    
    # 计算滤波后数据的功率谱
    wavelengths_filt, psd_filt, F_filt, K_radial_filt = compute_power_spectrum(filtered_grav, dlon, dlat)
    
    # 处理重力梯度数据
    print("处理重力梯度数据...")
    # 计算原始梯度数据功率谱
    wavelengths_curv, psd_curv, F_curv, K_radial_curv = compute_power_spectrum(values_curv, dlon, dlat)
    
    # 应用相同的带通滤波器
    filtered_curv, W_filter_curv, F_filtered_curv = apply_bandpass_filter(
        values_curv, dlon, dlat, LOW_CUTOFF, HIGH_CUTOFF, z0
    )
    
    # 计算滤波后梯度数据的功率谱
    wavelengths_filt_curv, psd_filt_curv, F_filt_curv, K_radial_filt_curv = compute_power_spectrum(filtered_curv, dlon, dlat)
    
    # 保存滤波后的数据
    print("\n保存滤波后数据...")
    # 保存重力异常数据
    save_to_netcdf(
        filtered_grav, lons, lats, 
        'filtered_gravity_bandpass_2.nc',
        variable_name='filtered_gravity',
        description=f'Bandpass filtered gravity anomalies ({LOW_CUTOFF}-{HIGH_CUTOFF}km) with downward continuation'
    )
    
    # 保存重力梯度数据
    save_to_netcdf(
        filtered_curv, lons, lats,
        'filtered_curvature_bandpass_2.nc',
        variable_name='filtered_curvature',
        description=f'Bandpass filtered gravity curvature ({LOW_CUTOFF}-{HIGH_CUTOFF}km) with downward continuation'
    )
     
    # 可视化 - 5个图：原图、频谱、功率谱、滤波器、滤波结果
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # 1. 原始重力异常
    im1 = axes[0].imshow(grav, cmap='RdBu_r', extent=[LON_RANGE[0], LON_RANGE[1], 
                                                      LAT_RANGE[0], LAT_RANGE[1]], 
                         origin='lower')
    axes[0].set_title('原始重力异常')
    axes[0].set_xlabel('经度')
    axes[0].set_ylabel('纬度')
    plt.colorbar(im1, ax=axes[0], label='重力异常 (mGal)')
    
    # 2. 频谱（傅里叶变换结果）
    spectrum_magnitude = np.log10(np.abs(F_data) + 1)
    im2 = axes[1].imshow(spectrum_magnitude, cmap='hot', origin='lower')
    axes[1].set_title('原始傅里叶频谱（幅度）')
    axes[1].set_xlabel('波数kx')
    axes[1].set_ylabel('波数ky')
    plt.colorbar(im2, ax=axes[1], label='对数幅度')
    
    # 3. 功率谱对比（原始和滤波后）
    axes[2].loglog(wavelengths_km, psd_avg, 'b-', linewidth=2, label='原始重力异常')
    axes[2].loglog(wavelengths_filt, psd_filt, 'r-', linewidth=2, label=f'滤波后重力异常')
    axes[2].loglog(wavelengths_curv, psd_curv, 'g--', linewidth=1.5, alpha=0.7, label='原始重力梯度')
    axes[2].loglog(wavelengths_filt_curv, psd_filt_curv, 'm--', linewidth=1.5, alpha=0.7, label='滤波后重力梯度')
    axes[2].axvline(HIGH_CUTOFF, color='k', linestyle='--', alpha=0.7, label=f'{HIGH_CUTOFF}km')
    axes[2].axvline(LOW_CUTOFF, color='k', linestyle=':', alpha=0.7, label=f'{LOW_CUTOFF}km')
    axes[2].set_xlabel('波长 (km)')
    axes[2].set_ylabel('功率谱密度')
    axes[2].set_title(f'功率谱对比\n带通滤波范围: {LOW_CUTOFF}-{HIGH_CUTOFF}km')
    axes[2].grid(True, which='both', linestyle='--', alpha=0.5)
    axes[2].legend(fontsize=8, loc='best')
    
    # 4. 带通滤波器核
    im4 = axes[3].imshow(W_filter, cmap='viridis', origin='lower')
    axes[3].set_title(f'带通滤波器\n({LOW_CUTOFF}-{HIGH_CUTOFF}km)')
    axes[3].set_xlabel('波数kx')
    axes[3].set_ylabel('波数ky')
    plt.colorbar(im4, ax=axes[3], label='滤波器响应')
    
    # 5. 滤波后的重力异常
    vmax = np.nanmax(np.abs(filtered_grav))
    im5 = axes[4].imshow(filtered_grav, cmap='RdBu_r', 
                         extent=[LON_RANGE[0], LON_RANGE[1], LAT_RANGE[0], LAT_RANGE[1]], 
                         origin='lower', vmin=-vmax, vmax=vmax)
    axes[4].set_title(f'带通滤波后重力异常\n({LOW_CUTOFF}-{HIGH_CUTOFF}km)')
    axes[4].set_xlabel('经度')
    axes[4].set_ylabel('纬度')
    plt.colorbar(im5, ax=axes[4], label='重力异常 (mGal)')
    
    # 6. 滤波后的重力梯度
    vmax_curv = np.nanmax(np.abs(filtered_curv))
    im6 = axes[5].imshow(filtered_curv, cmap='RdBu_r', 
                         extent=[LON_RANGE[0], LON_RANGE[1], LAT_RANGE[0], LAT_RANGE[1]], 
                         origin='lower', vmin=-vmax_curv, vmax=vmax_curv)
    axes[5].set_title(f'带通滤波后重力梯度\n({LOW_CUTOFF}-{HIGH_CUTOFF}km)')
    axes[5].set_xlabel('经度')
    axes[5].set_ylabel('纬度')
    plt.colorbar(im6, ax=axes[5], label='重力梯度')
    
    plt.tight_layout()
    plt.savefig('swot_bandpass_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n处理完成!")
    print(f"保存的文件:")
    print(f"  1. filtered_gravity_bandpass.nc - 滤波后重力异常数据")
    print(f"  2. filtered_curvature_bandpass.nc - 滤波后重力梯度数据")
    print(f"  3. original_gravity.nc - 原始重力异常数据")
    print(f"  4. original_curvature.nc - 原始重力梯度数据")
    print(f"  5. swot_bandpass_results.png - 结果可视化图")