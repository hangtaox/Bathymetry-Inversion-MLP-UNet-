import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from pathlib import Path
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# ======================
# 路径与范围配置
# ======================
OUT_DIR = Path("D:/project/data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 根据船测数据真实覆盖面，固定最优绘图范围
LON_RANGE = (104, 122)
LAT_RANGE = (0, 26)

def read_ship_txt(path):
    data = np.loadtxt(path, delimiter=",")  # 如果是空格，删掉 delimiter=","
    lon = data[:, 0]
    lat = data[:, 1]
    depth = data[:, 2]
    return lon, lat, depth

# ======================
# 绘图样式配置
# ======================
ocean_colors = ["#000018", "#0022B3", "#0066FF", "#00E5FF", "#E6FFFF"]
ocean_nodes_2 = [0.0, 0.35, 0.65, 0.9, 1.0]
cmap_ocean = mcolors.LinearSegmentedColormap.from_list("ocean_enhanced", list(zip(ocean_nodes_2, ocean_colors)))
norm_ocean = mcolors.Normalize(vmin=-6000, vmax=0)

def plot_ship_tracks(lon, lat, depth_values, cmap, norm, cbar_label, out_filename):
    """专门用于绘制船测轨迹(散点)的函数"""
    # 按照 18经度宽 x 26纬度高 的比例，稍微调整一下画布高度
    fig = plt.figure(figsize=(8, 11.5))
    
    # 主图
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], projection=ccrs.PlateCarree())
    
    # 底图特征 (海岸线已虚化：调细、变灰、加透明度，避免和陆地冲突)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='dimgray', alpha=0.6, zorder=2)
    ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=0) 
    
    # 将深度值统一转化为负值，以匹配 vmin=-6000 到 vmax=0 的色标
    plot_depth = -np.abs(depth_values)
    
    # 核心数据映射 (使用 scatter 绘制离散点)
    im = ax.scatter(lon, lat, c=plot_depth, s=0.8, cmap=cmap, norm=norm, 
                    transform=ccrs.PlateCarree(), zorder=1, edgecolors='none')
    
    # 坐标轴与刻度
    ax.set_extent([LON_RANGE[0], LON_RANGE[1], LAT_RANGE[0], LAT_RANGE[1]], crs=ccrs.PlateCarree())
    ax.margins(0) 
    
    # 设置刻度为每 5 度一标，覆盖新范围
    ax.set_xticks(np.arange(105, 125, 5), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(0, 30, 5), crs=ccrs.PlateCarree())
    
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    
    ax.tick_params(axis='x', labelsize=12, direction='in', pad=8)
    ax.tick_params(axis='y', labelsize=12, direction='in', pad=5)
    
    ax.gridlines(draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--', zorder=3)
    
    # 色标
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.06, aspect=40, shrink=0.8)
    cbar.set_ticks([-6000, -4000, -2000, 0])
    cbar.set_label(cbar_label, fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    
    # 提取并上移 0° 标签
    plt.draw() 
    for label in ax.yaxis.get_ticklabels():
        if '0' in label.get_text():
            label.set_verticalalignment('bottom')
            
    # 鹰眼图
    center_lon = sum(LON_RANGE) / 2
    center_lat = sum(LAT_RANGE) / 2
    inset_width_ratio = 0.15
    inset_height_ratio = 0.15 * (8.0 / 11.5) 
    inset_left = 0.21  
    inset_bottom = 0.9 - inset_height_ratio
    
    inset_ax = fig.add_axes([inset_left, inset_bottom, inset_width_ratio, inset_height_ratio], 
                            projection=ccrs.Orthographic(center_lon, center_lat))
    inset_ax.set_global()
    inset_ax.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='black', linewidth=0.4)
    inset_ax.add_feature(cfeature.OCEAN, facecolor='white')
    inset_ax.add_feature(cfeature.COASTLINE, linewidth=0.3)
    inset_ax.gridlines(linewidth=0.2, color='gray', alpha=0.5)
    
    lons = [LON_RANGE[0], LON_RANGE[1], LON_RANGE[1], LON_RANGE[0], LON_RANGE[0]]
    lats = [LAT_RANGE[0], LAT_RANGE[0], LAT_RANGE[1], LAT_RANGE[1], LAT_RANGE[0]]
    inset_ax.plot(lons, lats, color='red', linewidth=1.5, transform=ccrs.PlateCarree(), zorder=5)
    
    plt.savefig(out_filename, dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close()
    print(f"✅ 成功保存绘图: {out_filename.name}")

# ======================
# 1. 数据读取与修正
# ======================
print("正在读取数据...")
# 修复了文件名读取的反转错误！确保 mb 读取 mbes，sb 读取 sbes
lon_mb, lat_mb, depth_mb = read_ship_txt("D:/project/data/NCEI/tichu_nanhaimbes.txt")
lon_sb, lat_sb, depth_sb = read_ship_txt("D:/project/data/NCEI/tichu_nanhaisbes.txt")

# 应用计算出的经验公式，修正单波束系统误差
print("正在修正单波束系统误差 (-23m offset)...")
depth_sb_corrected = 0.99584 * depth_sb - 22.958

# ======================
# 2. 分别保存单波束和多波束数据
# ======================
ds_mb = xr.Dataset(
    data_vars=dict(depth=("point", depth_mb)),
    coords=dict(lon=("point", lon_mb), lat=("point", lat_mb)),
    attrs=dict(description="Multi-beam bathymetry points", depth_unit="meters", depth_positive="down")
)
mb_out_path = OUT_DIR / "ship_mbes.nc"
ds_mb.to_netcdf(mb_out_path)
print(f"多波束数据已保存至: {mb_out_path}")

ds_sb = xr.Dataset(
    data_vars=dict(depth=("point", depth_sb_corrected)),
    coords=dict(lon=("point", lon_sb), lat=("point", lat_sb)),
    attrs=dict(description="Single-beam bathymetry points (Calibrated)", depth_unit="meters", depth_positive="down")
)
sb_out_path = OUT_DIR / "ship_sbes_calibrated.nc"
ds_sb.to_netcdf(sb_out_path)
print(f"单波束数据已保存至: {sb_out_path}")

# ======================
# 3. 数据合并与保存
# ======================
lon_combined = np.concatenate([lon_mb, lon_sb])
lat_combined = np.concatenate([lat_mb, lat_sb])
depth_combined = np.concatenate([depth_mb, depth_sb_corrected])  

# 构建身份标签：1 代表多波束(MB)，0 代表单波束(SB)
source = np.concatenate([
    np.ones(len(lon_mb), dtype=np.int8),   
    np.zeros(len(lon_sb), dtype=np.int8),  
])

print(f"\n合并完成: 多波束 {len(lon_mb)} 点, 单波束 {len(lon_sb)} 点.")

ds_combined = xr.Dataset(
    data_vars=dict(depth=("point", depth_combined), source=("point", source)),
    coords=dict(lon=("point", lon_combined), lat=("point", lat_combined)),
    attrs=dict(description="Shipborne bathymetry points (Combined & SB calibrated)", depth_unit="meters", depth_positive="down")
)

combined_out_path = OUT_DIR / "ship_combined_calibrated.nc"
ds_combined.to_netcdf(combined_out_path)
print(f"合并数据已保存至: {combined_out_path}")

# ======================
# 4. 执行绘图任务
# ======================
print("\n==========================================")
print("开始绘制船测数据轨迹图...")
print("==========================================")

plot_ship_tracks(
    lon=lon_mb, lat=lat_mb, depth_values=depth_mb,
    cmap=cmap_ocean, norm=norm_ocean, cbar_label="多波束测量深度(m)",
    out_filename=OUT_DIR / "Map_Ship_01_MBES.png"
)

plot_ship_tracks(
    lon=lon_sb, lat=lat_sb, depth_values=depth_sb_corrected,
    cmap=cmap_ocean, norm=norm_ocean, cbar_label="单波束测量深度 (m)",
    out_filename=OUT_DIR / "Map_Ship_02_SBES.png"
)

plot_ship_tracks(
    lon=lon_combined, lat=lat_combined, depth_values=depth_combined,
    cmap=cmap_ocean, norm=norm_ocean, cbar_label="船载测量深度(m)",
    out_filename=OUT_DIR / "Map_Ship_03_Combined.png"
)

print("🎉 所有数据提取与船测分布绘图已全部完成！")