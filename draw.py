import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from pathlib import Path

# ======================
# 路径配置
# ======================
processed_dir = Path(r"D:\project\data\processed")
gebco_path = r"D:\project\data\GEBCO_2024\gebco_2024\GEBCO_2024.nc"

LON_RANGE = (105, 125)
LAT_RANGE = (0, 30)

# ======================
# 色带与归一化精调
# ======================

# 1. GEBCO 第1张：【终极修复】构建 9000 级高精度离散色带，彻底锁死海陆分界线
ocean_colors = ["#000018", "#0022B3", "#0066FF", "#00E5FF", "#E6FFFF"]
land_nodes = [0.0, 0.1, 0.4, 0.8, 1.0]
land_colors_list = ["#3CB371", "#FFD700", "#CD5C5C", "#8B0000", "#FFFFFF"]

# 先生成两个基础连续色带
ocean_cmap_base = mcolors.LinearSegmentedColormap.from_list("ocean_base", ocean_colors)
land_cmap_base = mcolors.LinearSegmentedColormap.from_list("land_base", list(zip(land_nodes, land_colors_list)))

# 强制提取 6000 个海洋颜色和 3000 个陆地颜色拼成绝对矩阵
ocean_array = ocean_cmap_base(np.linspace(0, 1, 6000))
land_array = land_cmap_base(np.linspace(0, 1, 3000))
combined_colors = np.vstack((ocean_array, land_array))

# 使用 ListedColormap，这样 matplotlib 就不会再进行 256 级的有损压缩插值了
cmap_gebco_1 = mcolors.ListedColormap(combined_colors)
norm_gebco_1 = mcolors.Normalize(vmin=-6000, vmax=3000)


# 2. GEBCO 第2张：纯海洋
ocean_nodes_2 = [0.0, 0.35, 0.65, 0.9, 1.0]
cmap_gebco_2 = mcolors.LinearSegmentedColormap.from_list("ocean_enhanced", list(zip(ocean_nodes_2, ocean_colors)))
norm_gebco_2 = mcolors.Normalize(vmin=-6000, vmax=0)

# 3. 重力色带
bright_geology_colors = [
    "#000066", "#1A40B3", "#3B85EB", "#2ED1D1", "#9CE05A", 
    "#FFEA00", "#FF9D00", "#E64545", "#FFAEC4", "#FFFFFF"
]
gravity_cmap_bright = mcolors.LinearSegmentedColormap.from_list("custom_grav_bright", bright_geology_colors)


def plot_study_area(data_array, cmap, norm, cbar_label, out_filename, cbar_ticks=None):
    fig = plt.figure(figsize=(8, 11))
    
    # 主图
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], projection=ccrs.PlateCarree())
    
    # 底图特征
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor='dimgray', alpha=0.6, zorder=2)
    ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=0) 
    
    # 核心数据映射
    lon = data_array.lon.values
    lat = data_array.lat.values
    data_values = data_array.values
    im = ax.pcolormesh(lon, lat, data_values, cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), zorder=1)
    
    # 坐标轴与刻度
    ax.set_extent([LON_RANGE[0], LON_RANGE[1], LAT_RANGE[0], LAT_RANGE[1]], crs=ccrs.PlateCarree())
    ax.margins(0) 
    
    ax.set_xticks(np.arange(105, 126, 5), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(0, 31, 5), crs=ccrs.PlateCarree())
    
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    
    ax.tick_params(axis='x', labelsize=12, direction='in', pad=8)
    ax.tick_params(axis='y', labelsize=12, direction='in', pad=5)
    
    ax.gridlines(draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--', zorder=3)
    
    # ==========================================
    # 色标
    # ==========================================
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.06, aspect=40, shrink=0.8)
    if cbar_ticks is not None:
        cbar.set_ticks(cbar_ticks)
    cbar.set_label(cbar_label, fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    
    # 提取并上移 0° 标签
    plt.draw() 
    for label in ax.yaxis.get_ticklabels():
        if '0' in label.get_text():
            label.set_verticalalignment('bottom')
            
    # ==========================================
    # 小地球（鹰眼图）
    # ==========================================
    center_lon = sum(LON_RANGE) / 2
    center_lat = sum(LAT_RANGE) / 2
    
    inset_width_ratio = 0.15
    inset_height_ratio = 0.15 * (8.0 / 11.0) 
    
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
    print(f"✅ 成功保存: {out_filename.name}")


# ==========================================
# 逐个出图执行
# ==========================================

print("正在处理原始 GEBCO 数据...")
gebco_ds = xr.open_dataset(gebco_path)
gebco_raw = gebco_ds['elevation'].sel(lon=slice(LON_RANGE[0], LON_RANGE[1]), lat=slice(LAT_RANGE[0], LAT_RANGE[1]))
plot_study_area(
    data_array=gebco_raw,
    cmap=cmap_gebco_1,  
    norm=norm_gebco_1,  
    cbar_label="Elevation / Depth (m)",
    out_filename=processed_dir / "Map_01_GEBCO_Raw.png",
    cbar_ticks=[-6000, -4000, -2000, 0, 2000] # 保留你想要的极简刻度
)

print("正在处理剔除陆地的 GEBCO 数据...")
b_ocean = xr.open_dataset(processed_dir / "B_ocean_raw.nc")['B_ocean_raw']
plot_study_area(
    data_array=b_ocean,
    cmap=cmap_gebco_2,
    norm=norm_gebco_2,  
    cbar_label="Depth (m)",
    out_filename=processed_dir / "Map_02_GEBCO_Ocean.png",
    cbar_ticks=[-6000, -4000, -2000, 0] 
)

print("正在处理海洋重力异常数据...")
g_ocean = xr.open_dataset(processed_dir / "G_ocean_raw.nc")['G_ocean_raw']
vmax_g = float(g_ocean.quantile(0.98))
vmin_g = float(g_ocean.quantile(0.02))
max_abs_g = max(abs(vmin_g), abs(vmax_g))
norm_g = mcolors.Normalize(vmin=-max_abs_g, vmax=max_abs_g)

plot_study_area(
    data_array=g_ocean,
    cmap=gravity_cmap_bright,  
    norm=norm_g,
    cbar_label="Gravity Anomaly (mGal)",
    out_filename=processed_dir / "Map_03_Gravity_Ocean.png"
)

print("正在处理海洋垂直重力梯度数据...")
vgg_ocean = xr.open_dataset(processed_dir / "VGG_ocean_raw.nc")['VGG_ocean_raw']
vmax_vgg = float(vgg_ocean.quantile(0.98))
vmin_vgg = float(vgg_ocean.quantile(0.02))
max_abs_vgg = max(abs(vmin_vgg), abs(vmax_vgg))
norm_vgg = mcolors.Normalize(vmin=-max_abs_vgg, vmax=max_abs_vgg)

plot_study_area(
    data_array=vgg_ocean,
    cmap=gravity_cmap_bright,  
    norm=norm_vgg,
    cbar_label="Vertical Gravity Gradient (Eotvos)",
    out_filename=processed_dir / "Map_04_VGG_Ocean.png"
)

print("🎉 大功告成！这回海是海，陆是陆，泾渭分明！")