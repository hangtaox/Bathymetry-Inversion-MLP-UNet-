import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from pathlib import Path

# ======================
# 路径配置
# ======================
processed_dir = Path(r"D:\project\data\processed")

LON_RANGE = (105, 125)
LAT_RANGE = (0, 30)

# ======================
# 色带与归一化精调 (严格继承上一版的完美配置)
# ======================

# 1. B_LP (低通地形) 使用与纯海洋 GEBCO 一致的色带与范围
ocean_colors = ["#000018", "#0022B3", "#0066FF", "#00E5FF", "#E6FFFF"]
ocean_nodes_2 = [0.0, 0.35, 0.65, 0.9, 1.0]
cmap_gebco_2 = mcolors.LinearSegmentedColormap.from_list("ocean_enhanced", list(zip(ocean_nodes_2, ocean_colors)))
norm_gebco_2 = mcolors.Normalize(vmin=-6000, vmax=0)

# 2. G_LP, G_BP, VGG_BP 使用提亮版地质重力色带
bright_geology_colors = [
    "#000066", "#1A40B3", "#3B85EB", "#2ED1D1", "#9CE05A", 
    "#FFEA00", "#FF9D00", "#E64545", "#FFAEC4", "#FFFFFF"
]
gravity_cmap_bright = mcolors.LinearSegmentedColormap.from_list("custom_grav_bright", bright_geology_colors)

def plot_study_area(data_array, cmap, norm, cbar_label, out_filename, cbar_ticks=None):
    """
    完全继承上一版的完美图框和鹰眼图配置
    """
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
    
    # 色标
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
            
    # 小地球（鹰眼图） - 保持你的完美坐标
    center_lon = sum(LON_RANGE) / 2
    center_lat = sum(LAT_RANGE) / 2
    
    inset_width_ratio = 0.15
    inset_height_ratio = 0.15 * (8.0 / 11.0) 
    
    inset_left = 0.22  
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
# 逐个读取滤波数据并出图
# ==========================================

# 1. B_LP (低通海底地形)
print("正在绘制 B_LP (低通海底地形)...")
b_lp = xr.open_dataset(processed_dir / "B_LP.nc")['B_LP']
plot_study_area(
    data_array=b_lp,
    cmap=cmap_gebco_2,
    norm=norm_gebco_2,  
    cbar_label="Low-pass Bathymetry (m)",
    out_filename=processed_dir / "Map_05_B_LP.png",
    cbar_ticks=[-6000, -4000, -2000, 0] # 同样保持极简刻度
)

# 2. G_LP (低通重力异常)
print("正在绘制 G_LP (低通重力异常)...")
g_lp = xr.open_dataset(processed_dir / "G_LP.nc")['G_LP']
vmax_glp = float(g_lp.quantile(0.98))
vmin_glp = float(g_lp.quantile(0.02))
max_abs_glp = max(abs(vmin_glp), abs(vmax_glp))
norm_glp = mcolors.Normalize(vmin=-max_abs_glp, vmax=max_abs_glp)

plot_study_area(
    data_array=g_lp,
    cmap=gravity_cmap_bright,  
    norm=norm_glp,
    cbar_label="Low-pass Gravity Anomaly (mGal)",
    out_filename=processed_dir / "Map_06_G_LP.png"
)

# 3. G_BP (带通重力异常)
print("正在绘制 G_BP (带通重力异常)...")
g_bp = xr.open_dataset(processed_dir / "G_BP.nc")['G_BP']
vmax_gbp = float(g_bp.quantile(0.98))
vmin_gbp = float(g_bp.quantile(0.02))
max_abs_gbp = max(abs(vmin_gbp), abs(vmax_gbp))
norm_gbp = mcolors.Normalize(vmin=-max_abs_gbp, vmax=max_abs_gbp)

plot_study_area(
    data_array=g_bp,
    cmap=gravity_cmap_bright,  
    norm=norm_gbp,
    cbar_label="Band-pass Gravity Anomaly (mGal)",
    out_filename=processed_dir / "Map_07_G_BP.png"
)

# 4. VGG_BP (带通垂直重力梯度)
print("正在绘制 VGG_BP (带通垂直重力梯度)...")
vgg_bp = xr.open_dataset(processed_dir / "VGG_BP.nc")['VGG_BP']
vmax_vggbp = float(vgg_bp.quantile(0.98))
vmin_vggbp = float(vgg_bp.quantile(0.02))
max_abs_vggbp = max(abs(vmin_vggbp), abs(vmax_vggbp))
norm_vggbp = mcolors.Normalize(vmin=-max_abs_vggbp, vmax=max_abs_vggbp)

plot_study_area(
    data_array=vgg_bp,
    cmap=gravity_cmap_bright,  
    norm=norm_vggbp,
    # 梯度图的单位通常用 Eotvos 展现更地道，由于前面保存了 s^-2，这里图例统一规范为 Eotvos
    cbar_label="Band-pass Vertical Gravity Gradient (Eotvos)",
    out_filename=processed_dir / "Map_08_VGG_BP.png"
)

print("🎉 4 张滤波处理分量图已全部绘制完毕！")