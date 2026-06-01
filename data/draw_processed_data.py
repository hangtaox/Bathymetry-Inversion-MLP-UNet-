import xarray as xr
import numpy as np
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
# 路径配置
# ======================
processed_dir = Path(r"D:\project\data\processed")

LON_RANGE = (104, 122)
LAT_RANGE = (0, 26)

# ======================
# 色带与归一化精调 
# ======================
ocean_colors = ["#000018", "#0022B3", "#0066FF", "#00E5FF", "#E6FFFF"]
ocean_nodes_2 = [0.0, 0.35, 0.65, 0.9, 1.0]
cmap_gebco_2 = mcolors.LinearSegmentedColormap.from_list("ocean_enhanced", list(zip(ocean_nodes_2, ocean_colors)))
norm_gebco_2 = mcolors.Normalize(vmin=-6000, vmax=0)

bright_geology_colors = [
    "#000066", "#1A40B3", "#3B85EB", "#2ED1D1", "#9CE05A", 
    "#FFEA00", "#FF9D00", "#E64545", "#FFAEC4", "#FFFFFF"
]
gravity_cmap_bright = mcolors.LinearSegmentedColormap.from_list("custom_grav_bright", bright_geology_colors)

def plot_study_area(data_array, cmap, norm, cbar_label, out_filename, cbar_ticks=None):
    fig = plt.figure(figsize=(8, 11.5))
    
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor='dimgray', alpha=0.6, zorder=2)
    ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=0) 
    
    lon = data_array.lon.values
    lat = data_array.lat.values
    data_values = data_array.values
    im = ax.pcolormesh(lon, lat, data_values, cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), zorder=1)
    
    ax.set_extent([LON_RANGE[0], LON_RANGE[1], LAT_RANGE[0], LAT_RANGE[1]], crs=ccrs.PlateCarree())
    ax.margins(0) 
    
    # 限制刻度范围
    ax.set_xticks(np.arange(105, 125, 5), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(0, 30, 5), crs=ccrs.PlateCarree())
    
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    
    ax.tick_params(axis='x', labelsize=12, direction='in', pad=8)
    ax.tick_params(axis='y', labelsize=12, direction='in', pad=5)
    ax.gridlines(draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--', zorder=3)
    
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.06, aspect=40, shrink=0.8)
    if cbar_ticks is not None:
        cbar.set_ticks(cbar_ticks)
    cbar.set_label(cbar_label, fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    
    plt.draw() 
    for label in ax.yaxis.get_ticklabels():
        if '0' in label.get_text():
            label.set_verticalalignment('bottom')
            
    center_lon = sum(LON_RANGE) / 2
    center_lat = sum(LAT_RANGE) / 2
    
    inset_width_ratio = 0.15
    inset_height_ratio = 0.15 * (8.0 / 11.5) 
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

print("正在绘制 B_LP (低通海底地形)...")
b_lp = xr.open_dataset(processed_dir / "B_LP.nc")['B_LP']
plot_study_area(
    data_array=b_lp, cmap=cmap_gebco_2, norm=norm_gebco_2,  
    cbar_label="深度 (m)", out_filename=processed_dir / "Map_07_B_LP.png",
    cbar_ticks=[-6000, -4000, -2000, 0]
)

print("正在绘制 G_LP (低通重力异常)...")
g_lp = xr.open_dataset(processed_dir / "G_LP.nc")['G_LP']
vmax_glp = float(g_lp.quantile(0.98))
vmin_glp = float(g_lp.quantile(0.02))
norm_glp = mcolors.Normalize(vmin=-max(abs(vmin_glp), abs(vmax_glp)), vmax=max(abs(vmin_glp), abs(vmax_glp)))
plot_study_area(
    data_array=g_lp, cmap=gravity_cmap_bright, norm=norm_glp,
    cbar_label="重力异常 (mGal)", out_filename=processed_dir / "Map_08_G_LP.png"
)

print("正在绘制 G_BP (带通重力异常)...")
g_bp = xr.open_dataset(processed_dir / "G_BP.nc")['G_BP']
vmax_gbp = float(g_bp.quantile(0.98))
vmin_gbp = float(g_bp.quantile(0.02))
norm_gbp = mcolors.Normalize(vmin=-max(abs(vmin_gbp), abs(vmax_gbp)), vmax=max(abs(vmin_gbp), abs(vmax_gbp)))
plot_study_area(
    data_array=g_bp, cmap=gravity_cmap_bright, norm=norm_gbp,
    cbar_label="重力异常 (mGal)", out_filename=processed_dir / "Map_09_G_BP.png"
)

print("正在绘制 VGG_BP (带通垂直重力梯度)...")
vgg_bp = xr.open_dataset(processed_dir / "VGG_BP.nc")['VGG_BP']
vmax_vggbp = float(vgg_bp.quantile(0.98))
vmin_vggbp = float(vgg_bp.quantile(0.02))
norm_vggbp = mcolors.Normalize(vmin=-max(abs(vmin_vggbp), abs(vmax_vggbp)), vmax=max(abs(vmin_vggbp), abs(vmax_vggbp)))
plot_study_area(
    data_array=vgg_bp, cmap=gravity_cmap_bright, norm=norm_vggbp,
    cbar_label="垂直重力梯度 (Eotvos)", out_filename=processed_dir / "Map_10_VGG_BP.png"
)

print("正在绘制 G_East_BP (带通 East 垂线偏差)...")
g_east_bp = xr.open_dataset(processed_dir / "G_East_BP.nc")['G_East_BP']
vmax_gebp = float(g_east_bp.quantile(0.98))
vmin_gebp = float(g_east_bp.quantile(0.02))
norm_gebp = mcolors.Normalize(vmin=-max(abs(vmin_gebp), abs(vmax_gebp)), vmax=max(abs(vmin_gebp), abs(vmax_gebp)))
plot_study_area(
    data_array=g_east_bp, cmap=gravity_cmap_bright, norm=norm_gebp,
    cbar_label="东向垂线偏差 (microradian)", out_filename=processed_dir / "Map_11_G_East_BP.png"
)

print("正在绘制 G_North_BP (带通 North 垂线偏差)...")
g_north_bp = xr.open_dataset(processed_dir / "G_North_BP.nc")['G_North_BP']
vmax_gnbp = float(g_north_bp.quantile(0.98))
vmin_gnbp = float(g_north_bp.quantile(0.02))
norm_gnbp = mcolors.Normalize(vmin=-max(abs(vmin_gnbp), abs(vmax_gnbp)), vmax=max(abs(vmin_gnbp), abs(vmax_gnbp)))
plot_study_area(
    data_array=g_north_bp, cmap=gravity_cmap_bright, norm=norm_gnbp,
    cbar_label="北向垂线偏差 (microradian)", out_filename=processed_dir / "Map_12_G_North_BP.png"
)

print("🎉 6 张滤波处理分量图已全部绘制完毕！")