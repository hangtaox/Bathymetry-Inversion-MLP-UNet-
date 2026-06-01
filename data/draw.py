import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
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
gebco_path = r"D:\project\data\GEBCO_2024\gebco_2024\GEBCO_2024.nc"

LON_RANGE = (104, 122)
LAT_RANGE = (0, 26)

# ======================
# 色带与归一化精调
# ======================
ocean_colors = ["#000018", "#0022B3", "#0066FF", "#00E5FF", "#E6FFFF"]
land_nodes = [0.0, 0.1, 0.4, 0.8, 1.0]
land_colors_list = ["#3CB371", "#FFD700", "#CD5C5C", "#8B0000", "#FFFFFF"]

ocean_cmap_base = mcolors.LinearSegmentedColormap.from_list("ocean_base", ocean_colors)
land_cmap_base = mcolors.LinearSegmentedColormap.from_list("land_base", list(zip(land_nodes, land_colors_list)))

ocean_array = ocean_cmap_base(np.linspace(0, 1, 6000))
land_array = land_cmap_base(np.linspace(0, 1, 3000))
combined_colors = np.vstack((ocean_array, land_array))

cmap_gebco_1 = mcolors.ListedColormap(combined_colors)
norm_gebco_1 = mcolors.Normalize(vmin=-6000, vmax=3000)

ocean_nodes_2 = [0.0, 0.35, 0.65, 0.9, 1.0]
cmap_gebco_2 = mcolors.LinearSegmentedColormap.from_list("ocean_enhanced", list(zip(ocean_nodes_2, ocean_colors)))
norm_gebco_2 = mcolors.Normalize(vmin=-6000, vmax=0)

bright_geology_colors = [
    "#000066", "#1A40B3", "#3B85EB", "#2ED1D1", "#9CE05A", 
    "#FFEA00", "#FF9D00", "#E64545", "#FFAEC4", "#FFFFFF"
]
gravity_cmap_bright = mcolors.LinearSegmentedColormap.from_list("custom_grav_bright", bright_geology_colors)

def plot_study_area(data_array, cmap, norm, cbar_label, out_filename, cbar_ticks=None, 
                    coast_lw=0.4, coast_color='dimgray', coast_alpha=0.6):
    fig = plt.figure(figsize=(8, 11.5))
    
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=coast_lw, edgecolor=coast_color, alpha=coast_alpha, zorder=2)
    ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=0) 
    
    lon = data_array.lon.values
    lat = data_array.lat.values
    data_values = data_array.values
    im = ax.pcolormesh(lon, lat, data_values, cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), zorder=1)
    
    # 严格限制图幅边界
    ax.set_extent([LON_RANGE[0], LON_RANGE[1], LAT_RANGE[0], LAT_RANGE[1]], crs=ccrs.PlateCarree())
    ax.margins(0) 
    
    # 限制刻度在范围内，避免将图框拉伸
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
# 原有流程：纯海洋及海陆分离出图
# ==========================================
print("正在处理原始 GEBCO 数据...")
gebco_ds = xr.open_dataset(gebco_path)
gebco_raw = gebco_ds['elevation'].sel(lon=slice(LON_RANGE[0], LON_RANGE[1]), lat=slice(LAT_RANGE[0], LAT_RANGE[1]))
plot_study_area(
    data_array=gebco_raw, cmap=cmap_gebco_1, norm=norm_gebco_1,  
    cbar_label="高程 (m)", out_filename=processed_dir / "Map_01_GEBCO_Raw.png",
    cbar_ticks=[-6000, -4000, -2000, 0, 2000] 
)

print("正在处理剔除陆地的 GEBCO 数据...")
b_ocean = xr.open_dataset(processed_dir / "B_ocean_raw.nc")['B_ocean_raw']
plot_study_area(
    data_array=b_ocean, cmap=cmap_gebco_2, norm=norm_gebco_2,  
    cbar_label="深度 (m)", out_filename=processed_dir / "Map_02_GEBCO_Ocean.png",
    cbar_ticks=[-6000, -4000, -2000, 0] 
)

print("正在处理海洋重力异常数据...")
g_ocean = xr.open_dataset(processed_dir / "G_ocean_raw.nc")['G_ocean_raw']
vmax_g = float(g_ocean.quantile(0.98)); vmin_g = float(g_ocean.quantile(0.02))
norm_g = mcolors.Normalize(vmin=-max(abs(vmin_g), abs(vmax_g)), vmax=max(abs(vmin_g), abs(vmax_g)))
plot_study_area(
    data_array=g_ocean, cmap=gravity_cmap_bright, norm=norm_g,
    cbar_label="重力异常 (mGal)", out_filename=processed_dir / "Map_03_Gravity_Ocean.png"
)

print("正在处理海洋垂直重力梯度数据...")
vgg_ocean = xr.open_dataset(processed_dir / "VGG_ocean_raw.nc")['VGG_ocean_raw']
vmax_vgg = float(vgg_ocean.quantile(0.98)); vmin_vgg = float(vgg_ocean.quantile(0.02))
norm_vgg = mcolors.Normalize(vmin=-max(abs(vmin_vgg), abs(vmax_vgg)), vmax=max(abs(vmin_vgg), abs(vmax_vgg)))
plot_study_area(
    data_array=vgg_ocean, cmap=gravity_cmap_bright, norm=norm_vgg,
    cbar_label="垂直重力梯度 (Eotvos)", out_filename=processed_dir / "Map_04_VGG_Ocean.png"
)

print("正在处理海洋 East 垂线偏差数据...")
g_east_ocean = xr.open_dataset(processed_dir / "G_East.nc")['G_East']
vmax_ge = float(g_east_ocean.quantile(0.98)); vmin_ge = float(g_east_ocean.quantile(0.02))
norm_ge = mcolors.Normalize(vmin=-max(abs(vmin_ge), abs(vmax_ge)), vmax=max(abs(vmin_ge), abs(vmax_ge)))
plot_study_area(
    data_array=g_east_ocean, cmap=gravity_cmap_bright, norm=norm_ge,
    cbar_label="东向垂线偏差 (microradian)", out_filename=processed_dir / "Map_05_G_East_Ocean.png"
)

print("正在处理海洋 North 垂线偏差数据...")
g_north_ocean = xr.open_dataset(processed_dir / "G_North.nc")['G_North']
vmax_gn = float(g_north_ocean.quantile(0.98)); vmin_gn = float(g_north_ocean.quantile(0.02))
norm_gn = mcolors.Normalize(vmin=-max(abs(vmin_gn), abs(vmax_gn)), vmax=max(abs(vmin_gn), abs(vmax_gn)))
plot_study_area(
    data_array=g_north_ocean, cmap=gravity_cmap_bright, norm=norm_gn,
    cbar_label="北向垂线偏差 (microradian)", out_filename=processed_dir / "Map_06_G_North_Ocean.png"
)

print("\n==========================================")
print("正在绘制用于数据介绍的带陆地原始数据...")
print("==========================================")

print("正在处理带陆地的 GEBCO 数据...")
gebco_land = xr.open_dataset(processed_dir / "GEBCO_raw_with_land.nc")['GEBCO_raw_with_land']
plot_study_area(
    data_array=gebco_land, cmap=cmap_gebco_1, norm=norm_gebco_1,
    cbar_label="高程 (m)", out_filename=processed_dir / "Map_Intro_01_GEBCO_Land.png",
    cbar_ticks=[-6000, -4000, -2000, 0, 2000]
)

print("正在处理带陆地的重力异常数据...")
g_land = xr.open_dataset(processed_dir / "G_raw_with_land.nc")['G_raw_with_land']
vmax_gl = float(g_land.quantile(0.98)); vmin_gl = float(g_land.quantile(0.02))
norm_gl = mcolors.Normalize(vmin=-max(abs(vmin_gl), abs(vmax_gl)), vmax=max(abs(vmin_gl), abs(vmax_gl)))
plot_study_area(
    data_array=g_land, cmap=gravity_cmap_bright, norm=norm_gl,
    cbar_label="重力异常 (mGal)", out_filename=processed_dir / "Map_Intro_02_Gravity_Land.png",
    coast_lw=1.2, coast_color='black', coast_alpha=1.0  
)

print("正在处理带陆地的垂直重力梯度数据...")
vgg_land = xr.open_dataset(processed_dir / "VGG_raw_with_land.nc")['VGG_raw_with_land']
vmax_vggl = float(vgg_land.quantile(0.98)); vmin_vggl = float(vgg_land.quantile(0.02))
norm_vggl = mcolors.Normalize(vmin=-max(abs(vmin_vggl), abs(vmax_vggl)), vmax=max(abs(vmin_vggl), abs(vmax_vggl)))
plot_study_area(
    data_array=vgg_land, cmap=gravity_cmap_bright, norm=norm_vggl,
    cbar_label="垂直重力梯度 (Eotvos)", out_filename=processed_dir / "Map_Intro_03_VGG_Land.png",
    coast_lw=1.2, coast_color='black', coast_alpha=1.0  
)

print("正在处理带陆地的 East 垂线偏差数据...")
ge_land = xr.open_dataset(processed_dir / "G_East_raw_with_land.nc")['G_East_raw_with_land']
vmax_gel = float(ge_land.quantile(0.98)); vmin_gel = float(ge_land.quantile(0.02))
norm_gel = mcolors.Normalize(vmin=-max(abs(vmin_gel), abs(vmax_gel)), vmax=max(abs(vmin_gel), abs(vmax_gel)))
plot_study_area(
    data_array=ge_land, cmap=gravity_cmap_bright, norm=norm_gel,
    cbar_label="东向垂线偏差数据 (microradian)", out_filename=processed_dir / "Map_Intro_04_G_East_Land.png",
    coast_lw=1.2, coast_color='black', coast_alpha=1.0  
)

print("正在处理带陆地的 North 垂线偏差数据...")
gn_land = xr.open_dataset(processed_dir / "G_North_raw_with_land.nc")['G_North_raw_with_land']
vmax_gnl = float(gn_land.quantile(0.98)); vmin_gnl = float(gn_land.quantile(0.02))
norm_gnl = mcolors.Normalize(vmin=-max(abs(vmin_gnl), abs(vmax_gnl)), vmax=max(abs(vmin_gnl), abs(vmax_gnl)))
plot_study_area(
    data_array=gn_land, cmap=gravity_cmap_bright, norm=norm_gnl,
    cbar_label="北向垂线偏差 (microradian)", out_filename=processed_dir / "Map_Intro_05_G_North_Land.png",
    coast_lw=1.2, coast_color='black', coast_alpha=1.0  
)

print("🎉 所有绘图任务已全部执行完毕！")