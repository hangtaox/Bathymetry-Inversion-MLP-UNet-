import xarray as xr
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.pyplot as plt
# ======================
# 1. 路径与特征配置
# ======================
processed_dir = Path(r"D:\project\data\processed")

# 根据你的需求，配置要计算相关系数的 11 个特征数据文件
feature_files = {
    # --- 1. 剔除陆地的原始数据 ---
    "GEBCO水深": "B_ocean_raw.nc",
    "重力异常": "G_ocean_raw.nc",
    "垂直重力梯度": "VGG_ocean_raw.nc",
    "东向垂线偏差": "G_East.nc",
    "北向垂线偏差": "G_North.nc",
    
    # --- 2. 低通分量 (长波) ---
    "低通水深": "B_LP.nc",
    "低通重力异常": "G_LP.nc",
    
    # --- 3. 带通分量 (短波) ---
    "带通重力异常": "G_BP.nc",
    "带通垂直重力梯度": "VGG_BP.nc",
    "带通东向垂线偏差": "G_East_BP.nc",
    "带通北向垂线偏差": "G_North_BP.nc"
}

print("正在读取 11 个特征数据...")
data_dict = {}

# ======================
# 2. 读取数据并展平提取有效海洋像素
# ======================
for var_name, file_name in feature_files.items():
    file_path = processed_dir / file_name
    if not file_path.exists():
        print(f"⚠️ 找不到文件: {file_path}")
        continue
    
    # 打开 nc 文件，动态获取内部的变量名
    ds = xr.open_dataset(file_path)
    da_name = list(ds.data_vars)[0]  
    
    # 将二维网格数据压平为一维数组
    val_1d = ds[da_name].values.flatten()
    data_dict[var_name] = val_1d

# 构建 Pandas DataFrame
df = pd.DataFrame(data_dict)

# 剔除包含 NaN 的行（这样就彻底剔除了带有陆地掩膜的坐标点）
print("正在剔除陆地 NaN 掩膜区域，提取纯海洋点位...")
df_clean = df.dropna()
print(f"✅ 成功提取有效海洋数据点: {len(df_clean)} 个像素")

# ======================
# 3. 计算皮尔逊相关系数并绘制漂亮的热力图
# ======================
print("\n正在计算 Pearson 皮尔逊相关系数矩阵...")
corr_matrix = df_clean.corr(method='pearson')

# 设置绘图风格与自适应大画板 (因为特征多，画板得够大)
plt.figure(figsize=(14, 11))
sns.set_theme(style="white")
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False
# 绘制热力图
heatmap = sns.heatmap(
    corr_matrix, 
    annot=True,            # 在格子里显示具体数值
    fmt=".2f",             # 保留两位小数 (如 0.85)
    cmap="coolwarm",       # 红蓝撞色色带 (越红越正相关，越蓝越负相关)
    vmin=-1, vmax=1,       # 皮尔逊系数的理论极值
    square=True,           # 确保每个格子都是正方形，更好看
    linewidths=1.0,        # 格子之间的白色分割线加粗，层次更分明
    annot_kws={"size": 10},# 调整格子内数字大小防止重叠
    cbar_kws={"shrink": 0.8, "label": "皮尔逊相关系数"} 
)

# 标题与刻度标签设置
plt.title("皮尔逊相关系数热力图", fontsize=18, fontweight='bold', pad=20)

# X轴标签倾斜 45 度防止文字重叠，Y轴保持水平
plt.xticks(fontsize=12, rotation=45, ha='right')
plt.yticks(fontsize=12, rotation=0)

# 保存高质量图片
out_heatmap = processed_dir / "Map_13_Correlation_Heatmap_Full.png"
plt.savefig(out_heatmap, dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.close()

print(f"🎉 11个特征的热力图绘制完毕！")
print(f"✅ 图片已完美保存至: {out_heatmap}")