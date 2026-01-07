# quick_check.py
import torch
from data.dataset import BathymetryPointDataset
import numpy as np

PATHS = {
    "grav_path": r"D:\project\filtered_gravity_bandpass.nc",
    "gebco_path": r"D:\project\data\GEBCO_2024\gebco_2024\GEBCO_2024.nc",
    "curv_path": r"D:\project\filtered_curvature_bandpass.nc"
}

LON_RANGE = (112, 114)
LAT_RANGE = (15, 18)

# 加载数据集获取归一化参数
dataset = BathymetryPointDataset(
    grav_path=PATHS["grav_path"],
    curv_path=PATHS["curv_path"],
    gebco_path=PATHS["gebco_path"],
    lon_range=LON_RANGE,
    lat_range=LAT_RANGE,
    normalize=True
)

y_std = dataset.y_std
y_mean = dataset.y_mean

print("=== 归一化参数 ===")
print(f"水深均值 (y_mean): {y_mean:.1f} m")
print(f"水深标准差 (y_std): {y_std:.1f} m")

# 计算真实误差
final_loss = 0.027  # 你的最终损失
rmse_real = np.sqrt(final_loss) * y_std
print(f"\n最终损失分析:")
print(f"归一化MSE: {final_loss:.4f}")
print(f"真实RMSE: {rmse_real:.1f} m")
print(f"相对误差: {rmse_real / abs(y_mean) * 100:.1f}%")

# 检查数据范围
y_real = dataset.y_raw
print(f"\n真实水深范围:")
print(f"最小值: {y_real.min():.1f} m")
print(f"最大值: {y_real.max():.1f} m")
print(f"平均值: {y_real.mean():.1f} m")
print(f"标准差: {y_real.std():.1f} m")