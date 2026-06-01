import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# ======================
# 用户定义的滤波目标参数
# ======================
LONG_CUTOFF_KM = 160.0         # 长波段截止 (剥离深部区域场)
SHORT_CUTOFF_KM = 5          # 极限短波截止波长 (在此处衰减 50%)

# ======================
# 数据路径与范围配置
# ======================
GEBCO_PATH = r"D:\project\data\GEBCO_2024\gebco_2024\GEBCO_2024.nc"
LON_RANGE = (104, 122)
LAT_RANGE = (0, 26)

# ======================
# 1. 动态计算真实平均水深
# ======================
print("正在读取 GEBCO 数据并计算研究区真实平均水深...")
ds = xr.open_dataset(GEBCO_PATH)
bathy = ds["elevation"].sel(
    lon=slice(LON_RANGE[0], LON_RANGE[1]),
    lat=slice(LAT_RANGE[0], LAT_RANGE[1])
)

ocean_depths = bathy.values[bathy.values < 0]
mean_depth_m = np.nanmean(ocean_depths)
DEPTH_KM = abs(mean_depth_m) / 1000.0

print(f"-> 计算得到的真实平均水深: {mean_depth_m:.2f} m")
print(f"-> 向下延拓参数 (DEPTH_KM): {DEPTH_KM:.4f} km\n")

# ======================
# 2. 直接正向推导精确的 A
# ======================
# 长波平滑系数 s (W1)
exact_s = LONG_CUTOFF_KM * (np.sqrt(np.log(2)/2) / np.pi)

# --- 核心：直接计算在 4.0km 处使得 W2 = 0.5 的 A 值 ---
# 公式推导: A * (1/lambda)^4 * exp(4*pi*d/lambda) = 1
exact_A = (SHORT_CUTOFF_KM**4) * np.exp(-4 * np.pi * DEPTH_KM / SHORT_CUTOFF_KM)

print(f"--- 自动推导的精确滤波器参数 ---")
print(f"长波截止 {LONG_CUTOFF_KM} km 对应的 s = {exact_s:.3f}")
print(f"为确保 {SHORT_CUTOFF_KM}km 处精确衰减至 50%, 算得 A = {exact_A:.4f}\n")

# ======================
# 3. 构建频率域坐标与滤波器
# ======================
wavelengths = np.logspace(np.log10(1000), np.log10(2), 500)
k = 1.0 / wavelengths

W1_lowpass = np.exp(-2 * (np.pi * k * exact_s)**2)
HP = 1.0 - W1_lowpass
W2 = 1.0 / (1.0 + exact_A * (k**4) * np.exp(4 * np.pi * k * DEPTH_KM))
DC = np.exp(2 * np.pi * k * DEPTH_KM)

# 带通公式：先用 HP 提取中高频，再用 W2 和 DC 剔除高频噪声并还原海底
Combined_Bandpass = HP * W2 * DC

# ======================
# 4. 分别绘制并保存两张独立的图片
# ======================
print("正在分别绘制并保存两张滤波器响应曲线图...")

# ----------------------------------------------------
# 第一张图：分滤波器响应曲线 (Individual Filter Components)
# ----------------------------------------------------
fig1, ax1 = plt.subplots(figsize=(10, 6.5), constrained_layout=True)

ax1.semilogx(wavelengths, W1_lowpass, 'b-', lw=2, label=f'高斯低通滤波响应曲线({LONG_CUTOFF_KM}km)')
ax1.semilogx(wavelengths, HP, 'r-', lw=2, label=f'高斯高通滤波响应曲线')
ax1.semilogx(wavelengths, W2, 'g-', lw=2, label=f'维纳低通滤波响应曲线 ({SHORT_CUTOFF_KM}km)')

ax1.axvline(LONG_CUTOFF_KM, color='blue', linestyle='--', alpha=0.5)
ax1.axvline(SHORT_CUTOFF_KM, color='green', linestyle='-', alpha=0.8)
ax1.axhline(0.5, color='gray', linestyle=':')

ax1.set_xlim(1000, 2)
ax1.set_ylim(-0.05, 1.05)
ax1.set_title(f'滤波器响应函数曲线', fontsize=14, fontweight='bold')
ax1.set_xlabel('波长(km)', fontsize=12)
ax1.set_ylabel('函数响应值', fontsize=12)
ax1.grid(True, which='both', ls='--', alpha=0.5)

# 【修复】：将图例放在图表正下方（X轴下方），彻底避开标题和曲线
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=10)

plt.savefig("Filter_Response_Individual.png", dpi=300, bbox_inches='tight')
print("-> 已保存第一张图: Filter_Response_Individual.png")


# ----------------------------------------------------
# 第二张图：组合滤波器响应曲线 (Combined Filter Response)
# ----------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(10, 6.5), constrained_layout=True)

ax2.loglog(wavelengths, DC, 'k--', lw=1.5, label='Downward Continuation (DC)')
ax2.loglog(wavelengths, W2, 'g--', lw=1.5, label='W2 Low-pass')
ax2.loglog(wavelengths, Combined_Bandpass, 'm-', lw=3, label='Combined Bandpass Filter (HP * W2 * DC)')

ax2.axvline(LONG_CUTOFF_KM, color='blue', linestyle='--', alpha=0.5)
ax2.axvline(SHORT_CUTOFF_KM, color='green', linestyle='-', alpha=0.8, label=f'Cutoff ({SHORT_CUTOFF_KM}km)')

ax2.set_xlim(1000, 2)
ax2.set_ylim(1e-3, 1e2) 
ax2.set_title(f'Combined Filter Response (Target Cutoff = {SHORT_CUTOFF_KM}km)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Wavelength (km)', fontsize=12)
ax2.set_ylabel('Gain (Log Scale)', fontsize=12)
ax2.grid(True, which='both', ls='--', alpha=0.5)

# 【修复】：同样将图例放在图表正下方
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=10)

plt.savefig("Filter_Response_Combined.png", dpi=300, bbox_inches='tight')
print("-> 已保存第二张图: Filter_Response_Combined.png")

plt.show()