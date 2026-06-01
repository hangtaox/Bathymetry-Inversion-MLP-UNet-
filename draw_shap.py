import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.cm as cm

# ==========================================
# 1. 全局学术字体设置
# ==========================================
# 推荐使用 Arial 或者 Times New Roman，这是 SCI 期刊最常用的字体
plt.rcParams['font.family'] = 'Arial'  
plt.rcParams['font.size'] = 12
plt.rcParams['mathtext.fontset'] = 'custom'

# ==========================================
# 2. 数据准备
# ==========================================
data = {
    'Feature': ['b_lp', 'vgg_bp', 'g_bp', 'g_east_bp', 'curv', 'g_north_bp', 
                'g_lp', 'grav', 'g_east', 'g_north', 'lat', 'lon'],
    'Importance (%)': [44.0, 9.2, 7.5, 5.4, 5.4, 5.1, 
                       4.6, 4.6, 3.9, 3.5, 3.5, 3.3]
}

df = pd.DataFrame(data)
df = df.sort_values(by='Importance (%)', ascending=True).reset_index(drop=True)

# ==========================================
# 3. 高颜值亮色色带策略
# ==========================================
# 使用 Spectral_r 色带，从 0.15 取到 0.95，避免两端的极寒/极暖色
# 呈现出 蓝 -> 绿 -> 黄 -> 红 的明亮渐变，非常吸睛且不发暗
color_map = cm.Spectral_r
colors = color_map(np.linspace(0.15, 0.95, len(df)))

# ==========================================
# 4. 开始绘图 (调整为 10x8 解决“太扁”的问题)
# ==========================================
fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

# height=0.65 让柱子稍微变细，增加条带间的空白，显得更高级
bars = ax.barh(df['Feature'], df['Importance (%)'], 
               height=0.65, color=colors, edgecolor='black', linewidth=0.8)

# ==========================================
# 5. 坐标轴与边框优化
# ==========================================
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_linewidth(1.2)
ax.spines['left'].set_linewidth(1.2)

# 网格线设为更轻柔的颜色
ax.xaxis.grid(True, linestyle='--', alpha=0.3, color='gray')
ax.set_axisbelow(True)

# 标题和坐标轴标签 (字号加大，仅这三个地方加粗)
ax.set_title('Global Feature Importance Ranking', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Relative Contribution to Model Output (%)', 
              fontsize=14, fontweight='bold')
ax.set_ylabel('Input Features', fontsize=14, fontweight='bold')

# 坐标轴刻度字体保持 regular 正常粗细，不抢戏
ax.tick_params(axis='both', which='major', labelsize=12)

# 预留 X 轴空间
ax.set_xlim(0, 50)

# ==========================================
# 6. 数值标签 (去掉粗体，显得更加干净整洁)
# ==========================================
for bar in bars:
    width = bar.get_width()
    ax.text(width + 0.6,                           
            bar.get_y() + bar.get_height() / 2,    
            f"{width:.1f}%",                       
            va='center', ha='left', 
            fontsize=11, color='black') # 去掉了原来的 fontweight='bold'

plt.tight_layout()

# ==========================================
# 7. 保存图片
# ==========================================
save_path = "shap_importance_bright_and_crisp.png"
plt.savefig(save_path, bbox_inches='tight', transparent=False)
print(f"✅ 全新高颜值图表已生成并保存至：{save_path}")

plt.show()