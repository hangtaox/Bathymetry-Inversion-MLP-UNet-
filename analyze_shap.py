import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import shap
from torch.utils.data import DataLoader, random_split

# 导入你自己的模型和数据集类
from models.dual_stream import DualStreamFusionNet 
from data.dataset_cnn import BathymetryShipPointCNNDataset

# ==========================================
# 1. 配置参数
# ==========================================
PATCH_SIZE = 13
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_DIR = f"./checkpoints/dual_stream/{PATCH_SIZE}"
RESULT_DIR = f"./result/dual_stream/{PATCH_SIZE}"
os.makedirs(RESULT_DIR, exist_ok=True)

PATHS = {
    "grav": "./data/processed/G_ocean_raw.nc",
    "curv": "./data/processed/VGG_ocean_raw.nc",
    "b_lp": "./data/processed/B_LP.nc",
    "g_lp": "./data/processed/G_LP.nc",
    "g_bp": "./data/processed/G_BP.nc",
    "vgg_bp": "./data/processed/VGG_BP.nc",
    "g_east": "./data/processed/G_East.nc",
    "g_north": "./data/processed/G_North.nc",
    "g_east_bp": "./data/processed/G_East_BP.nc",
    "g_north_bp": "./data/processed/G_North_BP.nc",
}

SHIP_NC = "./data/ship_combined_calibrated.nc"
ALIGNED_SHIP_NC = "./data/ship_bathymetry_points_aligned.nc"
LON_RANGE = (105, 125)
LAT_RANGE = (0, 30)

feature_names = list(PATHS.keys()) + ["lon", "lat"]

# ==========================================
# 核心包装器：处理 1D 输出和 inplace 报错
# ==========================================
class SHAPModelWrapper(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        
    def forward(self, x):
        out = self.base_model(x)
        if out.dim() == 1:
            out = out.unsqueeze(1)
        return out

def remove_inplace(model):
    """递归关闭所有的 inplace 操作，防止 SHAP 梯度钩子报错"""
    for child in model.modules():
        if hasattr(child, 'inplace'):
            child.inplace = False

def run_shap_analysis():
    print(f"[INFO] Using device: {DEVICE}")

    # ==========================================
    # 2. 加载数据
    # ==========================================
    dataset = BathymetryShipPointCNNDataset(
        feature_paths=PATHS,
        ship_nc_path=SHIP_NC,
        aligned_ship_nc_path=ALIGNED_SHIP_NC,
        lon_range=LON_RANGE,
        lat_range=LAT_RANGE,
        patch_size=PATCH_SIZE,       
        normalize=True,
    )

    N = len(dataset)
    n_train = int(0.8 * N)
    n_val = int(0.1 * N)
    n_test = N - n_train - n_val

    train_set, val_set, test_set = random_split(
        dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )

    # ★ 升级：提取 100 个样本做基准，2000 个测试样本进行充分的可解释性计算
    bg_loader = DataLoader(train_set, batch_size=100, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=2000, shuffle=True)

    background_X, _ = next(iter(bg_loader))
    test_X, _ = next(iter(test_loader))
    
    background_X = background_X.to(DEVICE)
    test_X = test_X.to(DEVICE)

    # ==========================================
    # 3. 加载模型
    # ==========================================
    base_model = DualStreamFusionNet(in_channels=dataset.X.shape[1], patch_size=PATCH_SIZE).to(DEVICE)
    model_path = f"{CKPT_DIR}/best_model.pt"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到模型权重文件: {model_path}")
        
    base_model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    remove_inplace(base_model)
    base_model.eval()

    wrapped_model = SHAPModelWrapper(base_model).to(DEVICE)

    # ==========================================
    # 4. 初始化 Explainer 并计算
    # ==========================================
    print("\n[INFO] 初始化 SHAP Explainer...")
    try:
        explainer = shap.DeepExplainer(wrapped_model, background_X)
        print(f"[INFO] 正在计算 {len(test_X)} 个样本的 SHAP 值 (DeepExplainer，这可能需要几分钟)...")
        shap_values = explainer.shap_values(test_X)
    except Exception as e:
        print(f"\n[⚠️ 警告] DeepExplainer 遇到复杂算子限制，自动切换为 GradientExplainer...")
        explainer = shap.GradientExplainer(wrapped_model, background_X)
        print(f"[INFO] 正在计算 {len(test_X)} 个样本的 SHAP 值 (GradientExplainer，这可能需要几分钟)...")
        shap_values = explainer.shap_values(test_X)

    # ==========================================
    # 5. 智能聚合、降维与百分比转换
    # ==========================================
    print("\n[INFO] 正在聚合空间维度特征重要性并转换为百分比...")
    
    if hasattr(shap_values, 'values'):
        shap_values = shap_values.values

    shap_arr = np.array(shap_values)

    # 扒掉多余的外壳，直到遇到 batch_size
    while shap_arr.ndim > 4 and shap_arr.shape[0] == 1:
        shap_arr = shap_arr[0]

    expected_channels = len(feature_names)
    channel_axis = -1
    
    for i, dim_size in enumerate(shap_arr.shape):
        if dim_size == expected_channels:
            channel_axis = i
            break
            
    if channel_axis == -1:
        raise ValueError(f"严重错误: 无法在输出 (shape={shap_arr.shape}) 中匹配到 {expected_channels} 个特征！")

    # 对 N, H, W 等维度求平均绝对值
    axes_to_mean = tuple(i for i in range(shap_arr.ndim) if i != channel_axis)
    mean_abs_shap = np.mean(np.abs(shap_arr), axis=axes_to_mean)
    mean_abs_shap = np.array(mean_abs_shap).flatten()

    # ★ 升级：将绝对值转换为 100% 的相对占比
    total_shap_importance = np.sum(mean_abs_shap)
    percentage_importance = (mean_abs_shap / total_shap_importance) * 100

    df_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance (%)': percentage_importance
    })
    
    # 升序排列（因为水平柱状图是从下往上画的，最小的放最底下）
    df_importance = df_importance.sort_values(by='Importance (%)', ascending=True)

    # ==========================================
    # 6. 绘制高颜值百分比柱状图
    # ==========================================
    plt.figure(figsize=(10, 6))
    
    # ★ 升级：引入 plasma 渐变色带 (数值越大的柱子颜色越明亮)
    # 你也可以把 'plasma' 换成 'viridis', 'magma', 或 'YlOrRd'
    norm = plt.Normalize(df_importance['Importance (%)'].min(), df_importance['Importance (%)'].max())
    colors = cm.plasma(norm(df_importance['Importance (%)']))

    # 绘制水平柱状图
    bars = plt.barh(df_importance['Feature'], df_importance['Importance (%)'], color=colors, edgecolor='black', linewidth=0.8)
    
    plt.xlabel('Relative Contribution to Model Output (%)', fontsize=12, fontweight='bold')
    plt.ylabel('Input Features', fontsize=12, fontweight='bold')
    plt.title('Global Feature Importance Ranking (SHAP Percentage)', fontsize=14, fontweight='bold', pad=15)
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    
    # 在柱子末端标注百分比数值（保留1位小数）
    for index, value in enumerate(df_importance['Importance (%)']):
        # 添加微小的缩进，防止文字贴着柱子太紧
        plt.text(value + 0.5, index, f"{value:.1f}%", va='center', fontsize=10, fontweight='bold')

    # 根据最大值稍微向右扩展一点 X 轴空间，防止文字被图表边缘切断
    plt.xlim(0, df_importance['Importance (%)'].max() * 1.15)

    plt.tight_layout()
    plot_path = f"{RESULT_DIR}/shap_feature_importance_percentage.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    csv_path = f"{RESULT_DIR}/shap_feature_importance_percentage.csv"
    df_importance.sort_values(by='Importance (%)', ascending=False).to_csv(csv_path, index=False)

    print(f"\n[SUCCESS] 2000样本百分比 SHAP 分析圆满完成！")
    print(f"📊 渐变色结果图表已保存至: {plot_path}")
    print(f"💾 具体数值数据已保存至: {csv_path}")

if __name__ == "__main__":
    run_shap_analysis()