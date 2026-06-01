import numpy as np
import pandas as pd
import xarray as xr
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr

# 导入推理专用 Dataset 和 CNN 模型
from data.dataset_cnn import BathymetryInferenceDataset  
from models.cnn import CNNEncoderFC

# =========================================================
# 【全局控制】当前要进行单模型分析和全域推理的感受野大小
# =========================================================
PATCH_SIZE = 13

# ============================================================
# 公共基础函数
# ============================================================
def depth_binned_statistics(y_true, y_pred, bin_size=200, min_points=30):
    """按深度分箱计算误差统计量"""
    bins = np.arange(y_true.min(), y_true.max() + bin_size, bin_size)
    stats = {"depth_center": [], "Number": [], "RMSE": [], "MAE": [], "Bias": [], "STD": []}
    residual = y_pred - y_true

    for i in range(len(bins) - 1):
        mask = (y_true >= bins[i]) & (y_true < bins[i + 1])
        n = mask.sum()
        stats["depth_center"].append(0.5 * (bins[i] + bins[i + 1]))
        stats["Number"].append(n)

        if n < min_points:
            for k in ["RMSE", "MAE", "Bias", "STD"]: stats[k].append(np.nan)
        else:
            r = residual[mask]
            stats["RMSE"].append(np.sqrt(np.mean(r**2)))
            stats["MAE"].append(np.mean(np.abs(r)))
            stats["Bias"].append(np.mean(r))
            stats["STD"].append(np.std(r))
    return pd.DataFrame(stats)

# ============================================================
# 模块一：单模型全面精度分析 (基于当前 PATCH_SIZE)
# ============================================================
def analyze_single_model_results(patch_size=PATCH_SIZE):
    """
    基于测试集 CSV 文件，绘制 4 张基础精度评估图
    """
    csv_path = f"./checkpoints/cnn/{patch_size}/test_results.csv"
    output_dir = f"./result/cnn/{patch_size}"
    
    print(f"\n[模块一] 正在处理 CNN 单模型全面精度分析 (Patch: {patch_size}x{patch_size})...")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    if not Path(csv_path).exists():
        print(f"    [跳过] 找不到 {csv_path}。请先运行该感受野的训练脚本！")
        return

    df = pd.read_csv(csv_path)
    y_true, y_pred = df["y_true"].values, df["y_pred"].values
    residual = y_pred - y_true

    # 1. 全局指标计算
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    bias = np.mean(residual)
    std = np.std(residual)  
    r2 = r2_score(y_true, y_pred)
    print(f"    -> MAE: {mae:.2f}m, RMSE: {rmse:.2f}m, STD: {std:.2f}m, Bias: {bias:.2f}m, R2: {r2:.4f}")

    # --- 图 1：Pred vs GT 散点图 ---
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=3, alpha=0.4)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, "r--", lw=1.5)
    plt.xlabel("Ground Truth Depth (m)")
    plt.ylabel("Predicted Depth (m)")
    plt.title(f"CNN Prediction vs Ground Truth (Patch {patch_size})")
    
    # 统一格式的左上角文本框
    textstr = f"MAE: {mae:.2f} m\nRMSE: {rmse:.2f} m\nSTD: {std:.2f} m\nBias: {bias:.2f} m\nR²: {r2:.3f}"
    props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
    plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
                   verticalalignment='top', bbox=props)

    plt.grid(alpha=0.3)
    plt.savefig(f"{output_dir}/scatter_pred_vs_gt.png", dpi=300, bbox_inches='tight')
    plt.close()

    # --- 图 2：误差直方图 ---
    plt.figure(figsize=(6, 4))
    plt.hist(residual, bins=100, density=True, alpha=0.7, color='steelblue')
    plt.axvline(0, color="r", linestyle="--", label="Zero Error")
    plt.xlabel("Residual (Pred - GT) [m]")
    plt.ylabel("Density")
    plt.title(f"CNN Error Distribution (Bias={bias:.2f} m)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(f"{output_dir}/residual_histogram.png", dpi=300, bbox_inches='tight')
    plt.close()

    # --- 图 3：误差随深度的分箱折线图 ---
    df_depth = depth_binned_statistics(y_true, y_pred, bin_size=200, min_points=50)
    plt.figure(figsize=(8, 5))
    plt.plot(df_depth["depth_center"], df_depth["MAE"], label="MAE", marker="s", color='blue')
    plt.plot(df_depth["depth_center"], df_depth["RMSE"], label="RMSE", marker="o", color='red')
    plt.plot(df_depth["depth_center"], df_depth["STD"], label="STD", marker="^", color='green')  
    plt.plot(df_depth["depth_center"], df_depth["Bias"], label="Bias", marker="x", color='orange') 
    plt.xlabel("Depth (m)")
    plt.ylabel("Error (m)")
    plt.title("CNN Error Statistics vs. Depth")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(f"{output_dir}/error_vs_depth.png", dpi=300, bbox_inches='tight')
    plt.close()

    # --- 图 4：空间残差热力分布图 ---
    plt.figure(figsize=(9, 7))
    vmax = np.percentile(np.abs(residual), 95)
    scatter = plt.scatter(df["lon"], df["lat"], c=residual, cmap='coolwarm', s=5, vmin=-vmax, vmax=vmax, alpha=0.8)
    plt.colorbar(scatter, label='Prediction Error (m)')
    plt.title(f'CNN Spatial Residual Map (Patch {patch_size})\nRMSE: {rmse:.2f} m')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.savefig(f"{output_dir}/spatial_residual_map.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("    -> 模块一分析完成，4 张基础评估图已保存。")

# ============================================================
# 模块二：全域网格推理 (基于当前 PATCH_SIZE)
# ============================================================
def run_full_inference(patch_size=PATCH_SIZE):
    """利用 InferenceDataset 进行纯 CNN 全域水深推断"""
    print(f"\n[模块二] 执行全域网格滑动推理 (CNN, Patch: {patch_size}x{patch_size})...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 【对齐】：严格保持 10 个物理特征！
    PATHS = {
        "grav": Path("./data/processed/G_ocean_raw.nc"),
        "curv": Path("./data/processed/VGG_ocean_raw.nc"),
        "b_lp": Path("./data/processed/B_LP.nc"),
        "g_lp": Path("./data/processed/G_LP.nc"),
        "g_bp": Path("./data/processed/G_BP.nc"),
        "vgg_bp": Path("./data/processed/VGG_BP.nc"),
        "g_east": Path("./data/processed/G_East.nc"),
        "g_north": Path("./data/processed/G_North.nc"),
        "g_east_bp": Path("./data/processed/G_East_BP.nc"),    
        "g_north_bp": Path("./data/processed/G_North_BP.nc")   
    }
    
    # 路径切换至 CNN 目录
    model_path = f"./checkpoints/cnn/{patch_size}/best_model.pt"
    NORM_PARAMS = f"./checkpoints/cnn/{patch_size}/normalization_params.json"
    OUTPUT_DIR = Path(f"./predictions/cnn/{patch_size}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if not Path(model_path).exists():
        print(f"    [错误] 找不到模型权重 {model_path}，请先运行 train_cnn_point.py 训练！")
        return None

    # 初始化推理数据集，缓存名必须加上 CNN，防冲突
    dataset = BathymetryInferenceDataset(
        feature_paths=PATHS, normalization_params_path=NORM_PARAMS,
        cache_dir=f"./cache/inference_cnn_{patch_size}",  
        lon_range=(105, 125), lat_range=(0, 30), patch_size=patch_size
    )
    
    with open(NORM_PARAMS, 'r') as f:
        input_dim = len(json.load(f)['x_means'])
    
    # 实例化动态 CNN 模型
    model = CNNEncoderFC(in_channels=input_dim, patch_size=patch_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    print("    -> 开始网络前向传播...")
    with torch.no_grad():
        patches_tensor = dataset.patches_tensor
        batch_size = 8192  # 纯 CNN 占用显存小，可以适当增大 Batch Size 提速
        predictions = []
        for i in range(0, len(patches_tensor), batch_size):
            batch = patches_tensor[i:i+batch_size].to(device)
            predictions.append(model(batch).cpu())
        predictions = torch.cat(predictions, dim=0)
    
    print("    -> 深度反归一化并重构地理网格...")
    predictions_real = dataset.inverse_transform(predictions.numpy().flatten())
    depth_grid = dataset.reconstruct_grid(predictions_real)
    
    ds = xr.Dataset(
        {"predicted_depth": (("lat", "lon"), depth_grid)}, 
        coords={"lon": dataset.lon, "lat": dataset.lat}
    )
    
    nc_out = OUTPUT_DIR / "predicted_bathymetry_cnn.nc"
    ds.to_netcdf(nc_out)
    
    # 全域深度图预览
    plt.figure(figsize=(10, 8))
    masked_data = np.ma.array(depth_grid, mask=np.isnan(depth_grid))
    vmin, vmax = np.percentile(depth_grid[~np.isnan(depth_grid)], [2, 98])
    im = plt.pcolormesh(dataset.lon, dataset.lat, masked_data, shading='auto', cmap='terrain', vmin=vmin, vmax=vmax)
    plt.colorbar(im, label='Depth (m)')
    plt.title(f'Predicted Bathymetry (CNN, Patch {patch_size})')
    plt.savefig(OUTPUT_DIR / "predicted_bathymetry_map.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"    -> 全域推理完成！已保存 NC 文件: {nc_out}")
    return nc_out

# ============================================================
# 主控调度
# ============================================================
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    
    # ---------------------------------------------------------
    # 测试执行 1: 基于当前 PATCH_SIZE 的 CNN 单模型精度评估
    # ---------------------------------------------------------
    analyze_single_model_results(patch_size=PATCH_SIZE)
    
    # ---------------------------------------------------------
    # 测试执行 2: 基于当前 PATCH_SIZE 的全图推理
    # ---------------------------------------------------------
    # run_full_inference(patch_size=PATCH_SIZE)
    
    print("\n===== CNN 推理与分析脚本执行结束 =====")