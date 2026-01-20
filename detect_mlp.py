import numpy as np
import xarray as xr
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import json
from data.dataset_mlp import BathymetryInferenceDataset


def run_inference(model_path, output_dir="./predictions"):
    """执行推理并保存结果"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 参数设置
    PATHS = {
        "grav": Path("./data/processed/G_ocean_raw.nc"),
        "curv": Path("./data/processed/VGG_ocean_raw.nc"),
        "b_lp": Path("./data/processed/B_LP.nc"),
        "g_lp": Path("./data/processed/G_LP.nc"),
        "g_bp": Path("./data/processed/G_BP.nc"),
        "vgg_bp": Path("./data/processed/VGG_BP.nc"),
    }
    
    LON_RANGE = (105, 125)
    LAT_RANGE = (0, 30)
    NORM_PARAMS = "./checkpoints/mlp/normalization_params.json"
    OUTPUT_DIR = Path(output_dir)
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # 1. 加载推理数据集
    print("加载推理数据集...")
    dataset = BathymetryInferenceDataset(
        feature_paths=PATHS,
        normalization_params_path=NORM_PARAMS,
        lon_range=LON_RANGE,
        lat_range=LAT_RANGE
    )
    
    # 调试信息：检查数据集
    print(f"数据集特征张量形状: {dataset.X_valid_tensor.shape}")
    print(f"数据集特征张量类型: {dataset.X_valid_tensor.dtype}")
    print(f"数据集特征是否有NaN: {torch.isnan(dataset.X_valid_tensor).any()}")
    
    # 2. 加载训练好的模型
    print("加载模型...")
    
    # 首先需要导入你的模型定义
    try:
        from models.mlp import ResidualMLP
    except ImportError:
        print("错误：无法导入模型定义")
        print("请确保模型定义文件存在并可导入")
        return
    
    # 从归一化参数获取输入特征维度
    with open(NORM_PARAMS, 'r') as f:
        norm_params = json.load(f)
    
    # 输入特征维度 = 特征数量 + 2（经度+纬度）
    input_dim = len(norm_params['x_means'])
    print(f"归一化参数中的特征维度: {input_dim}")
    
    # 实例化模型（需要根据你的实际模型参数调整）
    model = ResidualMLP(
        in_dim=8,
        hidden_dim=128,
        num_layers=6,
        out_dim=1
    ).to(device)
    
    # 加载模型参数
    checkpoint = torch.load(model_path, weights_only=False)
    
    # 检查加载的内容类型
    if isinstance(checkpoint, dict):
        print("检查点类型: 字典")
        print(f"检查点键: {checkpoint.keys()}")
        if 'model_state_dict' in checkpoint:
            # 如果是完整的checkpoint（包含模型、优化器等）
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            # 如果是纯state_dict
            model.load_state_dict(checkpoint)
    else:
        print("错误：无法识别模型文件格式")
        return
    
    model.eval()
    
    # 3. 执行推理
    print("执行推理...")
    with torch.no_grad():
        # 先测试一个小批量
        test_batch = dataset.X_valid_tensor[:10].to(device)
        print(f"测试批量形状: {test_batch.shape}")
        test_output = model(test_batch)
        print(f"测试输出: {test_output.cpu().numpy().flatten()}")
        print(f"测试输出是否有NaN: {torch.isnan(test_output).any()}")
        
        # 分批推理以避免内存问题
        batch_size = 8192
        predictions = []
        
        for i in range(0, len(dataset.X_valid_tensor), batch_size):
            batch = dataset.X_valid_tensor[i:i+batch_size]
            
            # 将数据移动到与模型相同的设备
            batch = batch.to(device)
            
            # 使用模型进行推理
            batch_pred = model(batch)
            
            predictions.append(batch_pred.cpu())  # 移回CPU保存
        
        predictions = torch.cat(predictions, dim=0)
    
    # 调试信息：检查预测结果
    print(f"预测结果形状: {predictions.shape}")
    print(f"预测结果范围: {predictions.min():.4f} to {predictions.max():.4f}")
    print(f"预测结果是否有NaN: {torch.isnan(predictions).any()}")
    
    # 4. 反归一化
    print("反归一化结果...")
    predictions_norm = predictions.numpy()
    predictions_real = dataset.inverse_transform(predictions_norm)
    
    print(f"反归一化后形状: {predictions_real.shape}")
    print(f"反归一化后范围: {predictions_real.min():.2f} to {predictions_real.max():.2f}")
    print(f"反归一化后是否有NaN: {np.isnan(predictions_real).any()}")
    
    # 5. 重构为网格
    print("重构网格...")
    depth_grid = dataset.reconstruct_grid(predictions_real)
    
    # 调试信息：检查重构网格
    print(f"重构网格形状: {depth_grid.shape}")
    print(f"重构网格中有效点数量: {np.sum(~np.isnan(depth_grid))}")
    print(f"重构网格中有效点比例: {np.sum(~np.isnan(depth_grid)) / (depth_grid.shape[0] * depth_grid.shape[1]) * 100:.2f}%")
    
    if np.sum(~np.isnan(depth_grid)) == 0:
        print("警告：重构网格中没有有效点！")
        print("检查 valid_indices 数量:", len(dataset.valid_indices))
        print("检查 predictions_real 长度:", len(predictions_real))
        print("检查 valid_indices 和 predictions_real 是否匹配")
        
        # 检查前几个点的值
        print(f"前10个有效索引: {dataset.valid_indices[:10]}")
        print(f"前10个预测值: {predictions_real[:10].flatten()}")
    
    # 只打印有有效数据的统计信息
    valid_mask = ~np.isnan(depth_grid)
    if np.any(valid_mask):
        valid_data = depth_grid[valid_mask]
        print(f"预测深度统计:")
        print(f"  最小值: {valid_data.min():.2f} m")
        print(f"  最大值: {valid_data.max():.2f} m")
        print(f"  平均值: {valid_data.mean():.2f} m")
        print(f"  标准差: {valid_data.std():.2f} m")
        print(f"  非NaN点数: {len(valid_data)}")
    else:
        print("警告：没有有效数据！")
        return None
    
    # 6. 保存为NetCDF文件
    print("保存NetCDF文件...")
    ds = xr.Dataset(
        {
            "predicted_depth": (("lat", "lon"), depth_grid)
        },
        coords={
            "lon": dataset.lon,
            "lat": dataset.lat,
        }
    )
    
    ds.attrs["description"] = "MLP predicted ocean depth"
    ds.attrs["model"] = "ResidualMLP"
    ds.attrs["lon_range"] = str(LON_RANGE)
    ds.attrs["lat_range"] = str(LAT_RANGE)
    
    nc_path = OUTPUT_DIR / "predicted_bathymetry.nc"
    ds.to_netcdf(nc_path)
    print(f"NetCDF文件已保存: {nc_path}")
    
    # 7. 保存为PNG图像（只有有效数据时才创建）
    if np.any(~np.isnan(depth_grid)):
        print("生成可视化图像...")
        
        # 创建两个图形：一个用于保存，一个用于显示检查
        # 图形1：用于保存的标准可视化
        fig_save = plt.figure(figsize=(12, 8))
        
        # 创建掩码以显示陆地（NaN值）
        masked_data = np.ma.array(depth_grid, mask=np.isnan(depth_grid))
        
        # 自动计算合适的vmin, vmax
        valid_data = depth_grid[~np.isnan(depth_grid)]
        if len(valid_data) > 0:
            vmin, vmax = np.percentile(valid_data, [5, 95])
        else:
            vmin, vmax = 0, 5000
        
        # 标准可视化
        im = plt.pcolormesh(dataset.lon, dataset.lat, masked_data, 
                        shading='auto', cmap='viridis_r',
                        vmin=vmin, vmax=vmax)
        plt.colorbar(im, label='Depth (m)')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title(f'MLP Predicted Bathymetry\nRange: {vmin:.0f} to {vmax:.0f} m')
        plt.grid(True, alpha=0.3)
        
        png_path = OUTPUT_DIR / "predicted_bathymetry.png"
        plt.tight_layout()
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.close(fig_save)  # 关闭保存的图形
        print(f"PNG图像已保存: {png_path}")
        
        # 图形2：用于检查大于0的区域的图形
        print("\n检查大于0的区域...")
        fig_check = plt.figure(figsize=(12, 8))
        
        # 找出大于0的区域
        positive_mask = depth_grid > 0
        positive_count = np.sum(positive_mask & ~np.isnan(depth_grid))
        print(f"预测值大于0的点数: {positive_count}")
        print(f"大于0的点占总有效点的比例: {positive_count/len(valid_data)*100:.2f}%")
        
        if positive_count > 0:
            # 获取大于0的点的坐标
            positive_indices = np.where(positive_mask)
            positive_rows = positive_indices[0]
            positive_cols = positive_indices[1]
            
            print(f"前10个大于0的点的位置和值:")
            for k in range(min(10, len(positive_rows))):
                i, j = positive_rows[k], positive_cols[k]
                value = depth_grid[i, j]
                lon_val = dataset.lon[j]
                lat_val = dataset.lat[i]
                print(f"  位置[{i},{j}] (经度{lon_val:.2f}, 纬度{lat_val:.2f}) = {value:.2f} m")
        
        # 创建基础图像
        im_check = plt.pcolormesh(dataset.lon, dataset.lat, masked_data, 
                                shading='auto', cmap='viridis_r',
                                vmin=vmin, vmax=vmax)
        plt.colorbar(im_check, label='Depth (m)')
        
        # 用红色标记大于0的区域
        if positive_count > 0:
            # 获取大于0的点的经纬度
            positive_lons = dataset.lon[positive_cols]
            positive_lats = dataset.lat[positive_rows]
            
            # 绘制红色散点标记
            plt.scatter(positive_lons, positive_lats, 
                    color='red', s=5, alpha=0.6, 
                    label=f'Depth > 0 m ({positive_count} points)')
            plt.legend()
        
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title(f'MLP Predicted Bathymetry with Depth>0 Highlighted (Red)\nTotal >0: {positive_count} points')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 显示检查图形
        plt.show()
        plt.close(fig_check)
        
        # 图形3：创建专门显示大于0区域分布的图形
        print("\n生成大于0区域的详细分析图...")
        fig_analysis, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 子图1：原始预测
        ax1 = axes[0, 0]
        im1 = ax1.pcolormesh(dataset.lon, dataset.lat, masked_data, 
                            shading='auto', cmap='viridis_r',
                            vmin=vmin, vmax=vmax)
        plt.colorbar(im1, ax=ax1, label='Depth (m)')
        ax1.set_title('Original Prediction')
        ax1.set_xlabel('Longitude')
        ax1.set_ylabel('Latitude')
        
        # 子图2：大于0的区域的二值掩码
        ax2 = axes[0, 1]
        positive_mask_display = np.zeros_like(depth_grid)
        positive_mask_display[positive_mask] = 1
        positive_masked = np.ma.array(positive_mask_display, mask=np.isnan(depth_grid))
        im2 = ax2.pcolormesh(dataset.lon, dataset.lat, positive_masked, 
                            shading='auto', cmap='Reds', vmin=0, vmax=1)
        plt.colorbar(im2, ax=ax2, label='Depth > 0 (1=Yes)')
        ax2.set_title(f'Depth > 0 Regions ({positive_count} points)')
        ax2.set_xlabel('Longitude')
        ax2.set_ylabel('Latitude')
        
        # 子图3：深度值的直方图
        ax3 = axes[1, 0]
        ax3.hist(valid_data, bins=100, edgecolor='black', alpha=0.7)
        ax3.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Sea Level (0 m)')
        ax3.set_xlabel('Depth (m)')
        ax3.set_ylabel('Frequency')
        ax3.set_title(f'Depth Distribution (Mean: {valid_data.mean():.2f} m)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 子图4：大于0的点的位置散点图
        ax4 = axes[1, 1]
        if positive_count > 0:
            # 绘制底图
            im4 = ax4.pcolormesh(dataset.lon, dataset.lat, masked_data, 
                                shading='auto', cmap='viridis_r',
                                vmin=vmin, vmax=vmax, alpha=0.5)
            plt.colorbar(im4, ax=ax4, label='Depth (m)')
            
            # 根据深度值大小绘制不同颜色的散点
            positive_values = depth_grid[positive_mask]
            
            # 创建颜色映射
            scatter = ax4.scatter(positive_lons, positive_lats, 
                                c=positive_values, cmap='hot_r',
                                s=20, alpha=0.8, edgecolors='black', linewidth=0.5)
            plt.colorbar(scatter, ax=ax4, label='Positive Depth (m)')
            ax4.set_title(f'Positive Depth Points Distribution\nMax: {positive_values.max():.2f} m')
        else:
            ax4.text(0.5, 0.5, 'No Depth > 0 points', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            ax4.set_title('No Positive Depth Points')
        
        ax4.set_xlabel('Longitude')
        ax4.set_ylabel('Latitude')
        
        plt.suptitle(f'Analysis of Positive Depth Predictions (Total >0: {positive_count})', fontsize=14)
        plt.tight_layout()
        
        # 保存分析图
        analysis_path = OUTPUT_DIR / "positive_depth_analysis.png"
        plt.savefig(analysis_path, dpi=300, bbox_inches='tight')
        print(f"分析图已保存: {analysis_path}")
        
        # 显示分析图
        plt.show()
        plt.close(fig_analysis)
    
    return ds, depth_grid


if __name__ == "__main__":
    # 在这里指定你的模型路径
    MODEL_PATH = "./checkpoints/mlp/best_model.pt"

    # 执行推理
    result = run_inference(MODEL_PATH)
    if result is not None:
        print("推理完成！")
    else:
        print("推理失败！")