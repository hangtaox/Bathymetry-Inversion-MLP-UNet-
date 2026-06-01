import xarray as xr
import numpy as np

def extract_global_multibeam():
    # 1. 定义输入和输出文件路径 (请根据实际情况修改后缀，比如 2024 或 2025)
    # elev_file = r"D:\project\data\GEBCO_2024\gebco_2024\GEBCO_2024.nc"
    # tid_file = r"D:\project\data\GEBCO_2024\gebco_2024_tid\GEBCO_2024_TID.nc"
    # output_file = r"D:\project\data\label\Global_Multibeam_label_2024.nc"
    
    elev_file = r"D:\project\data\GEBCO_2025\gebco_2025\GEBCO_2025.nc"
    tid_file = r"D:\project\data\GEBCO_2025\gebco_2025_tid\GEBCO_2025_TID.nc"
    output_file = r"D:\project\data\label\Global_Multibeam_label_2025.nc"
    
    print("正在配置分块读取 (Dask) ...")
    # 2. 设定分块策略 (Chunking)
    # GEBCO 全球尺寸为纬度 43200 x 经度 86400。
    # 这里设置为 10800x10800，相当于把全球切割成了 4行 x 8列 = 32个分块。
    # 如果你的电脑内存较小（例如小于16GB），可以把数字调小，例如 5400 或 4320，分块越多内存占用越小。
    chunks_config = {'lat': 5400, 'lon': 5400}
    
    # 带有 chunks 参数打开，xarray 会使用 Dask 数组(惰性+分块)代替普通的 numpy 数组
    ds_elev = xr.open_dataset(elev_file, chunks=chunks_config)
    ds_tid = xr.open_dataset(tid_file, chunks=chunks_config)
    
    print("正在建立全局提取逻辑...")
    # 3. 提取多波束数据
    # 注意：此时并没有发生真实计算，而是建立了一个分块计算任务图
    multibeam_elev = ds_elev['elevation'].where(ds_tid['tid'] == 11)
    
    # 4. 构建输出 Dataset
    ds_out = xr.Dataset(
        {
            "elevation": multibeam_elev
        }
    )
    
    # 补充描述属性
    ds_out['elevation'].attrs = {
        "long_name": "Elevation relative to sea level (Multibeam only)",
        "units": "meters",
        "description": "Global extracted multibeam data (TID=11) from GEBCO"
    }
    
    # 设置压缩与空值填充格式
    encoding = {
        'elevation': {
            'zlib': True,       # 必须开启压缩，因为全球大部分区域会被填充为 NaN
            'complevel': 5,
            '_FillValue': np.nan 
        }
    }
    
    print(f"开始分块计算并流式写入最终文件...\n输出路径: {output_file}")
    print("这可能需要较长的时间（取决于硬盘读写速度），请耐心等待...")
    
    # 5. 执行计算并保存
    # to_netcdf 遇到 Dask 数组时，会自动触发底层任务：
    # “读取块A -> 执行where提取 -> 压缩 -> 写入文件块A -> 释放内存 -> 读取块B...”
    ds_out.to_netcdf(output_file, encoding=encoding)
    
    # 释放资源
    ds_elev.close()
    ds_tid.close()
    ds_out.close()
    
    print("全球多波束数据提取并拼合完成！")

if __name__ == "__main__":
    extract_global_multibeam()