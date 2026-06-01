import xarray as xr
import numpy as np

def extract_multibeam_data():
    # 1. 定义输入和输出文件路径
    # elev_file = r"D:\project\data\GEBCO_2025\gebco_2025\GEBCO_2025.nc"
    # tid_file = r"D:\project\data\GEBCO_2025\gebco_2025_tid\GEBCO_2025_TID.nc"
    # output_file = r"D:\project\data\label\label_2025.nc"
    
    # elev_file = r"D:\project\data\GEBCO_2024\gebco_2024\GEBCO_2024.nc"
    # tid_file = r"D:\project\data\GEBCO_2024\gebco_2024_tid\GEBCO_2024_TID.nc"
    # output_file = r"D:\project\data\label\Singelbeam_label_2024.nc"

    # 2. 定义经纬度范围
    lon_range = (104, 122)
    lat_range = (0, 26)
    
    print("正在打开 GEBCO 数据集 (惰性加载)...")
    # 使用 xarray 打开数据集，此时不会将整个几个 G 的数据读入内存
    ds_elev = xr.open_dataset(elev_file)
    ds_tid = xr.open_dataset(tid_file)
    
    print(f"正在裁剪经度 {lon_range} 和纬度 {lat_range} 范围...")
    # 3. 裁剪目标经纬度区域
    # 使用 .sel 和 slice 进行切片，极大缩小数据处理范围
    region_elev = ds_elev.sel(
        lon=slice(lon_range[0], lon_range[1]), 
        lat=slice(lat_range[0], lat_range[1])
    )
    region_tid = ds_tid.sel(
        lon=slice(lon_range[0], lon_range[1]), 
        lat=slice(lat_range[0], lat_range[1])
    )
    
    print("正在提取多波束测量点 (TID == 11) ...")
    # 4. 提取多波束数据
    # .where() 会保留满足条件 (TID == 11) 的 elevation 值 ////10是单波束
    # 不满足条件的地方（如卫星反演数据或陆地）会被赋值为 NaN
    multibeam_elev = region_elev['elevation'].where(region_tid['tid'] == 10)
    
    # 5. 构建输出 Dataset
    # 保持与原有格式一致，变量名为 'elevation'，带有 'lon' 和 'lat' 坐标
    ds_out = xr.Dataset(
        {
            "elevation": multibeam_elev
        }
    )
    
    # 补充一些描述属性（可选，让生成的 nc 文件更规范）
    ds_out['elevation'].attrs = {
        "long_name": "Elevation relative to sea level (Multibeam only)",
        "units": "meters",
        "description": "Extracted multibeam data (TID=11) from GEBCO_2024"
    }
    
    print(f"准备保存数据，网格大小: {ds_out['elevation'].shape} ...")
    
    # 6. 保存为新的 NetCDF 文件
    # 由于只有多波束点有数据，其余大量点为 NaN，开启 zlib 压缩可以极大减小输出文件的体积
    encoding = {
        'elevation': {
            'zlib': True,       # 开启压缩
            'complevel': 5,     # 压缩等级 (1-9)
            '_FillValue': np.nan # 指定填充值/空值标志
        }
    }
    
    ds_out.to_netcdf(output_file, encoding=encoding)
    
    # 释放资源
    ds_elev.close()
    ds_tid.close()
    ds_out.close()
    
    print(f"处理完成！文件已成功保存至: {output_file}")

if __name__ == "__main__":
    extract_multibeam_data()