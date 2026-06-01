import numpy as np
import xarray as xr
import torch
from torch.utils.data import Dataset
from scipy import ndimage
from pathlib import Path
import json

class BathymetryShipPointDataset(Dataset):
    """
    点监督 Dataset (用于 3x3 SIREN)
    返回 3x3 的局部 Patch 及有效区域掩码
    """
    def __init__(
        self,
        feature_paths: dict,
        ship_nc_path: str,
        aligned_ship_nc_path: str | None = None,
        lon_range=None,
        lat_range=None,
        agg_method="median",
        normalize=True,
    ):
        print("[Dataset] 构建 BathymetryShipPointDataset (3x3 SIREN)")
        self.normalize = normalize

        # 1. 读取 SWOT 栅格特征
        feature_names = list(feature_paths.keys())
        feature_arrays = []
        ref_lon = ref_lat = None
        target_shape = None

        for name in feature_names:
            ds = xr.open_dataset(feature_paths[name])
            da = ds[list(ds.data_vars)[0]]

            if lon_range is not None:
                da = da.sel(lon=slice(*lon_range))
            if lat_range is not None:
                da = da.sel(lat=slice(*lat_range))

            if target_shape is None:
                ref_lon = da.lon.values
                ref_lat = da.lat.values
                target_shape = da.shape
                feature_arrays.append(da.values)
            else:
                if da.shape != target_shape:
                    values = self._resample_to_target(da.values, target_shape)
                else:
                    values = da.values
                feature_arrays.append(values)
            ds.close()

        self.features_grid = np.stack(feature_arrays, axis=0)  # (C,H,W)
        self.C, self.H, self.W = self.features_grid.shape
        self.lon = ref_lon
        self.lat = ref_lat
        self.lon2d, self.lat2d = np.meshgrid(self.lon, self.lat)

        # 2. 船测 → 网格
        aligned_ship_nc_path = Path(aligned_ship_nc_path) if aligned_ship_nc_path else None
        if aligned_ship_nc_path is not None and aligned_ship_nc_path.exists():
            print(f"[Dataset] 读取已缓存船测网格: {aligned_ship_nc_path}")
            ds = xr.open_dataset(aligned_ship_nc_path)
            ship_grid = ds["ship_depth"].values
            ds.close()
        else:
            print("[Dataset] 执行船测点 → SWOT 网格对齐")
            ship_grid = self._grid_ship_points(ship_nc_path, lon_range, lat_range, agg_method)
            if aligned_ship_nc_path is not None:
                aligned_ship_nc_path.parent.mkdir(parents=True, exist_ok=True)
                xr.Dataset({"ship_depth": (("lat", "lon"), ship_grid)},
                           coords={"lon": self.lon, "lat": self.lat}).to_netcdf(aligned_ship_nc_path)

        # 3. 构建 3x3 Patch 样本
        X_list, mask_list, y_list, ij_list = [], [], [], []
        
        # 避开最外层1个像素，方便截取 3x3
        for i in range(1, self.H - 1):
            for j in range(1, self.W - 1):
                y = ship_grid[i, j]
                if np.isnan(y):
                    continue

                # 中心点必须是海洋有效数据
                if np.isnan(self.features_grid[0, i, j]):
                    continue

                # 截取 3x3 物理特征
                x_patch = self.features_grid[:, i-1:i+2, j-1:j+2]
                
                # 截取 3x3 经纬度并拼接
                lon_patch = self.lon2d[i-1:i+2, j-1:j+2][np.newaxis, :, :]
                lat_patch = self.lat2d[i-1:i+2, j-1:j+2][np.newaxis, :, :]
                full_patch = np.concatenate([x_patch, lon_patch, lat_patch], axis=0) # [C+2, 3, 3]
                
                # 构建掩码 (1 为有效，0 为 NaN)
                mask_patch = ~np.isnan(full_patch[0:1]) # [1, 3, 3] 布尔值

                X_list.append(full_patch)
                mask_list.append(mask_patch)
                y_list.append(y)
                ij_list.append((i, j))

        self.X = np.asarray(X_list, dtype=np.float32)       # [N, C+2, 3, 3]
        self.mask = np.asarray(mask_list, dtype=np.float32) # [N, 1, 3, 3]
        self.y = np.asarray(y_list, dtype=np.float32).reshape(-1, 1) # [N, 1]
        self.ij = np.asarray(ij_list, dtype=np.int32)

        # 4. 忽略 NaN 计算归一化参数
        if normalize:
            # 沿样本数(N)、高度(3)、宽度(3)计算每个通道的均值和方差
            self.X_mean = np.nanmean(self.X, axis=(0, 2, 3), keepdims=True) # [1, C+2, 1, 1]
            self.X_std = np.nanstd(self.X, axis=(0, 2, 3), keepdims=True) + 1e-6
            self.y_mean = np.nanmean(self.y)
            self.y_std = np.nanstd(self.y) + 1e-6
            
            self.X = (self.X - self.X_mean) / self.X_std
            self.y = (self.y - self.y_mean) / self.y_std
        else:
            self.X_mean = self.X_std = None
            self.y_mean = self.y_std = None

        self.X = torch.from_numpy(self.X)
        self.mask = torch.from_numpy(self.mask)
        self.y = torch.from_numpy(self.y)
        print(f"[Dataset] 样本数: {len(self.y)}")

    def _grid_ship_points(self, ship_nc_path, lon_range, lat_range, agg_method):
        ds = xr.open_dataset(ship_nc_path)
        lon = ds["lon"].values
        lat = ds["lat"].values
        depth = ds["depth"].values
        ds.close()

        mask = np.ones_like(lon, dtype=bool)
        if lon_range: mask &= (lon >= lon_range[0]) & (lon <= lon_range[1])
        if lat_range: mask &= (lat >= lat_range[0]) & (lat <= lat_range[1])
        lon, lat, depth = lon[mask], lat[mask], depth[mask]

        grid = np.full((self.H, self.W), np.nan)
        def lonlat_to_ij(lo, la):
            j = np.searchsorted(self.lon, lo) - 1
            i = np.searchsorted(self.lat, la) - 1
            return i, j

        tmp = {}
        for lo, la, d in zip(lon, lat, depth):
            i, j = lonlat_to_ij(lo, la)
            if 0 <= i < self.H and 0 <= j < self.W:
                tmp.setdefault((i, j), []).append(d)

        for (i, j), v in tmp.items():
            grid[i, j] = np.median(v) if agg_method == "median" else np.mean(v)
        return grid

    def _resample_to_target(self, data, target_shape):
        scale = (target_shape[0]/data.shape[0], target_shape[1]/data.shape[1])
        return ndimage.zoom(data, scale, order=1)

    def inverse_transform_y(self, y_norm):
        return y_norm * self.y_std + self.y_mean

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.mask[idx], self.y[idx]

class BathymetryInferenceDataset:
    """推理专用数据读取类 (适配 3x3 SIREN)"""
    
    def __init__(
        self,
        feature_paths: dict,
        normalization_params_path: str,
        lon_range=None,
        lat_range=None
    ):
        print("[推理数据集] 初始化 3x3 SIREN 推理引擎...")
        
        with open(normalization_params_path, 'r') as f:
            norm_params = json.load(f)
        
        # 恢复归一化维度为 [1, C+2, 1, 1] 以便对 3x3 进行广播
        self.X_mean = np.array(norm_params['x_means'], dtype=np.float32).reshape(1, -1, 1, 1)
        self.X_std = np.array(norm_params['x_stds'], dtype=np.float32).reshape(1, -1, 1, 1)
        self.y_mean = norm_params['y_mean']
        self.y_std = norm_params['y_std']
        
        feature_names = list(feature_paths.keys())
        feature_arrays = []
        
        for i, (name, path) in enumerate(feature_paths.items()):
            ds = xr.open_dataset(path)
            da = ds[list(ds.data_vars)[0]]
            
            if lon_range is not None:
                da = da.sel(lon=slice(*lon_range))
            if lat_range is not None:
                da = da.sel(lat=slice(*lat_range))
            
            if i == 0:
                self.lon = da.lon.values
                self.lat = da.lat.values
                self.shape = da.shape
                self.lon2d, self.lat2d = np.meshgrid(self.lon, self.lat)
            
            feature_arrays.append(da.values)
            ds.close()
        
        self.features_grid = np.stack(feature_arrays, axis=0)  # (C, H, W)
        self.C, self.H, self.W = self.features_grid.shape
        
        self._build_full_features()
        
    def _build_full_features(self):
        X_list, mask_list, row_list, col_list = [], [], [], []
        
        # 填充外圈以处理边缘推断 (保证输出尺寸一致)
        fg_pad = np.pad(self.features_grid, ((0,0), (1,1), (1,1)), constant_values=np.nan)
        lon_pad = np.pad(self.lon2d, ((1,1), (1,1)), constant_values=np.nan)
        lat_pad = np.pad(self.lat2d, ((1,1), (1,1)), constant_values=np.nan)
        
        # 遍历原始坐标范围 (在 padded 数组中的索引就是 i+1, j+1)
        for i in range(self.H):
            for j in range(self.W):
                # 只有中心点有效才预测
                if np.isnan(fg_pad[0, i+1, j+1]):
                    continue
                
                # 提取 3x3 (在 padding 后的数组中)
                x_patch = fg_pad[:, i:i+3, j:j+3]
                l_patch = lon_pad[i:i+3, j:j+3][np.newaxis, :, :]
                la_patch = lat_pad[i:i+3, j:j+3][np.newaxis, :, :]
                
                full_patch = np.concatenate([x_patch, l_patch, la_patch], axis=0)
                mask_patch = ~np.isnan(full_patch[0:1])
                
                X_list.append(full_patch)
                mask_list.append(mask_patch)
                row_list.append(i)
                col_list.append(j)
                
        self.X_valid = np.asarray(X_list, dtype=np.float32)
        self.mask_valid = np.asarray(mask_list, dtype=np.float32)
        self.row_indices = np.array(row_list, dtype=np.int64)
        self.col_indices = np.array(col_list, dtype=np.int64)
        
        # 归一化
        self.X_valid_norm = (self.X_valid - self.X_mean) / self.X_std
        self.X_valid_tensor = torch.from_numpy(self.X_valid_norm)
        self.mask_valid_tensor = torch.from_numpy(self.mask_valid)
        
        print(f"[推理数据集] 网格大小: {self.H}x{self.W}")
        print(f"[推理数据集] 有效海洋推断点: {len(self.X_valid)}")
    
    def inverse_transform(self, y_norm):
        if isinstance(y_norm, torch.Tensor):
            y_norm = y_norm.numpy()
        return y_norm * self.y_std + self.y_mean
    
    def reconstruct_grid(self, predictions):
        grid = np.full((self.H, self.W), np.nan, dtype=np.float32)
        if predictions.ndim > 1:
            predictions = predictions.squeeze()
        
        grid[self.row_indices, self.col_indices] = predictions
        print(f"直接赋值后非NaN点数: {np.sum(~np.isnan(grid))}")
        return grid