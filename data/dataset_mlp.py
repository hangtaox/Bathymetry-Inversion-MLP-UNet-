import numpy as np
import xarray as xr
import torch
from torch.utils.data import Dataset
from scipy import ndimage
from pathlib import Path
import json

class BathymetryShipPointDataset(Dataset):
    """
    点监督 Dataset (用于 MLP)
    一个样本 = 一个 SWOT 网格点（至少 1 个船测）
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
        print("[Dataset] 构建 BathymetryShipPointDataset (MLP)")
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

        # 3. 构建点监督样本
        X_list, y_list, ij_list = [], [], []
        for i in range(self.H):
            for j in range(self.W):
                y = ship_grid[i, j]
                if np.isnan(y):
                    continue

                x_feat = self.features_grid[:, i, j]
                if np.any(np.isnan(x_feat)):
                    continue

                # 特征后拼接经纬度
                x = np.concatenate([x_feat, [self.lon2d[i, j]], [self.lat2d[i, j]]])
                X_list.append(x)
                y_list.append(y)
                ij_list.append((i, j))

        self.X = np.asarray(X_list, np.float32)
        self.y = np.asarray(y_list, np.float32).reshape(-1, 1)
        self.ij = np.asarray(ij_list, np.int32)

        # 4. 归一化
        if normalize:
            self.X_mean = self.X.mean(axis=0)
            self.X_std = self.X.std(axis=0) + 1e-6
            self.y_mean = self.y.mean()
            self.y_std = self.y.std() + 1e-6
            self.X = (self.X - self.X_mean) / self.X_std
            self.y = (self.y - self.y_mean) / self.y_std
        else:
            self.X_mean = self.X_std = None
            self.y_mean = self.y_std = None

        self.X = torch.from_numpy(self.X)
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
        return self.X[idx], self.y[idx]

class BathymetryInferenceDataset:
    """推理专用数据读取类，只读取特征数据并进行归一化"""
    
    def __init__(
        self,
        feature_paths: dict,
        normalization_params_path: str,
        lon_range=None,
        lat_range=None
    ):
        print("[推理数据集] 初始化...")
        
        # 加载归一化参数
        with open(normalization_params_path, 'r') as f:
            norm_params = json.load(f)
        
        self.X_mean = np.array(norm_params['x_means'], dtype=np.float32)
        self.X_std = np.array(norm_params['x_stds'], dtype=np.float32)
        self.y_mean = norm_params['y_mean']
        self.y_std = norm_params['y_std']
        
        # 读取特征数据
        feature_names = list(feature_paths.keys())
        feature_arrays = []
        
        for i, (name, path) in enumerate(feature_paths.items()):
            ds = xr.open_dataset(path)
            da = ds[list(ds.data_vars)[0]]
            
            if lon_range is not None:
                da = da.sel(lon=slice(*lon_range))
            if lat_range is not None:
                da = da.sel(lat=slice(*lat_range))
            
            # 保存第一个数据的坐标信息
            if i == 0:
                self.lon = da.lon.values
                self.lat = da.lat.values
                self.shape = da.shape
                self.lon2d, self.lat2d = np.meshgrid(self.lon, self.lat)
            
            feature_arrays.append(da.values)
            ds.close()
        
        # 堆叠特征
        self.features_grid = np.stack(feature_arrays, axis=0)  # (C, H, W)
        self.C, self.H, self.W = self.features_grid.shape
        
        # 构建完整特征网格（包含经纬度）
        self._build_full_features()
        
    def _build_full_features(self):
        """构建包含经纬度的完整特征张量"""
        # 创建空数组用于存储所有特征点
        total_features = self.C + 2  # 原始特征 + 经度 + 纬度
        
        # 展平所有网格点
        self.X_grid = np.full((self.H * self.W, total_features), np.nan, dtype=np.float32)
        
        valid_mask = np.zeros((self.H, self.W), dtype=bool)
        
        # 同时保存二维索引
        self.row_indices = []
        self.col_indices = []
        
        # 填充特征
        for i in range(self.H):
            for j in range(self.W):
                idx = i * self.W + j
                x_feat = self.features_grid[:, i, j]
                
                # 检查是否有NaN值（陆地位置）
                if not np.any(np.isnan(x_feat)):
                    # 特征 + 经度 + 纬度
                    self.X_grid[idx] = np.concatenate([
                        x_feat,
                        [self.lon2d[i, j]],
                        [self.lat2d[i, j]]
                    ])
                    valid_mask[i, j] = True
                    self.row_indices.append(i)
                    self.col_indices.append(j)
        
        self.valid_indices = np.where(valid_mask.flatten())[0]
        self.row_indices = np.array(self.row_indices, dtype=np.int64)
        self.col_indices = np.array(self.col_indices, dtype=np.int64)
        self.valid_mask_2d = valid_mask
        self.X_valid = self.X_grid[self.valid_indices]
        
        # 归一化
        self.X_valid_norm = (self.X_valid - self.X_mean) / self.X_std
        self.X_valid_tensor = torch.from_numpy(self.X_valid_norm)
        
        print(f"[推理数据集] 网格大小: {self.H}x{self.W}")
        print(f"[推理数据集] 有效海洋点: {len(self.valid_indices)}")
        print(f"[推理数据集] 总网格点数: {self.H * self.W}")
    
    def inverse_transform(self, y_norm):
        """反归一化深度值"""
        if isinstance(y_norm, torch.Tensor):
            y_norm = y_norm.numpy()
        return y_norm * self.y_std + self.y_mean
    
    def reconstruct_grid(self, predictions):
        """将预测结果重构回网格"""
        grid = np.full((self.H, self.W), np.nan, dtype=np.float32)
        
        # 将predictions展平为一维数组
        if predictions.ndim > 1:
            predictions = predictions.squeeze()  # 从 (N,1) 变为 (N,)
        
        print(f"重构网格调试信息:")
        print(f"  grid形状: {grid.shape}")  # 应该是 (1800, 1200)
        print(f"  valid_indices长度: {len(self.valid_indices)}")  # 应该是 1381787
        print(f"  predictions长度: {len(predictions)}")  # 应该是 1381787
        
        # 检查长度是否匹配
        if len(predictions) != len(self.valid_indices):
            print(f"警告：长度不匹配！predictions={len(predictions)}, valid_indices={len(self.valid_indices)}")
            # 取较小长度
            min_len = min(len(predictions), len(self.valid_indices))
            predictions = predictions[:min_len]
            valid_indices = self.valid_indices[:min_len]
        else:
            valid_indices = self.valid_indices
        
        # 将一维索引转换为二维索引
        rows = valid_indices // self.W
        cols = valid_indices % self.W
        
        print(f"  行索引范围: {rows.min()} to {rows.max()}")
        print(f"  列索引范围: {cols.min()} to {cols.max()}")
        
        # 直接赋值到二维网格
        grid[rows, cols] = predictions
        
        # 检查赋值结果
        non_nan_count = np.sum(~np.isnan(grid))
        print(f"直接赋值后非NaN点数: {non_nan_count}")
        
        if non_nan_count > 0:
            print(f"前10个赋值位置的值:")
            for k in range(min(10, len(predictions))):
                i, j = rows[k], cols[k]
                print(f"    位置[{i},{j}] = {predictions[k]:.2f}")
        
        return grid