import numpy as np
import torch
from torch.utils.data import Dataset
import xarray as xr
from pathlib import Path
from scipy import ndimage
import json

class BathymetryShipPointCNNDataset(Dataset):
    """
    CNN 点监督 Dataset
    一个样本 = 一个 SWOT 网格点（对应一个船测深度）
    输入: [C, k, k]  或 [C, 1, 1]
    输出: [1]
    """

    def __init__(
        self,
        feature_paths: dict,
        ship_nc_path: str,
        aligned_ship_nc_path: str | None = None,
        lon_range=None,
        lat_range=None,
        agg_method="median",
        patch_size=1,          
        normalize=True,
    ):
        print("[Dataset] 构建 BathymetryShipPointCNNDataset")

        self.patch_size = patch_size
        self.normalize = normalize

        # ======================================================
        # 1. 读取 SWOT 特征（作为参考网格）
        # ======================================================
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

        self.features = np.stack(feature_arrays, axis=0)  # (C,H,W)
        self.C, self.H, self.W = self.features.shape

        self.lon = ref_lon
        self.lat = ref_lat
        self.lon2d, self.lat2d = np.meshgrid(self.lon, self.lat)
        # ======================================================
        # 经纬度作为特征通道
        # ======================================================
        lon_feat = self.lon2d.astype(np.float32)
        lat_feat = self.lat2d.astype(np.float32)

        self.features = np.concatenate(
            [self.features, lon_feat[None, ...], lat_feat[None, ...]],
            axis=0
        )

        self.C += 2

        # ======================================================
        # 2. 船测点 → SWOT 网格
        # ======================================================
        aligned_ship_nc_path = (
            Path(aligned_ship_nc_path) if aligned_ship_nc_path else None
        )

        if aligned_ship_nc_path and aligned_ship_nc_path.exists():
            ds = xr.open_dataset(aligned_ship_nc_path)
            cache_lon = ds.lon.values
            cache_lat = ds.lat.values
            
            # 【核心修复 1】：增加网格坐标一致性严格校验！防止改变范围后读取错误缓存。
            if len(cache_lon) == len(self.lon) and len(cache_lat) == len(self.lat) \
               and np.allclose(cache_lon, self.lon) and np.allclose(cache_lat, self.lat):
                print(f"[Dataset] 坐标域校验通过，读取缓存船测网格: {aligned_ship_nc_path}")
                ship_grid = ds["ship_depth"].values
                ds.close()
            else:
                print(f"[Dataset] 检测到缓存网格与当前坐标域不匹配！正在重新执行对齐...")
                ds.close()
                ship_grid = self._grid_ship_points(ship_nc_path, lon_range, lat_range, agg_method)
                xr.Dataset(
                    {"ship_depth": (("lat", "lon"), ship_grid)},
                    coords={"lon": self.lon, "lat": self.lat},
                ).to_netcdf(aligned_ship_nc_path)
                print(f"[Dataset] 已覆写更新对齐网格缓存。")
        else:
            print("[Dataset] 执行船测点 → SWOT 网格对齐")
            ship_grid = self._grid_ship_points(
                ship_nc_path, lon_range, lat_range, agg_method
            )

            if aligned_ship_nc_path:
                aligned_ship_nc_path.parent.mkdir(parents=True, exist_ok=True)
                xr.Dataset(
                    {"ship_depth": (("lat", "lon"), ship_grid)},
                    coords={"lon": self.lon, "lat": self.lat},
                ).to_netcdf(aligned_ship_nc_path)

        # ======================================================
        # 3. 构建点监督样本
        # ======================================================
        X_list, y_list = [], []
        lon_list, lat_list = [], []
        
        r = patch_size // 2

        for i in range(self.H):
            for j in range(self.W):
                y = ship_grid[i, j]
                if np.isnan(y):
                    continue

                if i - r < 0 or i + r >= self.H:
                    continue
                if j - r < 0 or j + r >= self.W:
                    continue

                x = self.features[:, i - r:i + r + 1, j - r:j + r + 1]
                if np.any(np.isnan(x)):
                    continue

                X_list.append(x)
                y_list.append(y)

        self.X = np.asarray(X_list, np.float32)
        self.y = np.asarray(y_list, np.float32).reshape(-1, 1)

        # ======================================================
        # 4. 归一化
        # ======================================================
        if normalize:
            self.X_mean = self.X.mean(axis=(0, 2, 3), keepdims=True)
            self.X_std = self.X.std(axis=(0, 2, 3), keepdims=True) + 1e-6
            self.y_mean = self.y.mean()
            self.y_std = self.y.std() + 1e-6

            self.X = (self.X - self.X_mean) / self.X_std
            self.y = (self.y - self.y_mean) / self.y_std

        self.X = torch.from_numpy(self.X)
        self.y = torch.from_numpy(self.y)

        print(f"[Dataset] 样本数: {len(self.y)}")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def inverse_transform_y(self, y_norm):
        return y_norm * self.y_std + self.y_mean

    def _grid_ship_points(self, ship_nc_path, lon_range, lat_range, agg_method):
        ds = xr.open_dataset(ship_nc_path)
        lon = ds["lon"].values
        lat = ds["lat"].values
        depth = ds["depth"].values
        source = ds["source"].values if "source" in ds else np.zeros_like(depth)
        ds.close()

        mask = np.ones_like(lon, dtype=bool)
        if lon_range:
            mask &= (lon >= lon_range[0]) & (lon <= lon_range[1])
        if lat_range:
            mask &= (lat >= lat_range[0]) & (lat <= lat_range[1])

        lon, lat, depth, source = lon[mask], lat[mask], depth[mask], source[mask]

        grid = np.full((self.H, self.W), np.nan)

        def lonlat_to_ij(lo, la):
            j = np.searchsorted(self.lon, lo) - 1
            i = np.searchsorted(self.lat, la) - 1
            return i, j

        tmp_mb = {}
        tmp_sb = {}
        
        for lo, la, d, src in zip(lon, lat, depth, source):
            i, j = lonlat_to_ij(lo, la)
            if 0 <= i < self.H and 0 <= j < self.W:
                if src == 1:
                    tmp_mb.setdefault((i, j), []).append(d)
                else:
                    tmp_sb.setdefault((i, j), []).append(d)

        for i in range(self.H):
            for j in range(self.W):
                if (i, j) in tmp_mb:
                    grid[i, j] = np.median(tmp_mb[(i, j)]) if agg_method == "median" else np.mean(tmp_mb[(i, j)])
                elif (i, j) in tmp_sb:
                    grid[i, j] = np.median(tmp_sb[(i, j)]) if agg_method == "median" else np.mean(tmp_sb[(i, j)])

        return grid

    def _resample_to_target(self, data, target_shape):
        scale = (
            target_shape[0] / data.shape[0],
            target_shape[1] / data.shape[1],
        )
        return ndimage.zoom(data, scale, order=1)

class BathymetryInferenceDataset:
    """推理专用数据读取类"""
    
    def __init__(
        self,
        feature_paths: dict,
        normalization_params_path: str,
        cache_dir: str = None,
        lon_range=None,
        lat_range=None,
        patch_size=5 
    ):
        print("[推理数据集] 初始化...")
        self.patch_size = patch_size
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        with open(normalization_params_path, 'r') as f:
            norm_params = json.load(f)
        
        self.X_mean = np.array(norm_params['x_means'], dtype=np.float32)
        self.X_std = np.array(norm_params['x_stds'], dtype=np.float32)
        self.y_mean = norm_params['y_mean']
        self.y_std = norm_params['y_std']
        
        # ---------------------------------------------------------
        # 1. 尝试加载特征网格缓存
        # ---------------------------------------------------------
        cache_available = False
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_key = f"inference_cache_lon{lon_range}_lat{lat_range}_ps{patch_size}.npz"
            features_cache_path = self.cache_dir / cache_key
            
            if features_cache_path.exists():
                print(f"[推理数据集] 加载特征缓存: {features_cache_path}")
                cache_data = np.load(features_cache_path)
                self.features_grid = cache_data['features_grid']
                self.lon = cache_data['lon']
                self.lat = cache_data['lat']
                self.lon2d = cache_data['lon2d']
                self.lat2d = cache_data['lat2d']
                cache_available = True
        
        if not cache_available:
            print("[推理数据集] 读取原始特征数据...")
            feature_arrays = []
            
            for i, (name, path) in enumerate(feature_paths.items()):
                ds = xr.open_dataset(path)
                da = ds[list(ds.data_vars)[0]]
                
                if lon_range is not None: da = da.sel(lon=slice(*lon_range))
                if lat_range is not None: da = da.sel(lat=slice(*lat_range))
                
                if i == 0:
                    self.lon = da.lon.values
                    self.lat = da.lat.values
                    self.lon2d, self.lat2d = np.meshgrid(self.lon, self.lat)
                
                feature_arrays.append(da.values)
                ds.close()
            
            self.features_grid = np.stack(feature_arrays, axis=0) 
            lon_feat = self.lon2d.astype(np.float32)
            lat_feat = self.lat2d.astype(np.float32)

            self.features_grid = np.concatenate(
                [self.features_grid, lon_feat[None, ...], lat_feat[None, ...]], axis=0
            )
            
            if self.cache_dir and 'features_cache_path' in locals():
                np.savez_compressed(
                    features_cache_path, features_grid=self.features_grid,
                    lon=self.lon, lat=self.lat, lon2d=self.lon2d, lat2d=self.lat2d
                )
                
        self.C, self.H, self.W = self.features_grid.shape
        
        # ---------------------------------------------------------
        # 2. 尝试加载 Patch 张量缓存
        # ---------------------------------------------------------
        patches_cache_available = False
        if self.cache_dir:
            # 【核心修复 2】：将经纬度范围强制加入到 patches 缓存文件名中！
            patches_cache_key = f"inference_patches_lon{lon_range}_lat{lat_range}_ps{patch_size}.npz"
            patches_cache_path = self.cache_dir / patches_cache_key
            
            if patches_cache_path.exists():
                print(f"[推理数据集] 加载 Patch 缓存: {patches_cache_path}")
                cache_data = np.load(patches_cache_path)
                self.patches_tensor = torch.from_numpy(cache_data['patches_tensor'].astype(np.float32))
                self.positions = cache_data['positions']
                patches_cache_available = True
                print(f"[推理数据集] 缓存加载完成!")
        
        if not patches_cache_available:
            self._build_patch_features()
            if self.cache_dir and 'patches_cache_path' in locals():
                print(f"[推理数据集] 保存 Patch 缓存至: {patches_cache_path}")
                np.savez_compressed(
                    patches_cache_path,
                    patches_tensor=self.patches_tensor.numpy(),
                    positions=self.positions,
                )
        
        print(f"[推理数据集] 网格大小: {self.H}x{self.W} | 有效预测点: {len(self.patches_tensor)} | 输入通道: {self.C}")
    
    def _build_patch_features(self):
        print(f"[推理数据集] 正在构建 Patch (尺寸: {self.patch_size}x{self.patch_size})...")
        pad_size = self.patch_size // 2
        
        valid_mask2d = ~np.any(np.isnan(self.features_grid), axis=0)
        valid_rows, valid_cols = np.where(valid_mask2d)
        num_valid = len(valid_rows)
        
        if num_valid == 0:
            raise ValueError("错误：研究区内未找到任何有效的特征点！")
            
        print(f"    -> 扫描网格完毕，发现 {num_valid} 个有效海洋点，开始提取...")
        
        patches_array = np.empty((num_valid, self.C, self.patch_size, self.patch_size), dtype=np.float32)
        self.positions = np.empty((num_valid, 2), dtype=int)
        
        features_padded = np.pad(
            self.features_grid, 
            ((0, 0), (pad_size, pad_size), (pad_size, pad_size)), 
            mode='constant', constant_values=np.nan
        )
        
        for idx in range(num_valid):
            i, j = valid_rows[idx], valid_cols[idx]
            
            i_pad, j_pad = i + pad_size, j + pad_size
            patch = features_padded[:, i_pad-pad_size:i_pad+pad_size+1, j_pad-pad_size:j_pad+pad_size+1]
            
            if np.isnan(patch).any():
                center_vals = self.features_grid[:, i, j][:, None, None]
                patch = np.where(np.isnan(patch), center_vals, patch)
                
            patches_array[idx] = patch
            self.positions[idx] = [i, j]
            
            if (idx + 1) % 500000 == 0:
                print(f"    -> 已提取 {idx + 1} / {num_valid} ...")
                
        print("    -> 提取完成！开始进行全局 Z-score 归一化...")
        
        X_mean_exp = self.X_mean.reshape(self.C, 1, 1)
        X_std_exp = self.X_std.reshape(self.C, 1, 1)
        patches_array -= X_mean_exp
        patches_array /= X_std_exp
        
        self.patches_tensor = torch.from_numpy(patches_array)
    
    def inverse_transform(self, y_norm):
        if isinstance(y_norm, torch.Tensor):
            y_norm = y_norm.numpy()
        return y_norm * self.y_std + self.y_mean
    
    def reconstruct_grid(self, predictions):
            grid = np.full((self.H, self.W), np.nan, dtype=np.float32)
            if predictions.ndim > 1:
                predictions = predictions.squeeze()
                
            # 🚨 移除 clip 魔法，加入严格的长度校验！
            if len(predictions) != len(self.positions):
                raise ValueError(f"致命错误：预测结果数量 ({len(predictions)}) 与 坐标点位置数量 ({len(self.positions)}) 不一致！\n"
                                f"原因：读取了旧的缓存文件导致坐标错乱。请立刻删除 cache 文件夹后重新运行推理！")
                
            rows = self.positions[:, 0]
            cols = self.positions[:, 1]
            
            # 按照一一对应的关系还原地理网格
            grid[rows, cols] = predictions
            
            print(f"[重建网格] 成功填充 {len(predictions)} 个点 (网格NaN比例: {np.isnan(grid).sum() / grid.size * 100:.1f}%)")
            return grid