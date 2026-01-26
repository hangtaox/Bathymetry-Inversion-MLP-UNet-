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
        patch_size=1,          # ★ 1 = 纯点；>1 = 局部窗口
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
        # 2. 船测点 → SWOT 网格（与你 MLP 完全一致）
        # ======================================================
        aligned_ship_nc_path = (
            Path(aligned_ship_nc_path) if aligned_ship_nc_path else None
        )

        if aligned_ship_nc_path and aligned_ship_nc_path.exists():
            print(f"[Dataset] 读取缓存船测网格: {aligned_ship_nc_path}")
            ds = xr.open_dataset(aligned_ship_nc_path)
            ship_grid = ds["ship_depth"].values
            ds.close()
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

        r = patch_size // 2

        for i in range(self.H):
            for j in range(self.W):
                y = ship_grid[i, j]
                if np.isnan(y):
                    continue

                # ★ 边界检查
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

    # ======================================================
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    # ======================================================
    def inverse_transform_y(self, y_norm):
        return y_norm * self.y_std + self.y_mean

    # ======================================================
    def _grid_ship_points(self, ship_nc_path, lon_range, lat_range, agg_method):
        ds = xr.open_dataset(ship_nc_path)
        lon = ds["lon"].values
        lat = ds["lat"].values
        depth = ds["depth"].values
        ds.close()

        mask = np.ones_like(lon, dtype=bool)
        if lon_range:
            mask &= (lon >= lon_range[0]) & (lon <= lon_range[1])
        if lat_range:
            mask &= (lat >= lat_range[0]) & (lat <= lat_range[1])

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
        scale = (
            target_shape[0] / data.shape[0],
            target_shape[1] / data.shape[1],
        )
        return ndimage.zoom(data, scale, order=1)

class BathymetryInferenceDataset:
    """推理专用数据读取类，用于CNN模型（带patch提取）"""
    
    def __init__(
        self,
        feature_paths: dict,
        normalization_params_path: str,
        cache_dir: str = None,  # 添加缓存目录参数
        lon_range=None,
        lat_range=None,
        patch_size=5  # 添加patch_size参数，与训练保持一致
    ):
        print("[推理数据集] 初始化...")
        self.patch_size = patch_size
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        # 加载归一化参数
        with open(normalization_params_path, 'r') as f:
            norm_params = json.load(f)
        
        self.X_mean = np.array(norm_params['x_means'], dtype=np.float32)
        self.X_std = np.array(norm_params['x_stds'], dtype=np.float32)
        self.y_mean = norm_params['y_mean']
        self.y_std = norm_params['y_std']
        
        # 读取特征数据（添加缓存机制）
        feature_names = list(feature_paths.keys())
        
        # 检查是否有缓存
        cache_available = False
        features_cache_path = None
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            # 基于参数生成缓存文件名
            cache_key = f"inference_cache_lon{lon_range}_lat{lat_range}_ps{patch_size}.npz"
            features_cache_path = self.cache_dir / cache_key
            
            if features_cache_path.exists():
                print(f"[推理数据集] 加载缓存数据: {features_cache_path}")
                cache_data = np.load(features_cache_path)
                self.features_grid = cache_data['features_grid']
                self.lon = cache_data['lon']
                self.lat = cache_data['lat']
                self.lon2d = cache_data['lon2d']
                self.lat2d = cache_data['lat2d']
                cache_available = True
        
        if not cache_available:
            print("[推理数据集] 读取特征数据...")
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
            
            # 保存到缓存
            if features_cache_path:
                print(f"[推理数据集] 保存数据到缓存: {features_cache_path}")
                np.savez_compressed(
                    features_cache_path,
                    features_grid=self.features_grid,
                    lon=self.lon,
                    lat=self.lat,
                    lon2d=self.lon2d,
                    lat2d=self.lat2d
                )
        
        self.C, self.H, self.W = self.features_grid.shape
        
        # 检查patch数据是否有缓存
        patches_cache_available = False
        patches_cache_path = None
        
        if self.cache_dir:
            patches_cache_key = f"inference_patches_ps{patch_size}.npz"
            patches_cache_path = self.cache_dir / patches_cache_key
            
            if patches_cache_path.exists():
                print(f"[推理数据集] 加载patch缓存: {patches_cache_path}")
                cache_data = np.load(patches_cache_path)
                self.patches_tensor = torch.from_numpy(cache_data['patches_tensor'].astype(np.float32))
                self.positions = cache_data['positions']
                self.row_indices = cache_data['row_indices']
                self.col_indices = cache_data['col_indices']
                self.valid_mask = cache_data['valid_mask']
                self.lon_valid = cache_data['lon_valid']
                self.lat_valid = cache_data['lat_valid']
                patches_cache_available = True
                print(f"[推理数据集] 从缓存加载完成!")
        
        if not patches_cache_available:
            print("[推理数据集] 构建patch数据...")
            # 构建完整的patch数据
            self._build_patch_features()
            
            # 保存patch数据到缓存
            if self.cache_dir and patches_cache_path:
                print(f"[推理数据集] 保存patch数据到缓存: {patches_cache_path}")
                np.savez_compressed(
                    patches_cache_path,
                    patches_tensor=self.patches_tensor.numpy(),
                    positions=self.positions,
                    row_indices=self.row_indices,
                    col_indices=self.col_indices,
                    valid_mask=self.valid_mask,
                    lon_valid=self.lon_valid,
                    lat_valid=self.lat_valid
                )
        
        print(f"[推理数据集] 网格大小: {self.H}x{self.W}")
        print(f"[推理数据集] 有效海洋点: {len(self.patches_tensor)}")
        print(f"[推理数据集] 原始特征数: {self.C}")
        print(f"[推理数据集] Patch数据形状: {self.patches_tensor.shape}")
    
    def _build_patch_features(self):
        """构建包含patch的特征张量"""
        pad_size = self.patch_size // 2
        print(f"[推理数据集] Patch大小: {self.patch_size}x{self.patch_size}")
        
        # 对特征网格进行padding（使用nan填充）
        features_padded = np.pad(
            self.features_grid, 
            ((0, 0), (pad_size, pad_size), (pad_size, pad_size)), 
            mode='constant', 
            constant_values=np.nan
        )
        
        # 存储有效点的patch、位置信息和原始特征
        patches_list = []
        positions_list = []
        self.row_indices = []
        self.col_indices = []
        self.valid_mask = np.zeros((self.H, self.W), dtype=bool)
        
        print("[推理数据集] 提取patch...", end="")
        count = 0
        
        for i in range(self.H):
            for j in range(self.W):
                # 检查中心点是否有效
                if not np.any(np.isnan(self.features_grid[:, i, j])):
                    # 提取patch
                    i_pad = i + pad_size
                    j_pad = j + pad_size
                    patch = features_padded[:, 
                                          i_pad-pad_size:i_pad+pad_size+1,
                                          j_pad-pad_size:j_pad+pad_size+1]
                    
                    # 检查patch内是否有太多无效值
                    # 检查中心点是否有效
                    if not np.any(np.isnan(self.features_grid[:, i, j])):
                        # 提取patch
                        i_pad = i + pad_size
                        j_pad = j + pad_size
                        patch = features_padded[:, 
                                            i_pad-pad_size:i_pad+pad_size+1,
                                            j_pad-pad_size:j_pad+pad_size+1]
                        
                        # === 修改这里：放宽要求 ===
                        # 只要中心点有效，就接受这个patch
                        # 对于patch中的nan值，用中心点值填充
                        
                        patch_filled = patch.copy()
                        for c in range(patch.shape[0]):  # 对每个通道
                            channel_data = patch[c]
                            if np.any(np.isnan(channel_data)):
                                # 用中心点值填充nan
                                center_val = self.features_grid[c, i, j]
                                channel_data[np.isnan(channel_data)] = center_val
                                patch_filled[c] = channel_data
                        
                        patches_list.append(patch_filled)
                        positions_list.append([i, j])
                        self.row_indices.append(i)
                        self.col_indices.append(j)
                        self.valid_mask[i, j] = True
                        count += 1
            
            if i % 100 == 0 and i > 0:
                print(f" {i}/{self.H}", end="", flush=True)
        
        print(f" 完成! 共找到 {count} 个有效patch")
        
        if len(patches_list) == 0:
            raise ValueError("没有找到有效的patch数据！")
        
        # 转换为numpy数组
        patches_array = np.stack(patches_list, axis=0)  # (N, C, patch_size, patch_size)
        self.positions = np.array(positions_list)  # (N, 2)
        
        # 对patch进行归一化
        print("[推理数据集] 归一化patch数据...")
        
        # 对每个通道进行归一化
        C = self.features_grid.shape[0]
        X_mean_expanded = self.X_mean.reshape(C, 1, 1)  # (C, 1, 1)
        X_std_expanded = self.X_std.reshape(C, 1, 1)    # (C, 1, 1)
        
        # 广播归一化参数到整个patch
        patches_norm = (patches_array - X_mean_expanded) / X_std_expanded
        
        # 转换为tensor
        self.patches_tensor = torch.from_numpy(patches_norm.astype(np.float32))
        
        # 计算经纬度特征
        self.lon_valid = []
        self.lat_valid = []
        for i, j in self.positions:
            self.lon_valid.append(self.lon2d[i, j])
            self.lat_valid.append(self.lat2d[i, j])
        
        self.lon_valid = np.array(self.lon_valid)
        self.lat_valid = np.array(self.lat_valid)
    
    def inverse_transform(self, y_norm):
        """反归一化深度值"""
        if isinstance(y_norm, torch.Tensor):
            y_norm = y_norm.numpy()
        return y_norm * self.y_std + self.y_mean
    
    def reconstruct_grid(self, predictions):
        """将预测结果重构回网格"""
        grid = np.full((self.H, self.W), np.nan, dtype=np.float32)
        
        if predictions.ndim > 1:
            predictions = predictions.squeeze()  # 从 (N,1) 变为 (N,)
        
        # 确保预测结果数量与有效点数量一致
        if len(predictions) != len(self.positions):
            print(f"警告: 预测结果数量({len(predictions)})与有效点数量({len(self.positions)})不一致")
            print(f"预测数量: {len(predictions)}, 位置数量: {len(self.positions)}")
            min_len = min(len(predictions), len(self.positions))
            predictions = predictions[:min_len]
            positions = self.positions[:min_len]
        else:
            positions = self.positions
        
        print(f"[重建网格] 将 {len(predictions)} 个预测值填充到 {self.H}x{self.W} 的网格中")
        print(f"[重建网格] 第一个位置: {positions[0]}, 第一个预测值: {predictions[0]:.2f}")
        
        # 填充网格
        filled_count = 0
        for (i, j), pred in zip(positions, predictions):
            if 0 <= i < self.H and 0 <= j < self.W:
                grid[i, j] = pred
                filled_count += 1
            else:
                print(f"警告: 位置({i},{j})超出网格范围 {self.H}x{self.W}")
        
        print(f"[重建网格] 成功填充 {filled_count} 个点")
        print(f"[重建网格] 网格中NaN比例: {np.isnan(grid).sum() / grid.size * 100:.2f}%")
        
        return grid
    
    def __len__(self):
        return len(self.patches_tensor)
    
    def __getitem__(self, idx):
        """获取单个样本（用于测试）"""
        return self.patches_tensor[idx]