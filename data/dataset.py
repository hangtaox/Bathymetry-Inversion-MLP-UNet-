# data/dataset_fixed.py
import torch
from torch.utils.data import Dataset
import numpy as np
import xarray as xr

# MLP网络的数据读取（得到点集）
class BathymetryPointDataset(Dataset):
    """
    正确的数据集类：
    输入X: [重力, 梯度, 经度, 纬度] (4个特征)
    输出y: [水深] (1个目标值)
    """
    def __init__(
        self,
        grav_path=None,
        curv_path=None,
        gebco_path=None,
        lon_range=(112, 114),
        lat_range=(15, 18),
        mask=None,
        normalize=True,
        downsample_method='median'
    ):
        # 1. 加载重力异常数据
        ds_grav = xr.open_dataset(grav_path)
        grav_da = ds_grav['z'].sel(lon=slice(*lon_range), lat=slice(*lat_range))
        self.grav = grav_da.values
        self.lons = grav_da.lon.values
        self.lats = grav_da.lat.values
        ds_grav.close()
        
        # 2. 加载重力梯度数据
        ds_curv = xr.open_dataset(curv_path)
        self.curv = ds_curv['z'].sel(lon=slice(*lon_range), lat=slice(*lat_range)).values
        ds_curv.close()
        
        # 3. 加载水深数据并降采样
        ds_gebco = xr.open_dataset(gebco_path)
        # gebco是elevation
        gebco_highres = ds_gebco['elevation'].sel(lon=slice(*lon_range), lat=slice(*lat_range))

        ds_gebco.close()
        
        self.gebco = self._downsample_gebco(
            gebco_highres.values, 
            target_shape=self.grav.shape,
            method=downsample_method
        )
        
        print(f"数据形状:")
        print(f"  重力: {self.grav.shape}")
        print(f"  梯度: {self.curv.shape}")
        print(f"  水深: {self.gebco.shape}")
        
        self.H, self.W = self.grav.shape
        
        # 4. 创建经纬度网格
        lon_grid, lat_grid = np.meshgrid(self.lons, self.lats)
        
        # 5. 关键：输入特征只包含前4个，不包含水深！
        self.X = np.stack([
            self.grav.flatten(),    # 特征1: 重力异常
            self.curv.flatten(),    # 特征2: 重力梯度
            lon_grid.flatten(),     # 特征3: 经度
            lat_grid.flatten()      # 特征4: 纬度
        ], axis=1)  # 形状: (N, 4)
        
        # 6. 目标值是水深
        self.y = self.gebco.flatten()  # 形状: (N,)
        
        print(f"特征矩阵形状: {self.X.shape}")
        print(f"目标值形状: {self.y.shape}")
        
        # 7. 移除无效数据
        if mask is not None:
            self.mask = mask.flatten()
        else:
            # 只移除NaN值
            self.mask = ~np.isnan(self.y)
            nan_count = np.sum(~self.mask)
            if nan_count > 0:
                print(f"移除 {nan_count} 个NaN值")
        
        self.X = self.X[self.mask]
        self.y = self.y[self.mask]
        
        print(f"有效样本数: {len(self.y)}")
        
        # 8. 保存原始数据（用于反归一化）
        self.X_raw = self.X.copy()
        self.y_raw = self.y.copy()
        
        # 9. 归一化
        if normalize:
            self.X_mean = self.X.mean(axis=0)
            self.X_std = self.X.std(axis=0) + 1e-6
            self.y_mean = self.y.mean()
            self.y_std = self.y.std() + 1e-6
            
            print(f"归一化参数:")
            print(f"  X_mean: {self.X_mean}")
            print(f"  X_std: {self.X_std}")
            print(f"  y_mean: {self.y_mean:.2f}")
            print(f"  y_std: {self.y_std:.2f}")
            
            self.X = (self.X - self.X_mean) / self.X_std
            self.y = (self.y - self.y_mean) / self.y_std
        
        # 10. 转换为Tensor
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32).unsqueeze(1)  # (N, 1)
    
    def _downsample_gebco(self, gebco_highres, target_shape, method='mean'):
        """降采样函数"""
        
        H_high, W_high = gebco_highres.shape
        H_low, W_low = target_shape
        
        factor_h = H_high // H_low
        factor_w = W_high // W_low
        
        if H_high % H_low != 0 or W_high % W_low != 0:
            return gebco_highres[:H_low, :W_low]
        
        if method == 'mean':
            gebco_down = gebco_highres.reshape(
                H_low, factor_h, W_low, factor_w
            ).mean(axis=(1, 3))
        elif method == 'median':
            from scipy import ndimage
            scale_factor = 1 / factor_h
            gebco_down = ndimage.zoom(gebco_highres, scale_factor, order=0)
            gebco_down = gebco_down[:H_low, :W_low]
        else:
            raise ValueError(f"未知的降采样方法: {method}")
        
        return gebco_down
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
    def get_normalization_params(self):
        """获取归一化参数"""
        return {
            'X_mean': self.X_mean,
            'X_std': self.X_std,
            'y_mean': self.y_mean,
            'y_std': self.y_std
        }
    
    def inverse_transform_y(self, y_norm):
        """将归一化的水深转换回原始值"""
        return y_norm * self.y_std + self.y_mean

# UNet残差网络的训练用数据读取（得到矩阵）
class BathymetryPatchDataset(Dataset):
    """
    UNet 用 Patch Dataset
    输入:  [4, H, W]  (重力, 梯度, lon, lat)
    输出:  [1, H, W]  (水深)
    """

    def __init__(
        self,
        grav_path,
        curv_path,
        gebco_path,
        h_mlp_path,
        lon_range=(110, 114),
        lat_range=(15, 18),
        patch_size=64,
        stride=32,
        split="train",
        split_ratio=0.8,
        normalize=True,
        downsample_method="mean"
    ):
        super().__init__()

        # ------------------
        # 1. 读取重力数据（带坐标）
        # ------------------
        ds_grav = xr.open_dataset(grav_path)
        grav_da = ds_grav["z"].sel(
            lon=slice(*lon_range),
            lat=slice(*lat_range)
        )

        self.grav = grav_da.values

        # ===== 新增：经纬度网格 =====
        lon = grav_da["lon"].values
        lat = grav_da["lat"].values
        lon_grid, lat_grid = np.meshgrid(lon, lat)
        
        self.lons = lon
        self.lats = lat
        
        self.lon = lon_grid
        self.lat = lat_grid

        ds_grav.close()

        # ------------------
        # 2. 读取梯度数据
        # ------------------
        ds_curv = xr.open_dataset(curv_path)
        self.curv = ds_curv["z"].sel(
            lon=slice(*lon_range),
            lat=slice(*lat_range)
        ).values
        ds_curv.close()

        # ------------------
        # 3. 读取水深数据
        # ------------------
        ds_gebco = xr.open_dataset(gebco_path)
        gebco_highres = ds_gebco["elevation"].sel(
            lon=slice(*lon_range),
            lat=slice(*lat_range)
        )
        ds_gebco.close()

        self.gebco = self._downsample_gebco(
            gebco_highres.values,
            target_shape=self.grav.shape,
            method=downsample_method
        )

        ds_mlp = xr.open_dataset(h_mlp_path)
        self.h_base = ds_mlp["predicted_depth"].sel(
            lon=slice(*lon_range),
            lat=slice(*lat_range)
        ).values
        ds_mlp.close()
         
        self.H, self.W = self.grav.shape

        # ------------------
        # 4. 全局归一化
        # ------------------
        if normalize:
            # 重力 & 梯度
            self.grav_mean = self.grav.mean()
            self.grav_std = self.grav.std() + 1e-6
            
            self.curv_mean = self.curv.mean()
            self.curv_std = self.curv.std() + 1e-6
            
            self.h_base_mean = self.h_base.mean()
            self.h_base_std = self.h_base.std() + 1e-6
            

            self.grav = (self.grav - self.grav_mean) / self.grav_std
            self.curv = (self.curv - self.curv_mean) / self.curv_std

            self.residual = self.gebco - self.h_base 
            # 长波分量
            self.h_base = (self.h_base - self.h_base_mean) / self.h_base_std
            
            
            # 水深残差
            self.residual_mean = self.residual.mean()
            self.residual_std = self.residual.std() + 1e-6
            self.residual = (self.residual - self.residual_mean) / self.residual_std
            
            # ===== 新增：经纬度归一化 =====
            lon_min, lon_max = lon_range
            lat_min, lat_max = lat_range

            self.lon = (self.lon - lon_min) / (lon_max - lon_min)
            self.lat = (self.lat - lat_min) / (lat_max - lat_min)

        # ------------------
        # 5. Patch 索引
        # ------------------
        self.patch_size = patch_size
        self.stride = stride

        self.patch_coords = []
        for i in range(0, self.H - patch_size + 1, stride):
            for j in range(0, self.W - patch_size + 1, stride):
                self.patch_coords.append((i, j))

        if (self.H - patch_size) % stride != 0:
            i = self.H - patch_size
            for j in range(0, self.W - patch_size + 1, stride):
                self.patch_coords.append((i, j))

        if (self.W - patch_size) % stride != 0:
            j = self.W - patch_size
            for i in range(0, self.H - patch_size + 1, stride):
                self.patch_coords.append((i, j))

        self.patch_coords.append((self.H - patch_size, self.W - patch_size))

        # ------------------
        # 6. Train / Val 划分
        # ------------------
        split_idx = int(len(self.patch_coords) * split_ratio)
        if split == "train":
            self.patch_coords = self.patch_coords[:split_idx]
        elif split == "val":
            self.patch_coords = self.patch_coords[split_idx:]
        elif split == "all":
            pass
        else:
            raise ValueError(f"Unknown split: {split}")

    def __len__(self):
        return len(self.patch_coords)

    def __getitem__(self, idx):
        i, j = self.patch_coords[idx]

        grav_patch = self.grav[i:i+self.patch_size, j:j+self.patch_size]
        curv_patch = self.curv[i:i+self.patch_size, j:j+self.patch_size]
        lon_patch  = self.lon[i:i+self.patch_size, j:j+self.patch_size]
        lat_patch  = self.lat[i:i+self.patch_size, j:j+self.patch_size]

        residual_patch = self.residual[i:i+self.patch_size, j:j+self.patch_size]
        h_base_patch = self.h_base[i:i+self.patch_size, j:j+self.patch_size]
        # ===== 输入 4 通道 =====
        x = np.stack(
            [grav_patch, curv_patch, lon_patch, lat_patch, h_base_patch],
            # [grav_patch, curv_patch],
            axis=0
        )   # [4, H, W]

        delta = residual_patch
        y = delta[None, :, :]

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32)
        )


    # ------------------
    # Utils
    # ------------------
    def _downsample_gebco(self, gebco_highres, target_shape, method="mean"):
        H_high, W_high = gebco_highres.shape
        H_low, W_low = target_shape

        factor_h = H_high // H_low
        factor_w = W_high // W_low

        if method == "mean":
            return gebco_highres.reshape(
                H_low, factor_h, W_low, factor_w
            ).mean(axis=(1, 3))

        elif method == "median":
            return np.median(
                gebco_highres.reshape(
                    H_low, factor_h, W_low, factor_w
                ),
                axis=(1, 3)
            )

        else:
            raise ValueError(f"Unknown downsample method: {method}")

# UNet残差网络的推理用数据读取（得到矩阵）
class BathymetryInferencePatchDataset(Dataset):
    def __init__(
        self,
        grav_path,
        curv_path,
        h_mlp_path,
        lon_range,
        lat_range,
        patch_size=64,
        stride=32,
        normalize=True,
        stats_path=None
    ):
        self.patch_size = patch_size
        self.stride = stride
        self.normalize = normalize

        # ---------- 读取数据 ----------
        ds_grav = xr.open_dataset(grav_path)
        ds_curv = xr.open_dataset(curv_path)
        ds_mlp  = xr.open_dataset(h_mlp_path)

        grav = ds_grav['z'].sel(lon=slice(*lon_range), lat=slice(*lat_range)).values
        curv = ds_curv['z'].sel(lon=slice(*lon_range), lat=slice(*lat_range)).values
        h_mlp = ds_mlp['predicted_depth'].values

        self.lons = ds_grav['lon'].sel(lon=slice(*lon_range)).values
        self.lats = ds_grav['lat'].sel(lat=slice(*lat_range)).values

        ds_grav.close()
        ds_curv.close()
        ds_mlp.close()

        H, W = grav.shape
        self.H, self.W = H, W

        # ---------- 坐标通道 ----------
        lon_grid, lat_grid = np.meshgrid(self.lons, self.lats)
        
        lon_min, lon_max = lon_range
        lat_min, lat_max = lat_range
        
        # ---------- 归一化 ----------
        if normalize and stats_path is not None:
            stats = np.load(stats_path)
            grav = (grav - stats['grav_mean']) / stats['grav_std']
            curv = (curv - stats['curv_mean']) / stats['curv_std']
            h_mlp = (h_mlp - stats['h_base_mean']) / stats['h_base_std']
            lon_grid = (lon_grid - lon_min) / (lon_max - lon_min)
            lat_grid = (lat_grid - lat_min) / (lat_max - lat_min)
        self.inputs = np.stack(
            [grav, curv, lon_grid, lat_grid, h_mlp],
            axis=0
        )  # (5, H, W)

        # ---------- patch 坐标 ----------
        self.patch_coords = []

        # 常规 patch
        for i in range(0, H - patch_size + 1, stride):
            for j in range(0, W - patch_size + 1, stride):
                self.patch_coords.append((i, j))

        # ===== 补最下面一行 =====
        if (H - patch_size) % stride != 0:
            i = H - patch_size
            for j in range(0, W - patch_size + 1, stride):
                self.patch_coords.append((i, j))

        # ===== 补最右一列 =====
        if (W - patch_size) % stride != 0:
            j = W - patch_size
            for i in range(0, H - patch_size + 1, stride):
                self.patch_coords.append((i, j))

        # ===== 补右下角 =====
        if (H - patch_size) % stride != 0 and (W - patch_size) % stride != 0:
            self.patch_coords.append((H - patch_size, W - patch_size))

    def __len__(self):
        return len(self.patch_coords)

    def __getitem__(self, idx):
        i, j = self.patch_coords[idx]
        x = self.inputs[:, i:i+self.patch_size, j:j+self.patch_size]
        return torch.from_numpy(x).float()

# unet绝对深度网络的训练用数据读取
class BathymetryDataset(Dataset):
    """
    UNet用Patch Dataset
    输入: [5, H, W] (重力, 梯度, lon, lat, h_mlp)
    输出: [1, H, W] (水深绝对深度)
    """

    def __init__(
        self,
        grav_path,
        curv_path,
        gebco_path,
        h_mlp_path,
        lon_range=(110, 114),
        lat_range=(15, 18),
        patch_size=64,
        stride=32,
        split="train",
        split_ratio=0.8,
        normalize=True,
        stats_path=None,
        downsample_method="mean"  # 新增：下采样方法
    ):
        super().__init__()
        
        # 读取重力数据（基准网格）
        self.grav = self._read_data(grav_path, "z", lon_range, lat_range)
        self.H, self.W = self.grav.shape
        
        # 读取其他数据
        self.curv = self._read_data(curv_path, "z", lon_range, lat_range)
        self.h_mlp = self._read_data(h_mlp_path, "predicted_depth", lon_range, lat_range)
        
        # 读取GEBCO数据并下采样到与重力数据相同的分辨率
        self.depth = self._read_and_downsample_gebco(
            gebco_path, 
            "elevation", 
            lon_range, 
            lat_range, 
            target_shape=(self.H, self.W),
            method=downsample_method
        )
        
        # 验证所有数据形状一致
        self._validate_shapes()
        
        # 获取经纬度网格
        self._create_lonlat_grid(lon_range, lat_range)
        
        # 归一化
        if normalize:
            if stats_path:  # 使用预计算的统计信息
                self._load_normalize_stats(stats_path, lon_range, lat_range)
            else:  # 从当前数据计算统计信息
                self._compute_normalize_stats(lon_range, lat_range)
        
        # 生成patch坐标
        self.patch_size = patch_size
        self.patch_coords = self._generate_patch_coords(patch_size, stride)
        
        # 划分训练/验证集
        self._split_data(split, split_ratio)
    
    def __len__(self):
        return len(self.patch_coords)
    
    def __getitem__(self, idx):
        i, j = self.patch_coords[idx]
        
        # 输入: [重力, 梯度, 经度, 纬度, h_mlp]
        x = np.stack([
            self.grav[i:i+self.patch_size, j:j+self.patch_size],
            self.curv[i:i+self.patch_size, j:j+self.patch_size],
            self.lon_grid[i:i+self.patch_size, j:j+self.patch_size],
            self.lat_grid[i:i+self.patch_size, j:j+self.patch_size],
            self.h_mlp[i:i+self.patch_size, j:j+self.patch_size]
        ], axis=0)
        
        # 输出: 绝对深度
        y = self.depth[i:i+self.patch_size, j:j+self.patch_size]
        y = y[None, :, :]  # 增加通道维度
        
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32)
        )
    
    # 辅助方法
    def _read_data(self, path, var_name, lon_range, lat_range):
        """读取单个数据文件"""
        ds = xr.open_dataset(path)
        data = ds[var_name].sel(
            lon=slice(*lon_range),
            lat=slice(*lat_range)
        ).values
        ds.close()
        return data
    
    def _read_and_downsample_gebco(self, path, var_name, lon_range, lat_range, target_shape, method="mean"):
        """读取GEBCO数据并下采样到目标形状"""
        ds = xr.open_dataset(path)
        gebco_highres = ds[var_name].sel(
            lon=slice(*lon_range),
            lat=slice(*lat_range)
        ).values
        ds.close()
        
        H_high, W_high = gebco_highres.shape
        H_low, W_low = target_shape
        
        # 计算下采样因子
        factor_h = H_high // H_low
        factor_w = W_high // W_low
        
        if factor_h == 0 or factor_w == 0:
            raise ValueError(f"目标形状{target_shape}比原始形状({H_high}, {W_high})还大，无法下采样")
        
        # 检查是否能整除
        if H_high % factor_h != 0 or W_high % factor_w != 0:
            print(f"警告: GEBCO形状({H_high}, {W_high})不能整除下采样因子({factor_h}, {factor_w})")
            print("将使用调整后的因子进行下采样")
            factor_h = H_high // H_low
            factor_w = W_high // W_low
        
        # 重塑数组以进行下采样
        gebco_reshaped = gebco_highres.reshape(
            H_low, factor_h, W_low, factor_w
        )
        
        # 根据方法进行下采样
        if method == "mean":
            return gebco_reshaped.mean(axis=(1, 3))
        elif method == "median":
            return np.median(gebco_reshaped, axis=(1, 3))
        elif method == "max":
            return gebco_reshaped.max(axis=(1, 3))
        elif method == "min":
            return gebco_reshaped.min(axis=(1, 3))
        else:
            raise ValueError(f"未知的下采样方法: {method}")
    
    def _validate_shapes(self):
        """验证所有数据形状一致"""
        shapes = {
            "重力": self.grav.shape,
            "梯度": self.curv.shape,
            "水深": self.depth.shape,
            "MLP预测": self.h_mlp.shape
        }
        
        # 检查所有形状是否相同
        base_shape = self.grav.shape
        for name, shape in shapes.items():
            if shape != base_shape:
                raise ValueError(f"{name}数据形状{shape}与重力数据形状{base_shape}不匹配")
        
        print(f"所有数据形状一致: {base_shape}")
    
    def _create_lonlat_grid(self, lon_range, lat_range):
        """创建经纬度网格"""
        lon = np.linspace(lon_range[0], lon_range[1], self.W)
        lat = np.linspace(lat_range[1], lat_range[0], self.H)  # 注意纬度方向
        lon_grid, lat_grid = np.meshgrid(lon, lat)
        
        # 归一化到[0, 1]
        lon_min, lon_max = lon_range
        lat_min, lat_max = lat_range
        self.lon_grid = (lon_grid - lon_min) / (lon_max - lon_min)
        self.lat_grid = (lat_grid - lat_min) / (lat_max - lat_min)
    
    def _compute_normalize_stats(self, lon_range, lat_range):
        """计算归一化统计信息"""
        self.stats = {
            'grav_mean': self.grav.mean(),
            'grav_std': self.grav.std() + 1e-6,
            'curv_mean': self.curv.mean(),
            'curv_std': self.curv.std() + 1e-6,
            'h_mlp_mean': self.h_mlp.mean(),
            'h_mlp_std': self.h_mlp.std() + 1e-6,
            'depth_mean': self.depth.mean(),
            'depth_std': self.depth.std() + 1e-6,
        }
        
        # 应用归一化
        self.grav = (self.grav - self.stats['grav_mean']) / self.stats['grav_std']
        self.curv = (self.curv - self.stats['curv_mean']) / self.stats['curv_std']
        self.h_mlp = (self.h_mlp - self.stats['h_mlp_mean']) / self.stats['h_mlp_std']
        self.depth = (self.depth - self.stats['depth_mean']) / self.stats['depth_std']
        
        print("归一化统计信息:")
        for key, value in self.stats.items():
            if 'mean' in key:
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value:.6f}")
    
    def _load_normalize_stats(self, stats_path, lon_range, lat_range):
        """加载预计算的统计信息"""
        stats = np.load(stats_path)
        self.stats = stats
        
        # 应用归一化
        self.grav = (self.grav - stats['grav_mean']) / stats['grav_std']
        self.curv = (self.curv - stats['curv_mean']) / stats['curv_std']
        self.h_mlp = (self.h_mlp - stats['h_mlp_mean']) / stats['h_mlp_std']
        self.depth = (self.depth - stats['depth_mean']) / stats['depth_std']
    
    def _generate_patch_coords(self, patch_size, stride):
        """生成patch坐标"""
        coords = []
        H, W = self.H, self.W
        
        # 常规patches
        for i in range(0, H - patch_size + 1, stride):
            for j in range(0, W - patch_size + 1, stride):
                coords.append((i, j))
        
        # 边界补丁
        if (H - patch_size) % stride != 0:
            i = H - patch_size
            for j in range(0, W - patch_size + 1, stride):
                coords.append((i, j))
        
        if (W - patch_size) % stride != 0:
            j = W - patch_size
            for i in range(0, H - patch_size + 1, stride):
                coords.append((i, j))
        
        # 右下角
        if (H - patch_size) % stride != 0 and (W - patch_size) % stride != 0:
            coords.append((H - patch_size, W - patch_size))
        
        print(f"生成 {len(coords)} 个patch坐标")
        return coords
    
    def _split_data(self, split, split_ratio):
        """划分训练/验证集"""
        if split == "all":
            return
        
        split_idx = int(len(self.patch_coords) * split_ratio)
        if split == "train":
            self.patch_coords = self.patch_coords[:split_idx]
        elif split == "val":
            self.patch_coords = self.patch_coords[split_idx:]
        else:
            raise ValueError(f"Unknown split: {split}")
        
        print(f"{split}集样本数: {len(self.patch_coords)}")
    
    def save_stats(self, path):
        """保存归一化统计信息"""
        if hasattr(self, 'stats'):
            np.savez(path, **self.stats)
            print(f"归一化统计信息已保存到: {path}")
# unet绝对深度网络的推理用数据读取
class BathymetryInferenceDataset(Dataset):
    """推理用数据读取类"""
    
    def __init__(
        self,
        grav_path,
        curv_path,
        h_mlp_path,
        lon_range,
        lat_range,
        patch_size=64,
        stride=32,
        stats_path=None
    ):
        super().__init__()
        
        # 读取数据
        self.grav = self._read_data(grav_path, "z", lon_range, lat_range)
        self.curv = self._read_data(curv_path, "z", lon_range, lat_range)
        self.h_mlp = self._read_data(h_mlp_path, "predicted_depth", lon_range, lat_range)
        
        # 获取形状和经纬度
        self.H, self.W = self.grav.shape
        self.patch_size = patch_size
        
        # 创建经纬度网格
        self._create_lonlat_grid(lon_range, lat_range)
        
        # 归一化
        if stats_path:
            self._apply_normalization(stats_path, lon_range, lat_range)
        
        # 堆叠输入
        self.inputs = np.stack([
            self.grav, self.curv, self.lon_grid, self.lat_grid, self.h_mlp
        ], axis=0)  # [5, H, W]
        
        # 生成patch坐标
        self.patch_coords = self._generate_patch_coords(patch_size, stride)
        
        # 获取经纬度数组（用于保存结果）
        self._get_lonlat_arrays(lon_range, lat_range)
    
    def __len__(self):
        return len(self.patch_coords)
    
    def __getitem__(self, idx):
        i, j = self.patch_coords[idx]
        patch = self.inputs[:, i:i+self.patch_size, j:j+self.patch_size]
        return torch.tensor(patch, dtype=torch.float32)
    
    # 辅助方法
    def _read_data(self, path, var_name, lon_range, lat_range):
        ds = xr.open_dataset(path)
        data = ds[var_name].sel(
            lon=slice(*lon_range),
            lat=slice(*lat_range)
        ).values
        ds.close()
        return data
    
    def _create_lonlat_grid(self, lon_range, lat_range):
        """创建归一化的经纬度网格"""
        lon = np.linspace(lon_range[0], lon_range[1], self.W)
        lat = np.linspace(lat_range[1], lat_range[0], self.H)  # 注意纬度方向
        lon_grid, lat_grid = np.meshgrid(lon, lat)
        
        lon_min, lon_max = lon_range
        lat_min, lat_max = lat_range
        self.lon_grid = (lon_grid - lon_min) / (lon_max - lon_min)
        self.lat_grid = (lat_grid - lat_min) / (lat_max - lat_min)
    
    def _get_lonlat_arrays(self, lon_range, lat_range):
        """获取一维经纬度数组（用于保存结果）"""
        self.lons = np.linspace(lon_range[0], lon_range[1], self.W)
        self.lats = np.linspace(lat_range[1], lat_range[0], self.H)  # 注意纬度方向
    
    def _apply_normalization(self, stats_path, lon_range, lat_range):
        stats = np.load(stats_path)
        self.grav = (self.grav - stats['grav_mean']) / stats['grav_std']
        self.curv = (self.curv - stats['curv_mean']) / stats['curv_std']
        self.h_mlp = (self.h_mlp - stats['h_mlp_mean']) / stats['h_mlp_std']
    
    def _generate_patch_coords(self, patch_size, stride):
        coords = []
        H, W = self.H, self.W
        
        for i in range(0, H - patch_size + 1, stride):
            for j in range(0, W - patch_size + 1, stride):
                coords.append((i, j))
        
        if (H - patch_size) % stride != 0:
            i = H - patch_size
            for j in range(0, W - patch_size + 1, stride):
                coords.append((i, j))
        
        if (W - patch_size) % stride != 0:
            j = W - patch_size
            for i in range(0, H - patch_size + 1, stride):
                coords.append((i, j))
        
        if (H - patch_size) % stride != 0 and (W - patch_size) % stride != 0:
            coords.append((H - patch_size, W - patch_size))
        
        print(f"推理数据集生成 {len(coords)} 个patch坐标")
        return coords