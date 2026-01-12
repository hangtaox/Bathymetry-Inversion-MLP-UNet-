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

# UNet网络的数据读取（得到矩阵）
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
        self.split = split
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

            self.grav = (self.grav - self.grav_mean) / self.grav_std
            self.curv = (self.curv - self.curv_mean) / self.curv_std

            # 水深
            self.gebco_mean = self.gebco.mean()
            self.gebco_std = self.gebco.std() + 1e-6
            self.gebco = (self.gebco - self.gebco_mean) / self.gebco_std

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
        
        # 获取原始patch
        grav_patch = self.grav[i:i+self.patch_size, j:j+self.patch_size]
        curv_patch = self.curv[i:i+self.patch_size, j:j+self.patch_size]
        lon_patch = self.lon[i:i+self.patch_size, j:j+self.patch_size]
        lat_patch = self.lat[i:i+self.patch_size, j:j+self.patch_size]
        gebco_patch = self.gebco[i:i+self.patch_size, j:j+self.patch_size]
        
        # 只在训练时进行增强
        if self.split == "train":
            # 随机水平翻转（50%概率）
            if np.random.rand() > 0.5:
                grav_patch = np.fliplr(grav_patch).copy()
                curv_patch = np.fliplr(curv_patch).copy()
                lon_patch = np.fliplr(lon_patch).copy()
                lat_patch = np.fliplr(lat_patch).copy()
                gebco_patch = np.fliplr(gebco_patch).copy()
            
            # 随机垂直翻转（50%概率）
            if np.random.rand() > 0.5:
                grav_patch = np.flipud(grav_patch).copy()
                curv_patch = np.flipud(curv_patch).copy()
                lon_patch = np.flipud(lon_patch).copy()
                lat_patch = np.flipud(lat_patch).copy()
                gebco_patch = np.flipud(gebco_patch).copy()
            
            # 随机90度旋转（0, 90, 180, 270度）
            k = np.random.randint(0, 4)  # 0,1,2,3
            if k > 0:
                grav_patch = np.rot90(grav_patch, k).copy()
                curv_patch = np.rot90(curv_patch, k).copy()
                lon_patch = np.rot90(lon_patch, k).copy()
                lat_patch = np.rot90(lat_patch, k).copy()
                gebco_patch = np.rot90(gebco_patch, k).copy()
        
        # 堆叠输入通道
        x = np.stack([grav_patch, curv_patch, lon_patch, lat_patch], axis=0)
        y = gebco_patch[None, :, :]
        
        # 关键修复：确保返回元组 (x, y)
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