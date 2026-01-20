import numpy as np
import xarray as xr
import torch
from torch.utils.data import Dataset
from scipy import ndimage
from pathlib import Path


class BathymetryShipPointDataset(Dataset):
    """
    点监督 Dataset
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
        print("[Dataset] 构建 BathymetryShipPointDataset")

        self.normalize = normalize

        # ======================================================
        # 1. 读取 SWOT 栅格特征（作为参考网格）
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

        self.features_grid = np.stack(feature_arrays, axis=0)  # (C,H,W)
        self.C, self.H, self.W = self.features_grid.shape

        self.lon = ref_lon
        self.lat = ref_lat
        self.lon2d, self.lat2d = np.meshgrid(self.lon, self.lat)

        # ======================================================
        # 2. 船测 → 网格（带缓存）
        # ======================================================
        aligned_ship_nc_path = (
            Path(aligned_ship_nc_path) if aligned_ship_nc_path else None
        )

        if aligned_ship_nc_path is not None and aligned_ship_nc_path.exists():
            print(f"[Dataset] 读取已缓存船测网格: {aligned_ship_nc_path}")
            ds = xr.open_dataset(aligned_ship_nc_path)
            ship_grid = ds["ship_depth"].values
            ds.close()
        else:
            print("[Dataset] 执行船测点 → SWOT 网格对齐（首次）")
            ship_grid = self._grid_ship_points(
                ship_nc_path, lon_range, lat_range, agg_method
            )

            if aligned_ship_nc_path is not None:
                aligned_ship_nc_path.parent.mkdir(parents=True, exist_ok=True)
                xr.Dataset(
                    {
                        "ship_depth": (("lat", "lon"), ship_grid)
                    },
                    coords={
                        "lon": self.lon,
                        "lat": self.lat,
                    },
                ).to_netcdf(aligned_ship_nc_path)
                print(f"[Dataset] 已保存船测网格: {aligned_ship_nc_path}")

        # ======================================================
        # 3. 构建点监督样本
        # ======================================================
        X_list, y_list, ij_list = [], [], []

        for i in range(self.H):
            for j in range(self.W):
                y = ship_grid[i, j]
                if np.isnan(y):
                    continue

                x_feat = self.features_grid[:, i, j]
                if np.any(np.isnan(x_feat)):
                    continue

                x = np.concatenate([
                    x_feat,
                    [self.lon2d[i, j]],
                    [self.lat2d[i, j]],
                ])

                X_list.append(x)
                y_list.append(y)
                ij_list.append((i, j))

        self.X = np.asarray(X_list, np.float32)
        self.y = np.asarray(y_list, np.float32).reshape(-1, 1)
        self.ij = np.asarray(ij_list, np.int32)

        # ======================================================
        # 4. 归一化（并保存 scaler）
        # ======================================================
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

    # --------------------------------------------------
    def _grid_ship_points(self, ship_nc_path, lon_range, lat_range, agg_method):
        ds = xr.open_dataset(ship_nc_path)
        lon = ds["lon"].values
        lat = ds["lat"].values
        depth = ds["depth"].values
        ds.close()

        mask = np.ones_like(lon, dtype=bool)
        if lon_range is not None:
            mask &= (lon >= lon_range[0]) & (lon <= lon_range[1])
        if lat_range is not None:
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

    # --------------------------------------------------
    def _resample_to_target(self, data, target_shape):
        scale = (
            target_shape[0] / data.shape[0],
            target_shape[1] / data.shape[1],
        )
        return ndimage.zoom(data, scale, order=1)

    # --------------------------------------------------
    def inverse_transform_y(self, y_norm):
        return y_norm * self.y_std + self.y_mean

    # --------------------------------------------------
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class BathymetryGridDataset:
    """
    全图推理 Dataset（无标签）
    """

    def __init__(self, feature_paths, lon_range, lat_range, ref_dataset):
        feature_arrays = []

        for name in feature_paths:
            ds = xr.open_dataset(feature_paths[name])
            da = ds[list(ds.data_vars)[0]]

            if lon_range is not None:
                da = da.sel(lon=slice(*lon_range))
            if lat_range is not None:
                da = da.sel(lat=slice(*lat_range))

            feature_arrays.append(da.values)
            ds.close()

        self.features_grid = np.stack(feature_arrays, axis=0)
        self.lon = da.lon.values
        self.lat = da.lat.values
        self.lon2d, self.lat2d = np.meshgrid(self.lon, self.lat)

        # 用训练集 scaler
        self.features_grid = (
            self.features_grid - ref_dataset.X_mean[:ref_dataset.C, None, None]
        ) / ref_dataset.X_std[:ref_dataset.C, None, None]
