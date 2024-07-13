import numpy as np
from tqdm import tqdm
from scipy.interpolate import griddata
from scipy.spatial import KDTree


class GriddataPartial:
    def __init__(self, ground: np.ndarray, method: str = 'linear') -> None:
        self.ground = ground
        self.method = method

    def __call__(self, xi: np.ndarray) -> np.ndarray:
        return griddata(points=self.ground[:, :2], values=self.ground[:, 2], xi=xi, method=self.method)


class HeightVariation:
    def __init__(self, point_data_normalized_height: np.ndarray, kdtree: KDTree, r: float) -> None:
        self.point_data = point_data_normalized_height
        self.kdtree = kdtree
        self.r = r

    def __call__(self, index_range: tuple[int, int]) -> np.ndarray:
        start, end = index_range
        point_data_xy = self.point_data[start:end][:, :2]
        hv = np.zeros(len(point_data_xy))
        for i, xy in enumerate(point_data_xy):
            indices = self.kdtree.query_ball_point(xy, r=self.r, p=2.0)
            if len(indices) == 1:
                continue
            points_in_disk_z = self.point_data[indices][:, 2]
            hv[i] = points_in_disk_z.max() - points_in_disk_z.min()
        return hv


class NormalVariation:
    def __init__(
        self, point_data_xy: np.ndarray, normals: np.ndarray, kdtree: KDTree, r: float, use_tqdm: bool = False
    ) -> None:
        self.point_data_xy = point_data_xy
        self.kdtree = kdtree
        self.normals = normals
        self.r = r
        self.use_tqdm = use_tqdm

    def __call__(self, index_range: tuple[int, int]) -> np.ndarray:
        start, end = index_range
        point_data_xy = self.point_data_xy[start:end]
        nv = np.empty(len(point_data_xy))
        for i, p in tqdm(enumerate(point_data_xy), total=len(point_data_xy), disable=not self.use_tqdm):
            indices = self.kdtree.query_ball_point(p, r=self.r, p=2.0)
            normal_dot_products = self.normals[indices] @ self.normals[start + i]
            nv[i] = -normal_dot_products.mean()
        return nv


def hag_pipeline(input_path, output_path):
    hag_json = f'''{{
    "pipeline": [
        "{input_path}",
        {{
        "type": "filters.hag_delaunay"
        }},
        "{output_path}"
    ]
}}'''
    return hag_json
