import numpy as np
from tqdm import tqdm
from scipy.interpolate import griddata
from scipy.spatial import KDTree


class GriddataPartial:
    def __init__(self, ground: np.ndarray, method: str = 'linear') -> None:
        '''
        This class is a wrapper around `scipy.interpolate.griddata` to facilitate
        parallelization with multiprocessing.Pool for computing ground elevation of
        non-ground points.

        Parameters
        ----------
        ground : np.ndarray
            The ground points in the point cloud.
        method : str, optional
            The method to pass to `griddata`. One of 'linear', 'cubic', or 'nearest'.
            Default is 'linear'.
        '''

        self.ground = ground
        self.method = method

    def __call__(self, xi: np.ndarray) -> np.ndarray:
        '''
        Interpolate the z coordinate of the ground values at the points in `xi`.

        Parameters
        ----------
        xi : np.ndarray
            The points on which to evaluate the interpolation. I.e. the non-ground points.

        Returns
        -------
        np.ndarray
            The array of interpolated values.
        '''

        return griddata(points=self.ground[:, :2], values=self.ground[:, 2], xi=xi, method=self.method)


class HeightVariation:
    def __init__(self, point_data_normalized_height: np.ndarray, kdtree: KDTree, r: float) -> None:
        '''
        This class computes the height variation of point data and facilitates
        parallelization with multiprocessing.Pool. Height variation is the absolute difference
        between min and max values of normalized height within a disk of radius r.

        Parameters
        ----------
        point_data_normalized_height : np.ndarray
            The point cloud after height normalization.
        kdtree : scipy.spatial.KDTree
            A K-d tree build from the x- and y-coordinates of the point cloud data.
            This facilitates fast ball point queries.
        r : float
            The radius of the ball point queries.
        '''

        self.point_data = point_data_normalized_height
        self.kdtree = kdtree
        self.r = r

    def __call__(self, index_range: tuple[int, int]) -> np.ndarray:
        '''
        Compute the height variation of the point data between indices i and j
        for `index_range = (i, j)`.

        Parameters
        ----------
        index_range : tuple[int, int]
            The indices defining the range of points in `self.point_data` on which to
            compute height variation.

        Returns
        -------
        np.ndarray
            An array of the height variations of points between indices i and j.
        '''

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
        '''
        This class computes the normal variation of point data and facilitates
        parallelization with multiprocessing.Pool. Normal variation is the negative
        of the average dot product of each normal with other normals within a disk of
        radius r. This value gives a measure of planarity near each point.

        Parameters
        ----------
        point_data_xy : np.ndarray
            The- x and y-coordinates of the point clouddata.
        normals : np.ndarray
            An array of the (estimated) surface normal vectors at each point.
        kdtree : scipy.spatial.KDTree
            A K-d tree build from the x- and y-coordinates of the point cloud data.
            This facilitates fast ball point queries.
        r : float
            The radius of the ball point queries.
        use_tqdm : bool, optional
            Whether or not to print progress using tqdm. Default is False.
        '''

        self.point_data_xy = point_data_xy
        self.kdtree = kdtree
        self.normals = normals
        self.r = r
        self.use_tqdm = use_tqdm

    def __call__(self, index_range: tuple[int, int]) -> np.ndarray:
        '''
        Compute the normal variation of the point data between indices i and j
        for `index_range = (i, j)`.

        Parameters
        ----------
        index_range : tuple[int, int]
            The indices defining the range of points in `self.point_data` on which to
            compute normal variation.

        Returns
        -------
        np.ndarray
            An array of the noraml variations of points between indices i and j.
        '''

        start, end = index_range
        point_data_xy = self.point_data_xy[start:end]
        nv = np.empty(len(point_data_xy))
        for i, p in tqdm(enumerate(point_data_xy), total=len(point_data_xy), disable=not self.use_tqdm):
            indices = self.kdtree.query_ball_point(p, r=self.r, p=2.0)
            normal_dot_products = self.normals[indices] @ self.normals[start + i]
            nv[i] = -normal_dot_products.mean()
        return nv


def hag_pipeline(input_path, output_path):
    '''
    Defines the height above ground (hag) pipeline for height normalization with PDAl.
    '''

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
