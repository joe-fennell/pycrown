import numpy as np
import logging
import xarray
from skimage.morphology import binary_closing


class CanopySegmentor:
    """Canopy delineation estimator
    """

    def __init__(self, algorithm='dalponteCIRC', th_tree=15.,
                 th_seed=0.7, th_crown=0.55, max_crown=10., resolution=1,
                 post_process=True, fill_limit=2):
        """
        Args:
            algorithm (str): one of ['dalponteCIRC', 'dalponte',
                'dalponte_cython']
            th_tree (float): minimum height of tree seed in m
            th_crown (float): factor two for minimum height of tree crown
            th_seed (float): factor one for minimum height of tree crown
            max_crown (float): max crown radius in m
            post_process (bool): if True applies a morphological closing and
                interpolation
            fill_limit (float): approx distance in metres to fill from tree
        """
        self.resolution = resolution
        self.algorithm = algorithm
        self.th_seed = np.float(th_seed)
        self.th_tree = np.float(th_tree)
        self.th_crown = np.float(th_crown)
        self.max_crown = max_crown
        self._post_process = post_process
        self.fill_limit = fill_limit

    @property
    def fill_limit(self):
        return float(self._fill_limit) * self.resolution

    @fill_limit.setter
    def fill_limit(self, x):
        self._fill_limit = int(np.ceil(x / self.resolution))

    @property
    def max_crown(self):
        return float(self._max_crown) * self.resolution

    @max_crown.setter
    def max_crown(self, x):
        self._max_crown = np.float(x / self.resolution)

    @property
    def algorithm(self):
        return self._algorithm_name

    @algorithm.setter
    def algorithm(self, name):
        if name == 'dalponte':
            from ._crown_dalponte_numba import _crown_dalponte
            self._algorithm_name = name
            self._algorithm = _crown_dalponte

        elif name == 'dalponteCIRC':
            from ._crown_dalponteCIRC_numba import _crown_dalponteCIRC
            self._algorithm_name = name
            self._algorithm = _crown_dalponteCIRC

        elif name == 'dalponte_cython':
            try:
                import pyximport; pyximport.install(
                    setup_args={'include_dirs': np.get_include()})
            except ImportError:
                raise RuntimeError('pyximport must be installed to use cython')

            from ._crown_dalponte_cython import _crown_dalponte
            self._algorithm = _crown_dalponte
            self._algorithm_name = name
            logging.warning('Cython version may be slow!')
        else:
            raise NotImplementedError('No support for {}'.format(name))

    def post_process(self, predicted_segments):
        """Removes small holes

        Args:
            predicted_segments (array): 2D array of predicted segmentation
        """
        x = predicted_segments.copy().astype(float)
        kernel = [[0,1,1,0],
                  [1,1,1,1],
                  [1,1,1,1],
                  [0,1,1,0]]
        # kernel=None
        tree_mask = binary_closing(x)#,
                                   #footprint=kernel)
        interp_mask = x.copy()
        interp_mask[x==0] = np.nan
        interp = self._interpolate_na(interp_mask, self._fill_limit)
        return interp

    def predict(self, canopy_height_model, crown_centres):
        """Predict the canopy delineation for a canopy and list of centres
        """
        # Crown centres are output in row-col and must be in col-row
        crown_centres = np.array(crown_centres[[1,0], :], dtype=np.int32)
        crowns = self._algorithm(np.array(canopy_height_model, dtype=np.float32),
                                 crown_centres,
                                 self.th_seed,
                                 self.th_crown,
                                 self.th_tree,
                                 self._max_crown)
        out = np.array(crowns, np.int32)

        if self._post_process:
            out = self.post_process(out)

        return out

    def _interpolate_na(self, x, limit):
        # expects a 2xn array
        ar = xarray.DataArray(x)
        dims = ar.dims
        if len(dims) != 2:
            raise ValueError('2D array expected')
        ar = ar.interpolate_na(dim=dims[0],
                               method='nearest',
                               limit=limit).interpolate_na(
                                   dim=dims[1],
                                   method='nearest',
                                   limit=limit)
        return ar.values
