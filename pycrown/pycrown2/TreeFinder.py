"""
"""

import numpy as np
import scipy.ndimage as ndimage

## NumPy level funcs

class TreeFinder:
    """Tree canopy maximum locator.

    Performs median filtering and then maximum filtering.
    """
    def __init__(self, resolution, window_size=5, min_height=20):
        """
        Args:
            resolution (float): pixel_resolution in metres
            window_size (float): moving window size in metres
            min_height (float): minimum height for a local maximum

        Returns:
            n x 2 array of tree top pixel coordinates

        """
        self.resolution = resolution
        self.window_size = window_size
        self.min_height = min_height

    @property
    def window_size(self):
        return float(self._window_size * self.resolution)

    @window_size.setter
    def window_size(self, x):
        if x % self.resolution:
            raise ValueError('resolution must be an exact divisor of window_size')
        self._window_size = int(x / self.resolution)

    def predict(self, canopy_height_model):
        """Predicts the x and y coords of every canopy
        """
        X = canopy_height_model

        X_median = ndimage.filters.median_filter(
            X,
            footprint=self._get_kernel(
                self.window_size,
                circular=False
                )
        )
        X_max = ndimage.filters.maximum_filter(
            X_median,
            footprint=self._get_kernel(
                self.window_size,
                circular=True
                )
            )
        tree_max = X == X_max

        # set trees below min height to 0
        tree_max[X <= self.min_height] = 0

        # label trees
        tree_markers, num_objects = ndimage.label(tree_max)

        # if no trees, return a 0 x 2 array
        if num_objects == 0:
            return np.array([[],[]]).T

        # if canopy height is the same for multiple pixels,
        # place the tree top in the center of mass of the pixel bounds
        yx = np.array(
                ndimage.center_of_mass(
                    X,
                    tree_markers,
                    range(1, num_objects+1)
                    ),
                dtype=np.float32) + 0.5
        return np.array((yx[:, 0], yx[:, 1])).T

    def _get_kernel(self, radius=5, circular=False):
        # returns a block or disc-shaped filter kernel with given radius
        if circular:
            y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
            return x**2 + y**2 <= radius**2
        else:
            return np.ones((int(radius), int(radius)))
