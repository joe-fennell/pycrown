import numpy as np

class CanopySegmentor:
    """Canopy delineation estimator
    """

    def __init__(self, algorithm='dalponteCIRC_numba', th_tree=15.,
                 th_seed=0.7, th_crown=0.55, max_crown=10.):
        """
        """
        self.algorithm = algorithm
        self.th_seed = th_seed
        self.th_tree = th_tree
        self.th_crown = th_crown
        self.max_crown = max_crown

    @property
    def algorithm(self):
        return self._algorithm_name

    @algorithm.setter
    def algorithm(self, name):
        if name == 'dalponte_numba':
            from ._crown_dalponte_numba import _crown_dalponte
            self._algorithm_name = name
            self._algorithm = _crown_dalponte

        elif name == 'dalponteCIRC_numba':
            from ._crown_dalponteCIRC_numba import _crown_dalponteCIRC
            self._algorithm_name = name
            self._algorithm = _crown_dalponteCIRC

        elif name == 'dalponte_cython':
            raise NotImplementedError('Cython not yet supported')

        else:
            raise NotImplementedError('No support for {}'.format(name))

    def predict(self, canopy_height_model, crown_centres):
        """Predict the canopy delineation for a canopy and list of centres
        """
        crowns = self._algorithm(np.array(canopy_height_model, dtype=np.float32),
                                 np.array(crown_centres, dtype=np.int32),
                                 np.float(self.th_seed),
                                 np.float(self.th_crown),
                                 np.float(self.th_tree),
                                 np.float(self.max_crown))
        return np.array(crowns, np.int32)

# def crown_delineation():
#     """ Function calling external crown delineation algorithms
#
#     Parameters
#     ----------
#     algorithm :  str
#                  crown delineation algorithm to be used, choose from:
#                  ['dalponte_cython', 'dalponte_numba',
#                   'dalponteCIRC_numba', 'watershed_skimage']
#     loc :        str, optional
#                  tree seed position: `top` or `top_cor`
#     th_seed :    float
#                  factor 1 for minimum height of tree crown
#     th_crown :   float
#                  factor 2 for minimum height of tree crown
#     th_tree :    float
#                  minimum height of tree seed (in m)
#     max_crown :  float
#                  maximum radius of tree crown (in m)
#
#     Returns
#     -------
#     ndarray
#         raster of individual tree crowns
#     """
#     timeit = 'Tree crowns delineation: {:.3f}s'
#
#     # get the tree seeds (starting points for crown delineation)
#     seeds = self._tree_colrow(loc, self.resolution)
#     inraster = kwargs.get('inraster')
#
#     if not isinstance(inraster, np.ndarray):
#         inraster = self.chm
#     else:
#         inraster = inraster
#
#     if kwargs.get('max_crown'):
#         max_crown = kwargs['max_crown'] / self.resolution
#
#     if algorithm == 'dalponte_cython':
#         tt = time.time()
#         crowns = _crown_dalponte_cython._crown_dalponte(
#             inraster, seeds,
#             th_seed=float(kwargs['th_seed']),
#             th_crown=float(kwargs['th_crown']),
#             th_tree=float(kwargs['th_tree']),
#             max_crown=float(max_crown)
#         )
#         print(timeit.format(time.time() - tt))
#
#     elif algorithm == 'dalponte_numba':
#         tt = time.time()
#         crowns = _crown_dalponte_numba._crown_dalponte(
#             inraster, seeds,
#             th_seed=float(kwargs['th_seed']),
#             th_crown=float(kwargs['th_crown']),
#             th_tree=float(kwargs['th_tree']),
#             max_crown=float(max_crown)
#         )
#         print(timeit.format(time.time() - tt))
#
#     elif algorithm == 'dalponteCIRC_numba':
#         tt = time.time()
#         crowns = _crown_dalponteCIRC_numba._crown_dalponteCIRC(
#             inraster, seeds,
#             th_seed=float(kwargs['th_seed']),
#             th_crown=float(kwargs['th_crown']),
#             th_tree=float(kwargs['th_tree']),
#             max_crown=float(max_crown)
#         )
#         print(timeit.format(time.time() - tt))
#
#     elif algorithm == 'watershed_skimage':
#         tt = time.time()
#         crowns = self._watershed(
#             inraster, th_tree=float(kwargs['th_tree'])
#         )
#         print(timeit.format(time.time() - tt))
#
#     self.crowns = np.array(crowns, dtype=np.int32)
