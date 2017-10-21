import numpy as np
import math

import utils
import camera


def __argmaxN_viewPairs__(viewPairs, w_viewPairs, N_argmax):
    """
    inputs:
        viewPairs:  (N_viewPairs, 2) np.int 
        w_viewPairs:  (N_validCubes, N_viewPairs)
        N_argmax: argmax_N
    outputs:
        argmaxN_viewPairs: (N_validCubes, N_argmax, 2)
        argmaxN_w: (N_validCubes, N_argmax)   

    -----------
    usages:
    >>> w_viewPairs = np.array([[3,1,2], [0,-1,70]])
    >>> viewPairs = utils.k_combination_np(range(3), k = 2) # the 3 positions corresponding to viewPairs [[0,1], [0,2], [1,2]]
    >>> __argmaxN_viewPairs__(viewPairs, w_viewPairs, 1)   # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    (array([[[0, 1]],
    <BLANKLINE>
           [[1, 2]]]), array([[ 3],
                              [70]]))
    >>> __argmaxN_viewPairs__(viewPairs, w_viewPairs, 2)   # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    (array([[[1, 2],
             [0, 1]],
    <BLANKLINE>
            [[0, 1],
             [1, 2]]]), array([[ 2,  3],
                               [ 0, 70]]))
    """

    N_validCubes, N_viewPairs = w_viewPairs.shape
    indice_cube, _ = np.indices((N_validCubes, N_argmax))   # (2, N_validCubes, N_argmax)
    indice_N_max = w_viewPairs.argsort(axis=1)[:, -1*N_argmax:]    # (N_validCubes, N_viewPairs) np.float32 --> (N_validCubes, N_argmax) np.int
    argmaxN_viewPairs = np.repeat(viewPairs[None,...], N_validCubes, axis=0)[indice_cube, indice_N_max] # (N_viewPairs, 2) --> (N_validCubes, N_viewPairs, 2) --> (N_validCubes, N_argmax, 2)
    argmaxN_w = w_viewPairs[indice_cube, indice_N_max]  # (N_validCubes, N_argmax)
    return argmaxN_viewPairs, argmaxN_w


def viewPairSelection(cameraTs_np, e_viewPairs, d_viewPairs, validCubes, cubeCenters_xyz, viewPair_relativeImpt_fn, batchSize, N_viewPairs4inference, viewPairs):
    """

    ------------
    inputs:
    e_viewPairs: patches' embedding  (N_cubes, N_views, D_embedding)
    d_viewPairs
    validCubes
    viewPair_relativeImpt_fn
    batchSize
    viewPairs: (N_viewPairs, 2) np.int
    ------------
    outputs:
        w_viewPairs: 
    """

    N_cubes, N_viewPairs = d_viewPairs.shape[:2]
    N_validCubes = validCubes.sum()
    D_embedding = e_viewPairs.shape[-1]
    theta_viewPairs = camera.viewPairAngles_wrt_pts(cameraTs = cameraTs_np, pts_xyz = cubeCenters_xyz[validCubes])[..., None]  # (N_validCubes, N_viewPairs, 1)
    d_viewPairs = d_viewPairs[validCubes][..., None]  # (N_validCubes, N_viewPairs, 1)
    w_viewPairs = np.empty((N_validCubes, N_viewPairs), dtype = np.float32)

    # TODO: change fn to accept pair-by-pair inputs rather than the cube-by-cube inputs
    for _batch in utils.yield_batch_npBool(N_all = N_validCubes, batch_size = int(math.floor(float(batchSize) / N_viewPairs))):
        N_batch = _batch.sum()
        # (N_cubes, N_views, D_embedding) --> (N_validCubes, N_views, D_embedding) --> (N_batch, N_viewPairs * 2, D_embedding) --> (N_batch, N_viewPairs, 2 * D_embedding)
        _e_viewPairs = e_viewPairs[validCubes][_batch][:,viewPairs.flatten()].reshape((N_batch, N_viewPairs, 2 * D_embedding))
        N_features = 2 * D_embedding + 2
        # (N_batch, N_viewPairs, N_features),  
        features_viewPairs = np.concatenate([_e_viewPairs, d_viewPairs[_batch], theta_viewPairs[_batch]], axis=-1).astype(np.float32).reshape((N_batch*N_viewPairs, N_features))

        # TODO: check        
        w_viewPairs[_batch] = viewPair_relativeImpt_fn(features_viewPairs, n_samples_perGroup = N_viewPairs)    # (N_batch*N_viewPairs, N_features) --> (N_batch, N_viewPairs) 

    # select N_argmax viewPairs
    selected_viewPairs, selected_similNet_weight = __argmaxN_viewPairs__(viewPairs = viewPairs, w_viewPairs = w_viewPairs, N_argmax = N_viewPairs4inference)

    return selected_viewPairs, selected_similNet_weight




import doctest
doctest.testmod()
