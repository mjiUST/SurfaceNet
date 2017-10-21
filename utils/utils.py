import numpy as np
import math
import itertools
         
## only defines some operations that could be used from multiple files
    

def generate_voxelLevelWeighted_coloredCubes(viewPair_coloredCubes, viewPair_surf_predictions, weight4viewPair):
    """
    fuse the color based on the viewPair's colored cubes, surface predictions, and weight4viewPair

    inputs
    -----
    weight4viewPair (N_cubes, N_viewPairs): relative importance of each viewPair
    viewPair_surf_predictions (N_cubes, N_viewPairs, D,D,D): relative importance of each voxel in the same cube
    viewPair_coloredCubes (N_cubes * N_viewPairs, 6, D,D,D): rgb values from the views in the same viewPair 
        randomly select one viewPair_coloredCubes (N_cubes, N_viewPairs, 3, D,D,D), otherwise the finnal colorized cube could have up/down view bias
        or simply take average

    outputs
    ------
    new_coloredCubes: (N_cubes, 3, D,D,D)

    notes
    ------
    The fusion idea is like this: 
        weight4viewPair * viewPair_surf_predictions = voxel_weight (N_cubes, N_viewPairs, D,D,D) generate relative importance of voxels in all the viewPairs
        weighted_sum(randSelect_coloredCubes, normalized_voxel_weight) = new_colored_cubes (N_cubes, 3, D,D,D)
    """
    N_cubes, N_viewPairs, _D = viewPair_surf_predictions.shape[:3]
    # (N_cubes, N_viewPairs,1,1,1) * (N_cubes, N_viewPairs, D,D,D) ==> (N_cubes, N_viewPairs, D,D,D)
    voxel_weight = weight4viewPair[...,None,None,None] * viewPair_surf_predictions
    voxel_weight /= np.sum(voxel_weight, axis=1, keepdims=True) # normalization along different view pairs

    # take average of the view0/1
    # (N_cubes, N_viewPairs, 2, 3, D,D,D) ==> (N_cubes, N_viewPairs, 3, D,D,D) 
    mean_viewPair_coloredCubes = np.mean(viewPair_coloredCubes.astype(np.float32).reshape((N_cubes, N_viewPairs, 2, 3, _D,_D,_D)), axis=2)

    # sum[(N_cubes, N_viewPairs, 1, D,D,D) * (N_cubes, N_viewPairs, 3, D,D,D), axis=1] ==>(N_cubes, 3, D,D,D)
    new_coloredCubes = np.sum(voxel_weight[:,:,None,...] * mean_viewPair_coloredCubes, axis=1)

    return new_coloredCubes.astype(np.uint8)


def gen_batch_index(N_all, batch_size):
    """
    return list of index lists, which can be used to access each batch

    ---------------
    inputs:
        N_all: # of all elements
        batch_size: # of elements in each batch
    outputs:
        batch_index_list[i] is the indexes of batch i.

    ---------------
    notes:
        Python don't have out range check, the simpliest version could be:
        for _i in range(0, N_all, batch_size):
            yield range(_i, _i + batch_size)
    ---------------
    examples:
    >>> gen_batch_index(6,3) == [[0,1,2],[3,4,5]]
    True
    >>> gen_batch_index(7,3) == [[0,1,2],[3,4,5],[6]]
    True
    >>> gen_batch_index(8,3) == [[0,1,2],[3,4,5],[6,7]]
    True
    """
    batch_index_list = []
    for _batch_start_indx in range(0, N_all, batch_size):
        _batch_end_indx = int(min(_batch_start_indx + batch_size, N_all))
        batch_index_list.append(range(_batch_start_indx, _batch_end_indx))
    return batch_index_list


def gen_non0Batch_npBool(boolIndicators, batch_size):
    """
    return list of boolSelector, which select the non-0 elements in each batch. This can be used to access each numpy batch and keep the same `ndim`, (special case is when batch size=1)

    ---------------
    inputs:
        boolIndicators: np.bool (N_all,)   each batch only select the ones with boolIndicator = True.
        batch_size: # of elements in each batch

    ---------------
    outputs:
        npSelectBool[i]: (N_all,)

    ---------------
    usages:
    >>> indicators = np.array([0,1,1,1,0,0,1,0,1,1,1], dtype=np.bool)
    >>> selectors = gen_non0Batch_npBool(indicators, 3)
    >>> indicators0 = np.array([0,1,1,1,0,0,0,0,0,0,0], dtype=np.bool)
    >>> indicators1 = np.array([0,0,0,0,0,0,1,0,1,1,0], dtype=np.bool)
    >>> indicators2 = np.array([0,0,0,0,0,0,0,0,0,0,1], dtype=np.bool)
    >>> np.allclose(selectors[0], indicators0)
    True
    >>> np.allclose(selectors[1], indicators1)
    True
    >>> np.allclose(selectors[2], indicators2)
    True
    """
    SelectBool_list = []
    cumsumIndicators = np.cumsum(boolIndicators)
    for _indexList in gen_batch_index(N_all = boolIndicators.sum(), batch_size = batch_size):
        SelectBool_list.append((cumsumIndicators >= (min(_indexList)+1) ) & \
                (cumsumIndicators <= (max(_indexList)+1) ) & \
                boolIndicators)
    return np.array(SelectBool_list)


def gen_batch_npBool(N_all, batch_size):
    """
    return list of boolSelector, which can be used to access each numpy batch and keep the same `ndim`, (special case is when batch size=1)
    memory hungery!

    ---------------
    inputs:
        N_all: # of all elements
        batch_size: # of elements in each batch

    ---------------
    outputs:
        npSelectBool[i]: (N_all,)

    ---------------
    usages:
    >>> gen_batch_npBool(6,3)
    array([[ True,  True,  True, False, False, False],
           [False, False, False,  True,  True,  True]], dtype=bool)
    >>> gen_batch_npBool(6,100)
    array([[ True,  True,  True,  True,  True,  True]], dtype=bool)
    >>> npSelectBool = gen_batch_npBool(7,3)
    >>> npSelectBool
    array([[ True,  True,  True, False, False, False, False],
           [False, False, False,  True,  True,  True, False],
           [False, False, False, False, False, False,  True]], dtype=bool)
    >>> np.arange(7*2).reshape((7,2))[npSelectBool[2]]      # note that the output shape is (1,2) rather than (2,)
    array([[12, 13]])
    """
    batch_index_list = gen_batch_index(N_all = N_all, batch_size = batch_size)
    npSelectBool = np.zeros((len(batch_index_list), N_all), dtype=np.bool)
    for _i, _batch_index in enumerate(batch_index_list):
        npSelectBool[_i][_batch_index] = True
    return npSelectBool


def yield_batch_npBool(N_all, batch_size):
    """
    return list of boolSelector, which can be used to access each numpy batch and keep the same `ndim`, (special case is when batch size=1)

    ---------------
    inputs:
        N_all: # of all elements
        batch_size: # of elements in each batch

    ---------------
    outputs:
        npSelectBool[i]: (N_all,)

    ---------------
    usages:
    >>> batches1 = []
    >>> for _batch in yield_batch_npBool(7,3):
    ...     batches1.append(_batch)
    >>> batches2 = gen_batch_npBool(7,3)
    >>> np.allclose(batches1[0], batches2[0])
    True
    >>> np.allclose(batches1[-1], batches2[-1])
    True
    """
    batch_index_list = gen_batch_index(N_all = N_all, batch_size = batch_size)
    for _i, _batch_index in enumerate(batch_index_list):
        npSelectBool = np.zeros((N_all, ), dtype=np.bool)
        npSelectBool[_batch_index] = True
        yield npSelectBool



def yield_batch_ij_npBool(ij_lists, batch_size):
    """
    in each iteration, yield a boolSelector_i/j along the first 2 axes.
    For example, featureArray has shape (3,6,8), the batch will select along the first 2 axis with total elements=3*6.
    The way to use the output: featureArray[i,j] with shape of (N_batch, 8)

    if generate boolSelectors at one time, it consumes too much memory:
    {{{python
    i_cubes, j_viewPairs = np.meshgrid(range(N_validCubes), range(N_viewPairs), indexing='ij')   # (N_validCubes, N_viewPairs) for each
    for _batch in utils.gen_batch_npBool(N_all = N_validCubes * N_viewPairs, batch_size = batchSize):      # note that bool selector: _batch.shape == (N_cubes,). 
        N_batch = _batch.sum()
        _i_cubes = i_cubes.flatten()[_batch]    # (N_cubes * N_viewPairs, ) --> (N_batch, )
        _j_viewPairs = j_viewPairs.flatten()[_batch]
    }}}

    ---------------
    inputs:
        ij_lists: indices lists along first 2 axes
        batch_size: # of elements in each batch

    ---------------
    outputs:
        i, j: (N_batch, ), index of 2 axes, N_batch <= batchSize

    ---------------
    usages:
    >>> featureArray = np.arange(3*6*8).reshape((3,6,8))
    >>> for _batch_size in [3, 5, 13, 10000]:
    ...     batch_arrays1 = []
    ...     for _i, _j in yield_batch_ij_npBool(ij_lists = (range(3),range(6)), batch_size = 5):
    ...         batch_arrays1.append(featureArray[_i, _j])
    ...     batch_arrays2 = [featureArray.reshape((3*6, 8))[batch] for batch in gen_batch_npBool(3*6, 5)]
    ...     np.allclose(batch_arrays1[0], batch_arrays2[0]) and np.allclose(batch_arrays1[-1], batch_arrays2[-1])
    True
    True
    True
    True
    """

    i = np.empty((batch_size, ), dtype=np.uint32)
    j = np.empty((batch_size, ), dtype=np.uint32)
    nBatch = 0
    for ni, _i in enumerate(ij_lists[0]):
        for nj, _j in enumerate(ij_lists[1]):
            i[nBatch] = _i
            j[nBatch] = _j
            nBatch += 1
            if (nBatch == batch_size) or ((ni == len(ij_lists[0]) - 1) and (nj == len(ij_lists[1]) - 1)):
                yield i[:nBatch], j[:nBatch]  # if nBatch < batch_size for the last iteration, need to use [:nBatch]
                nBatch = 0


def k_combination_np(iterable, k = 2):
    """
    list all the k-combination along the output rows:
    input: [2,5,8], list 2-combination to a numpy array
    output: np.array([[2,5],[2,8],[5,8]])

    ----------
    usages:
    >>> k_combination_np([2,5,8])
    array([[2, 5],
           [2, 8],
           [5, 8]])
    >>> k_combination_np([2,5,8]).dtype
    dtype('int64')
    >>> k_combination_np([2.2,5.5,8.8,9.9], k=3)
    array([[ 2.2,  5.5,  8.8],
           [ 2.2,  5.5,  9.9],
           [ 2.2,  8.8,  9.9],
           [ 5.5,  8.8,  9.9]])
    """
    combinations = []
    for _combination in itertools.combinations(iterable, k):
        combinations.append(_combination)
    return np.asarray(combinations) 


import doctest
doctest.testmod()
