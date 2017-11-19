import numpy as np
import scipy.ndimage as ndim
import scipy.ndimage.measurements as measure

dtype_clusterLabel = np.uint32

def __cluster_inCube__(vxl_ijk_list, vxl_mask_list=[], neighbor_dist = 1):
    """
    in order to eliminate the floating noise in each voxel cube.
    Clustering in each cube, 

    -------
    inputs:
        vxl_ijk_list: [(N_pts, 3), ...] ijk of point set
        vxl_mask_list: used to filter voxels
        neighbor_dist: only consider the vxls within this range as neighborhood

    -------
    outputs:
        vxl_labeles_list: [(N_pts, ), ...]. List of pts labels: {0, 1, ...} 0 means background.
        N_labels_list: NO. of clusters in each cube. 

    -------
    examples:
    >>> vxl_ijk_list = [np.array([[1,0,0], [2,2,2], [3,3,3], [1,0,1], [2,3,3], [0,3,3], [1,2,2]]), \
                        np.array([[0,2,3], [0,1,0], [0,0,0], [0,3,3]]), \
                        np.array([[0,2,3], [0,1,0], [0,2,3]]), \
                        np.array([[0,2,3], [0,1,3], [0,0,0], [0,3,3], [3,3,3]], dtype=np.uint8) ]
    >>> vxl_mask_list = [np.array([1,0,1,1,1,1,1], dtype=np.bool), \
                         np.array([1,1,0,1], dtype=np.bool), \
                         np.array([0,0,0], dtype=np.bool), \
                         np.array([1,1,1,1,1], dtype=np.bool)]
    >>> __cluster_inCube__(vxl_ijk_list, vxl_mask_list)     # the masked index should not appear
    ([array([2, 0, 4, 2, 4, 1, 3], dtype=uint32), array([2, 1, 0, 2], dtype=uint32), array([ 0.,  0.,  0.]), array([2, 2, 1, 2, 3], dtype=uint32)], [4, 2, 0, 3])
    >>> __cluster_inCube__(vxl_ijk_list, vxl_mask_list, neighbor_dist = 3)     # consider all the neighbors
    ([array([2, 0, 1, 2, 1, 1, 1], dtype=uint32), array([2, 1, 0, 2], dtype=uint32), array([ 0.,  0.,  0.]), array([2, 2, 1, 2, 3], dtype=uint32)], [2, 2, 0, 3])
    """

    vxl_labeles_list, N_labels_list = [], []
    for _cube, _select in enumerate(vxl_mask_list):
        N_pts = _select.sum()
        vxl_ijk = vxl_ijk_list[_cube]
        if N_pts == 0:
            vxl_labeles_list.append(np.zeros(vxl_ijk.shape[:1]))
            N_labels_list.append(0)
            continue

        vxl_ijk_masked = vxl_ijk[_select]  # (N_pts, 3)

        matrix3D = np.zeros(vxl_ijk_masked.max(axis=0) + 1, dtype=np.bool) # construct dense 3D array
        matrix3D[vxl_ijk_masked[:,0], vxl_ijk_masked[:,1], vxl_ijk_masked[:,2]] = 1
        binary_structure = ndim.generate_binary_structure(3, neighbor_dist)
        labeled_3Darray, N_labels = measure.label(matrix3D, binary_structure) # matrix3D.shape, consider all the neighbors.
        _vxl_labeles = np.zeros(_select.shape, dtype=dtype_clusterLabel)
        _vxl_labeles[_select] = labeled_3Darray[vxl_ijk_masked[:,0], vxl_ijk_masked[:,1], vxl_ijk_masked[:,2]].astype(dtype_clusterLabel)
        vxl_labeles_list.append(_vxl_labeles)
        N_labels_list.append(N_labels)

    return vxl_labeles_list, N_labels_list




def __mark_overlappingLabels__(cube_ijk_np, vxl_ijk_list, vxl_mask_list, D_cube, neighbor_dist = 1):
    """
    mark the overlapping voxels in each cube.
    Note the calculation efficiency: for a cube pair, only need to calculate once.

    -------
    inputs:
        cube_ijk_np: ijk of cubes
        vxl_ijk_list: [(N_pts, 3), ...] ijk of point set
        probThresh_list: 
        vxl_mask_list: used to filter voxels
        D_cube: size of cube = (D_cube, )*3

    -------
    outputs:
        overlappingLabels_list: mark the overlapping status of ALL the voxels. Should have the same shape with vxl_mask_list.
        vxl_labeles_list: the clusters' label for each vxl

    -------
    examples:
    >>> cube_ijk_np = np.array([[1,6,8], [2,6,8], [2,7,8], [2,5,8]], dtype=np.uint8)
    >>> vxl_ijk_list = [np.array([[1,0,0], [2,2,2], [3,2,3], [3,3,3], [1,0,1], [2,3,3], [3,0,3]], dtype=np.uint8), \
                        np.array([[0,2,3], [0,1,3], [0,0,0], [0,3,3], [1,0,3], [3,3,0]], dtype=np.uint8), \
                        np.array([[0,2,3], [0,1,3], [0,0,0], [0,3,3]], dtype=np.uint8), \
                        np.array([[0,2,3], [0,1,3], [0,0,0], [0,3,3], [3,3,3]], dtype=np.uint8) ]
    >>> vxl_mask_list = [np.array([1,0,0,1,1,1,1], dtype=np.bool), \
                         np.array([1,1,0,1,1,1], dtype=np.bool), \
                         np.array([0,0,0,0], dtype=np.bool), \
                         np.array([1,1,1,1,1], dtype=np.bool)]
    >>> __mark_overlappingLabels__(cube_ijk_np, vxl_ijk_list, vxl_mask_list, D_cube = 4)
    ([[2L, 3L], [1L, 2L], [], [2L]], [array([1, 0, 0, 2, 1, 2, 3], dtype=uint32), array([1, 1, 0, 1, 2, 3], dtype=uint32), array([ 0.,  0.,  0.,  0.]), array([2, 2, 1, 2, 3], dtype=uint32)])
    >>> # vxl_overlappingMask_list looks like: [array([False, False, False, False,  True], dtype=bool), array([False,  True, False,  True, False], dtype=bool), array([False, False, False, False], dtype=bool), array([False, False, False,  True, False], dtype=bool)]
    """

    vxl_labeles_list, N_labels_list = __cluster_inCube__(vxl_ijk_list, vxl_mask_list=vxl_mask_list, neighbor_dist = neighbor_dist)
    N_cubes = len(N_labels_list)
    overlappingLabels_list = [[] for _ in range(N_cubes)]
    cube_ijk2index = {}
    # TODO: record the processed neigh_shifts for each cube to avoid redundant computation.
    # neigh_shifts = np.asarray([[1,0,0],[0,1,0],[0,0,1],[-1,0,0],[0,-1,0],[0,0,-1]]).astype(np.int8)
    neigh_shifts = np.delete(np.array(np.indices((3,3,3))).reshape((3,-1)).T - 1, 3**3/2, axis=0) # consider all the overlapping cubes as neighborhood. Remove the 0-shift neighbor (itself).

    # cube_ijk2index = {tuple(_ijk): _n for _n, _ijk in enumerate(cube_ijk_np)}
    for _n, _ijk in enumerate(cube_ijk_np):     # construct the ijk dictionary
        if vxl_mask_list[_n].sum() > 0:     # ignore the obviously empty cubes.
            cube_ijk2index.update({tuple(_ijk): _n})

    for _n, _ijk in enumerate(cube_ijk_np):
        if not cube_ijk2index.has_key(tuple(_ijk)):
            continue # this cube is filtered in the very beginning
        i_current = cube_ijk2index[tuple(_ijk)]
        _vxl_mask_current = vxl_mask_list[i_current]
        vxl_ijk_current = vxl_ijk_list[i_current][_vxl_mask_current].astype(np.int64)  # (N_unmasked_pts_current, 3)
        view1d_current = vxl_ijk_current.view(dtype = vxl_ijk_current.dtype.descr * 3)  # (N_unmasked_pts_current, 1)

        for _ijk_shift in neigh_shifts:
            ijk_neigh = _ijk + _ijk_shift
            if not cube_ijk2index.has_key(tuple(ijk_neigh)):     
                continue
            else:   # exist the overlapping cube
                i_neigh = cube_ijk2index[tuple(ijk_neigh)]
                _vxl_mask_neigh = vxl_mask_list[i_neigh]
                vxl_ijk_neigh = vxl_ijk_list[i_neigh][_vxl_mask_neigh].astype(np.int64)   # even consider the masked vxls to keep the order. 
                vxl_ijk_newCoords_neigh = (vxl_ijk_neigh + (D_cube / 2) * _ijk_shift).astype(np.int64)   # (N_vxls, 3)
                view1d_neigh = vxl_ijk_newCoords_neigh.view(dtype = vxl_ijk_newCoords_neigh.dtype.descr * 3)

                view1d_intersect = np.intersect1d(view1d_current, view1d_neigh)
                if view1d_intersect.size != 0:
                    intersectBool_current = np.in1d(view1d_current, view1d_intersect)   # (N_pts_current, ) bool
                    intersectBool_neigh = np.in1d(view1d_neigh, view1d_intersect)   # (N_pts_neigh, ) bool
                    overlappingLabels_current = vxl_labeles_list[i_current][_vxl_mask_current][intersectBool_current]
                    overlappingLabels_neigh = vxl_labeles_list[i_neigh][_vxl_mask_neigh][intersectBool_neigh]
                    overlappingLabels_list[i_current] = list(set(overlappingLabels_list[i_current] + np.ndarray.tolist(overlappingLabels_current)))
                    overlappingLabels_list[i_neigh] = list(set(overlappingLabels_list[i_neigh] + np.ndarray.tolist(overlappingLabels_neigh)))


    return overlappingLabels_list, vxl_labeles_list




def denoise_crossCubes(cube_ijk_np, vxl_ijk_list, vxl_mask_list, D_cube):
    """
    Remove the floating noise due to the limited receptive field of ConvNet.
    Clustering the voxels in each cube, delete the non-overlapping ones.

    -------
    inputs:
        cube_ijk_np: ijk of cubes
        vxl_ijk_list: [(N_pts, 3), ...] ijk of point set
        probThresh_list: 
        vxl_mask_list: used to filter voxels
        D_cube: size of cube = (D_cube, )*3

    -------
    outputs:
        vxl_overlappingMask_list: mark the overlapping status of ALL the voxels. Should have the same shape with vxl_mask_list.

    -------
    examples:
    >>> cube_ijk_np = np.array([[1,6,8], [2,6,8], [2,7,8], [2,5,8]], dtype=np.uint8)
    >>> vxl_ijk_list = [np.array([[1,0,0], [2,2,2], [3,3,3], [1,0,1], [2,3,3]], dtype=np.uint8), \
                        np.array([[0,2,3], [0,1,3], [0,0,0], [0,3,3], [3,3,0]], dtype=np.uint8), \
                        np.array([[0,2,3], [0,1,3], [0,0,0], [0,3,3]], dtype=np.uint8), \
                        np.array([[0,2,3], [0,1,3], [0,0,0], [0,3,3], [3,3,3]], dtype=np.uint8) ]
    >>> vxl_mask_list = [np.array([1,0,1,1,1], dtype=np.bool), \
                         np.array([1,1,0,1,1], dtype=np.bool), \
                         np.array([0,0,0,0], dtype=np.bool), \
                         np.array([1,1,1,1,1], dtype=np.bool)]
    >>> denoise_crossCubes(cube_ijk_np, vxl_ijk_list, vxl_mask_list, D_cube = 4)
    [array([False, False,  True, False,  True], dtype=bool), array([ True,  True, False,  True, False], dtype=bool), array([False, False, False, False], dtype=bool), array([ True,  True, False,  True, False], dtype=bool)]
    """

    vxl_maskDenoise_list = []
    overlappingLabels_list, vxl_labeles_list = __mark_overlappingLabels__(cube_ijk_np, vxl_ijk_list, vxl_mask_list, D_cube, neighbor_dist = 3)

    for _cube, _vxl_labels in enumerate(vxl_labeles_list):
        _overlappingLabels = overlappingLabels_list[_cube]
        vxl_maskDenoise_list.append(np.in1d(_vxl_labels, _overlappingLabels))    # only keep the vxls with overlapping cluster labels

    return vxl_maskDenoise_list







import doctest
doctest.testmod()
