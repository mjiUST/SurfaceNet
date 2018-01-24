import os
import numpy as np
np.random.seed(201711)
from scipy.spatial import cKDTree
from plyfile import PlyData, PlyElement

import utils


def __extract_vxls_ijk__(kdTree, cubeCenter, cube_D, resolution, density_dtype = 'uint'):
    """
    consider the hypercube around cubeCenter (2D/3D/nD) with radius = cube_D * resolution / 2

    TODO: to support multiple cubeCenters queries

    -----------
    inputs:
        kdTree, 
        cubeCenter, 
        cube_D, 
        resolution, 
        density_dtype: determine the returned value of density
            'float': float values with max density = 1 / 0; 
            'bool': whether there are more than 1 point in the voxel
            'uint': only return the NO. of pts in the voxel

    -----------
    outputs:
        vxls_ijk: (N_vxls, 2/3/n)
        density

    -----------
    examples:
    >>> xy = np.mgrid[0:3, 0:5].reshape((2,-1))
    >>> xy = np.c_[xy, np.array([[5,7], [8,8]]).T].T    # (N_pts, 3)
    >>> kdTree = cKDTree(xy)
    >>> __extract_vxls_ijk__(kdTree, cubeCenter = np.array([1.5, 3.5]), cube_D = 3, resolution = 2, density_dtype = 'uint')
    (array([[ 0.,  0.],
            [ 0.,  1.],
            [ 1.,  0.],
            [ 1.,  1.]]), array([2, 2, 4, 4]))
    >>> __extract_vxls_ijk__(kdTree, cubeCenter = np.array([1.5, 3.5]), cube_D = 2, resolution = 2, density_dtype = 'float')
    (array([[ 0.,  0.],
            [ 0.,  1.],
            [ 1.,  0.],
            [ 1.,  1.]]), array([ 1.  ,  0.5 ,  0.5 ,  0.25]))
    >>> _, NO_pts_inVxl = __extract_vxls_ijk__(kdTree, cubeCenter = np.array([1.5, 3.5]), cube_D = 2, resolution = 2, density_dtype = 'uint')
    >>> NO_pts_inVxl
    array([4, 2, 2, 1])
    >>> __extract_vxls_ijk__(kdTree, cubeCenter = np.array([5, 0]), cube_D = 2, resolution = 2, density_dtype = 'float')
    (array([], shape=(0, 2), dtype=float32), array([], dtype=float32))
    >>> __extract_vxls_ijk__(kdTree, cubeCenter = np.array([50, 50]), cube_D = 2, resolution = 2, density_dtype = 'float')
    (array([], shape=(0, 2), dtype=float32), array([], dtype=float32))
    """

    ndim = cubeCenter.size
    cube_D_mm = cube_D * resolution
    search_radius = (ndim+1)**.5 * cube_D_mm/2   # nD case: distance from cubeCenter to cubeCorner = n**.5 * cube_D_mm/2
    vxls_inSphere = kdTree.data[kdTree.query_ball_point(cubeCenter, r = search_radius)]
    vxls_inSphere -= (cubeCenter - cube_D_mm/2)
    coords = np.floor(vxls_inSphere / resolution)  # integer
    inCube = np.sum((coords < cube_D) & (coords >= 0), axis=-1) == ndim  # (N, 3) --> (N,)
    vxls_ijk_redundant = coords[inCube]
    if vxls_ijk_redundant.shape[0] is 0:
        return np.zeros((0, ndim), dtype=np.float32), np.zeros((0, ), dtype=np.float32)
    vxls_ijk_redundant_1D = vxls_ijk_redundant.view(dtype = vxls_ijk_redundant.dtype.descr * ndim)
    _, indices, counts = np.unique(vxls_ijk_redundant_1D, return_index=True, return_counts=True)
    vxls_ijk_unique = vxls_ijk_redundant[indices]

    if density_dtype is 'float': 
        density = counts / float(counts.max())
    elif density_dtype is 'uint':
        density = counts
    elif density_dtype is 'bool':
        density = (counts > 1)

    return vxls_ijk_unique, density



def sample_pts_from_kdTree(kdTree, N_pts, distance_min, distance_max):
    """
    randomly sample (on/off-surface) pts with distance > distance_min to the nearest points formed in the kdTree
    If the pts are treated as the cube's center, distance_min = cube_D_mm / 2

    -----------
    inputs:
        kdTree
        N_pts: how many pts to be sampled
        distance_min/max: distance to the nearest neighbor should in the range [distance_min, distance_max]
                If distance_max == 0: Just randomly sample some leaves of the tree

    -----------
    outputs:
        pts: (N_pts, 3/2), xyz / xy coords depands on the 3D / 2D kdTree

    -----------
    examples:
    >>> np.random.seed(201711)
    >>> xy = np.mgrid[0:3, 0:5].reshape((2,-1))
    >>> xy = np.c_[xy, np.array([[5,7], [8,8]]).T].T    # (N_pts, 3)
    >>> tree = cKDTree(xy)
    >>> pts = sample_pts_from_kdTree(tree, N_pts = 2, distance_min = 0, distance_max = 0)
    >>> dd, ii = tree.query(pts, k=1)
    >>> dd
    array([ 0.,  0.])
    >>> pts
    array([[ 2.,  0.],
           [ 2.,  3.]])
    >>> pts = sample_pts_from_kdTree(tree, N_pts = 2, distance_min = 1, distance_max = 2)
    >>> dd, ii = tree.query(pts, k=1)
    >>> dd
    array([ 1.72343836,  1.32660501])
    """

    pts = []
    max_trial = 100
    if distance_max == 0:   # only need to randomly select leaves in the tree
        selector = np.zeros(kdTree.n, ).astype(np.bool)
        selector[np.random.randint(0, kdTree.n, N_pts)] = 1
        pts = kdTree.data[selector]
    else: 
        for _i in range(N_pts):
            # if 100 proposals cannot fall in the range, print a warning. TODO: Need to find more efficent way
            for _j in range(max_trial):    
                vrand = utils.sample_randomVector(vmin = kdTree.mins.astype(np.float32), vmax = kdTree.maxes.astype(np.float32)) # (3,)
                dd, ii = kdTree.query(vrand, k=1)
                if (dd >= distance_min) and (dd <= distance_max):
                    pts.append(vrand)
                    break   # the vector sample is OK
                if _j is (max_trial-1):
                    print("off surface distance threshold is too large: {} - {}".format(distance_min, distance_max))

    return pts


def save_surfacePts_2file(inputFile, outputFile, N_pts_onSurface, N_pts_offSurface, \
        cube_D, cube_resolutionList, inputDataType = 'pcd', density_dtype = 'float', 
        silentLog = False):
    """
    read a 3D model, record the surface pts information in the on/off-surface cubes, save to lists (like util/sparseCubes.py)

    --------
    inputs:
        inputFile: 'xx.ply' file
        outputFile: 'xx.npz' file
        N_pts_on/offSurface: how many on/off-surface cubes to be sampled
        cube_D: cube.shape = (cube_D, )*3, larger than the cube_D_4training because of data augmentation
        cube_resolutionList: sample multi-scale cubes with different resolutions
        inputDataType: 'pcd' / 'mesh'

    --------
    outputs: (wite to file)
        cube_param, (N_cubes, ) 'min_xyz' / 'cube_D' / 'resolution'
        vxl_ijk_list, N_cubes elements: each of which is numpy array (N_vxls, 3) uint32
        density_list, N_cubes elements: each of which is numpy array (N_vxls, ) float32

    """

    outputStream = ""
    cube_param_list, vxl_ijk_list, density_list = [], [], []
    model3D = PlyData.read(inputFile)
    if inputDataType is 'pcd':
        pcd = model3D
        pts_xyz = np.array([pcd['vertex'][_coord] for _coord in ['x', 'y', 'z']]).T     # (N_pts, 3)
        kdTree = cKDTree(pts_xyz)
    else:
        # mesh = model3D
        # TODO
        print("Need to be done.")

    for _resol in cube_resolutionList:
        cube_D_mm = cube_D * _resol


        ############
        # onSurface
        ############
        pts_onSurface = sample_pts_from_kdTree(kdTree, N_pts_onSurface, distance_min = 0, distance_max = 0) # (N_pts_onSurface, 3)
        for _cubeCenter in pts_onSurface:
            _cube_min = _cubeCenter - cube_D_mm / 2
            vxls_ijk, density = __extract_vxls_ijk__(kdTree, _cubeCenter, cube_D, _resol, density_dtype = density_dtype) # (N_vxls, 3), (N_vxls, )
            _cube_param = np.array([(_cube_min, _resol, cube_D)], dtype = [('min_xyz', np.float32, (3, )), ('resolution', np.float32), ('cube_D', np.uint32)])
            cube_param_list.append(_cube_param)
            vxl_ijk_list.append(vxls_ijk)
            density_list.append(density)


        ############
        # offSurface
        ############
        pts_offSurface = sample_pts_from_kdTree(kdTree, N_pts_offSurface, distance_min = cube_D_mm, distance_max = cube_D_mm * 10) # (N_pts_offSurface, 3)
        for _cubeCenter in pts_offSurface:
            _cube_min = _cubeCenter - cube_D_mm / 2
            _cube_param = np.array([(_cube_min, _resol, cube_D)], dtype = [('min_xyz', np.float32, (3, )), ('resolution', np.float32), ('cube_D', np.uint32)])
            cube_param_list.append(_cube_param)
            vxl_ijk_list.append([])
            density_list.append([])

    cube_param = np.concatenate(cube_param_list, axis=0)

    utils.mkdirs_ifNotExist(os.path.dirname(outputFile))
    with open(outputFile, 'wb') as f:
        np.savez_compressed(f, cube_param = cube_param, vxl_ijk_list = vxl_ijk_list, density_list = density_list)
    if not silentLog:
        print("Saved surface pts to file: {}".format(outputFile))

    return cube_param, vxl_ijk_list, density_list


def read_saved_surfacePts(inputFile, silentLog = False):
    """
    read saved on/off-surface pts
    Return cube_param, vxl_ijk_list, density_list
    """

    with open(inputFile, 'r') as f:
        npz = np.load(f)
        cube_param, vxl_ijk_list, density_list = npz['cube_param'], npz['vxl_ijk_list'], npz['density_list']
    if not silentLog:
        print("Loaded surface pts from file: {}".format(inputFile))
    return cube_param, vxl_ijk_list, density_list





if __name__ == '__main__':
    import doctest
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)
