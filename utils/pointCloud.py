import numpy as np
from scipy.spatial import cKDTree
from plyfile import PlyData, PlyElement

import utils

np.random.seed(201711)

def __extract_vxls_inCube__(kdTree, cubeCenter, cube_D, resolution, normalizedDensity = True):
    """
    consider the hypercube around cubeCenter (2D/3D/nD) with radius = cube_D * resolution / 2

    -----------
    inputs:
        kdTree, 
        cubeCenter, 
        cube_D, 
        resolution, 
        normalizedDensity: True: float values with max density = 1 / 0; False: bool value

    -----------
    outputs:
        vxls_inCube: (N_vxls, 2/3/n)
        density: NO. of pts in each vxl. normalizedDensity normalize to [0, 1] range.

    -----------
    examples:
    >>> xy = np.mgrid[0:3, 0:5].reshape((2,-1))
    >>> xy = np.c_[xy, np.array([[5,7], [8,8]]).T].T    # (N_pts, 3)
    >>> kdTree = cKDTree(xy)
    >>> __extract_vxls_inCube__(kdTree, cubeCenter = np.array([1.5, 3.5]), cube_D = 3, resolution = 2, normalizedDensity = False)
    (array([[ 0.,  0.],
            [ 0.,  1.],
            [ 1.,  0.],
            [ 1.,  1.]]), array([2, 2, 4, 4]))
    >>> __extract_vxls_inCube__(kdTree, cubeCenter = np.array([1.5, 3.5]), cube_D = 2, resolution = 2, normalizedDensity = True)
    (array([[ 0.,  0.],
            [ 0.,  1.],
            [ 1.,  0.],
            [ 1.,  1.]]), array([ 1.  ,  0.5 ,  0.5 ,  0.25]))
    >>> _, NO_pts_inVxl = __extract_vxls_inCube__(kdTree, cubeCenter = np.array([1.5, 3.5]), cube_D = 2, resolution = 2, normalizedDensity = False)
    >>> NO_pts_inVxl
    array([4, 2, 2, 1])
    >>> __extract_vxls_inCube__(kdTree, cubeCenter = np.array([5, 0]), cube_D = 2, resolution = 2, normalizedDensity = True)
    (array([], shape=(0, 2), dtype=float32), array([], dtype=float32))
    >>> __extract_vxls_inCube__(kdTree, cubeCenter = np.array([50, 50]), cube_D = 2, resolution = 2, normalizedDensity = True)
    (array([], shape=(0, 2), dtype=float32), array([], dtype=float32))
    """

    ndim = cubeCenter.size
    cube_D_mm = cube_D * resolution
    search_radius = (ndim+1)**.5 * cube_D_mm/2   # nD case: distance from cubeCenter to cubeCorner = n**.5 * cube_D_mm/2
    vxls_inSphere = kdTree.data[kdTree.query_ball_point(cubeCenter, r = search_radius)]
    vxls_inSphere -= (cubeCenter - cube_D_mm/2)
    coords = np.floor(vxls_inSphere / resolution)  # integer
    inCube = np.sum((coords < cube_D) & (coords >= 0), axis=-1) == ndim  # (N, 3) --> (N,)
    vxls_inCube_redundant = coords[inCube]
    if vxls_inCube_redundant.shape[0] is 0:
        return np.zeros((0, ndim), dtype=np.float32), np.zeros((0, ), dtype=np.float32)
    vxls_inCube_redundant_1D = vxls_inCube_redundant.view(dtype = vxls_inCube_redundant.dtype.descr * ndim)
    _, indices, counts = np.unique(vxls_inCube_redundant_1D, return_index=True, return_counts=True)
    vxls_inCube_unique = vxls_inCube_redundant[indices]

    if normalizedDensity:
        density = counts / float(counts.max())
    else:
        density = counts

    return vxls_inCube_unique, density



def sample_pts_from_kdTree(kdTree, N_pts, distance_min, distance_max):
    """
    randomly sample pts from kdTree with distance_min to the nearest pts in the kdTree
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


def generate_surface_inCube_gt(inputFile, outputFile, N_pts_onSurface, N_pts_offSurface, \
        cube_D, resolutionList, inputDataType = 'pcd', recordDensity = True):
    """
    read a 3D model, record the surface information of the sampled on/off surface cubes

    --------
    inputs:
        inputFile:
        outputFile: write line by line
        inputDataType: 'pcd' / 'mesh'

    --------
    outputs:
        1
    """

    model3D = PlyData.read(inputFile)
    if inputDataType is 'pcd':
        pcd = model3D
        pts_xyz = np.array([pcd['vertex'][_coord] for _coord in ['x', 'y', 'z']]).T     # (N_pts, 3)
        kdTree = cKDTree(pts_xyz)
    else:
        mesh = model3D
        print("Need to be done.")
    
    for _resol in resolutionList:
        cube_D_mm = cube_D * _resol
        pts_onSurface = sample_pts_from_kdTree(kdTree, N_pts_onSurface, distance_min = 0, distance_max = 0)
        for _cubeCenter in pts_onSurface: 
            fe

"""
# read ply
# constants
    # NO of iterations & offSurface cubes
    # (50,)*3, 0.4
# cameraP/T, img_hw
kdtree
for iteration:
    rand pt index, cube_min = pt - cube_D/2
    print to file "{} {} {} {},".format(xyz_min, resol)
    for x,y,z in (50,)* cube, use kdtree.query_ball_point
        N_pts_vxl = how many pts are in the voxel
        print to file "{} {} {} {},".format(x,y,z,N_pts_vxl)
    print to file ";"
    for views:  ???
        print to file inScope & visibility
    print to file "\n"

thresh_offSurf_min/max
for N_offSurf
    generate offSurf cube_min, kdtree could help
    print to file "{} {} {} {},".format(xyz_min, resol)
    for views:  ???
        print to file inScope & visibility
    print to file "\n"


visualization:
    show sphere / cube mesh around the on/off_surface pts in the input ply file and save.


# reconstruct KDTree
    # this octree can also be used to detect occlusion (coarse2fine, view occlusion detection in the upper level).
    # generate similarityNet training 3D pts
    # OctNet is used to find the distance to the nearest point of the model.
# generate 3D surface gt (prob,normal_xyz), with shift, rotation augmentation.
# generate off surface f
# if mesh data, generate TSDF. TODO: how to generate the TSDF
"""



if __name__ == '__main__':
    import doctest
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)
