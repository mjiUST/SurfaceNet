import numpy as np
from scipy.spatial import cKDTree
from plyfile import PlyData, PlyElement

import utils

np.random.seed(201711)

def __extract_vxls_inCube__(kdTree, cubeCenter, cube_D, resolution, normalizedDensity = True):
    """
    consider the hypercube around cubeCenter (2D/3D/nD) with radius = cube_D * resolution / 2

    TODO: to support multiple cubeCenters queries

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
        cube_D, cube_resolutionList, inputDataType = 'pcd', recordDensity = True):
    """
    read a 3D model, record the surface pts information in the on/off-surface cubes

    --------
    inputs:
        inputFile:
        outputFile: 
        N_pts_on/offSurface: how many on/off-surface cubes to be sampled
        cube_D: cube.shape = (cube_D, )*3, larger than the cube_D_4training because of data augmentation
        cube_resolutionList: sample multi-scale cubes with different resolutions
        inputDataType: 'pcd' / 'mesh'

    --------
    outputs:
        1
    """

    outputStream = ""
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
        kdTree.query_ball_point(pts_onSurface, r = )
        for _cubeCenter in pts_onSurface:
            _cube_min = _cubeCenter - cube_D_mm / 2
            outputStream += "{} {} {} {},".format(_cube_min[0], _cube_min[1], _cube_min[2], cube_D, _resol)     # save xyz of cube_min
            vxls_inCube, density = __extract_vxls_inCube__(kdTree, _cubeCenter, cube_D, _resol, normalizedDensity = True) # (N_vxls, 3), (N_vxls, )
            for _vxl, _density in enumerate(density):
                # save xyz of vxls
                outputStream += "{} {} {} {},".format(vxls_inCube[_vxl, 0], vxls_inCube[_vxl, 1], vxls_inCube[_vxl, 2], _density)
        # outputStream += ";"   # can add other information, say visibility
        outputStream += "\n"


        ############
        # offSurface
        ############
        pts_offSurface = sample_pts_from_kdTree(kdTree, N_pts_offSurface, distance_min = cube_D_mm, distance_max = cube_D_mm * 10) # (N_pts_offSurface, 3)
        for _cubeCenter in pts_offSurface:
            outputStream += "{} {} {} {},".format(_cubeCenter[0], _cubeCenter[1], _cubeCenter[2], _resol)
        outputStream += "\n"

    utils.mkdirs_ifNotExist(os.path.dirname(outputFile))
    with open(outputFile, 'w') as f:
        f.write(outputStream)
    return 1


def read_saved_surfacePts(inputFile):
    """
    read saved on/off-surface pts
    Return 
    """

    with open(inputFile, 'r') as f:
        lines = f.readLines()

    for _line in lines:
        pts = _line.split('\n')[0].split(',')
        cube_min_x, cube_min_y, cube_min_z, cube_D, resolution = pts[0].split(' ')
        if len(pts) == 1:   # off surface cube
            continue
        for _pt in pts[1:]
            x, y, z, density = _pt[0].split(' ')
        

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
