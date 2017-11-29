import numpy as np
from scipy.spatial import cKDTree
from plyfile import PlyData, PlyElement

import utils

np.random.seed(201711)

def calculate_pts_density(pts, normalized = True):
    pass

def generate_on_off_surface_pts():
    """ 
    Generate 3D pts candidates for similarityNet
    """
    pass


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
    >>> xy = np.c_[xy, np.array([[5,7], [8,8]]).T]
    >>> tree = cKDTree(xy.T)
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
    if distance_max == 0:   # only need to randomly select leaves in the tree
        selector = np.zeros(kdTree.n, ).astype(np.bool)
        selector[np.random.randint(0, kdTree.n, N_pts)] = 1
        pts = kdTree.data[selector]
    else: 
        for _i in range(N_pts):
            for _j in range(100):    # if 10 proposals cannot fall in the range, print a warning.
                vrand = utils.sample_randomVector(vmin = kdTree.mins.astype(np.float32), vmax = kdTree.maxes.astype(np.float32)) # (3,)
                dd, ii = kdTree.query(vrand, k=1)
                if (dd >= distance_min) and (dd <= distance_max):
                    pts.append(vrand)
                    break   # the vector sample is OK
                if _j is 99:
                    print("off surface distance threshold is too large: {} - {}".format(distance_min, distance_max))

    return pts


# xy = np.mgrid[0:5, 2:8].reshape((2,-1))
# tree = cKDTree(xy.T)
# pts = sample_pts_from_kdTree(tree, N_pts = 2, distance_min = 0, distance_max = 0)
# dd, ii = tree.query(pts, k=1)
# dd
# pts
# pts = sample_pts_from_kdTree(tree, N_pts = 2, distance_min = 1, distance_max = 2)
# dd, ii = tree.query(pts, k=1)
# dd
# pts



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

import doctest
doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)
