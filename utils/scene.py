import numpy as np
import math
from plyfile import PlyData, PlyElement

import mesh_util

def initializeCubes(resol, cube_D, cube_Dcenter, cube_overlapping_ratio, BB):
    """
    generate {N_cubes} 3D overlapping cubes, each one has {N_cubeParams} embeddings
    for the cube with size of cube_D^3 the valid prediction region is the center part, say, cube_Dcenter^3
    E.g. cube_D=32, cube_Dcenter could be = 20. Because the border part of each cubes don't have accurate prediction because of ConvNet.

    ---------------
    inputs:
        resol: resolusion of each voxel in the CVC (mm)
        cube_D: size of the CVC (Colored Voxel Cube)
        cube_Dcenter: only keep the center part of the CVC, because of the boundery effect of ConvNet.
        cube_overlapping_ratio: pertantage of the CVC are covered by the neighboring ones
        BB: bounding box, numpy array: [[x_min,x_max],[y_min,y_max],[z_min,z_max]]
    outputs: 
        cubes_param_np: (N_cubes, N_params) np.float32
        cube_D_mm: scalar

    ---------------
    usage:
    >>> cubes_param_np, cube_D_mm = initializeCubes(resol=1, cube_D=22, cube_Dcenter=10, cube_overlapping_ratio=0.5, BB=np.array([[3,88],[-11,99],[-110,-11]]))
    xyz bounding box of the reconstructed scene: [ 3 88], [-11  99], [-110  -11]
    >>> print cubes_param_np[:3] 
    [([   3.,  -11., -110.], [0, 0, 0],  1.)
     ([   3.,  -11., -105.], [0, 0, 1],  1.)
     ([   3.,  -11., -100.], [0, 0, 2],  1.)]
    >>> print cubes_param_np['xyz'][18:22]
    [[   3.  -11.  -20.]
     [   3.  -11.  -15.]
     [   3.   -6. -110.]
     [   3.   -6. -105.]]
    >>> np.allclose(cubes_param_np['xyz'][18:22], cubes_param_np[18:22]['xyz'])
    True
    >>> print cube_D_mm
    22
    """

    cube_D_mm = resol * cube_D   # D size of each cube along each axis, 
    cube_Center_D_mm = resol * cube_Dcenter   # D size of each cube's center that is finally remained 
    cube_stride_mm = cube_Center_D_mm * cube_overlapping_ratio # the distance between adjacent cubes, 
    safeMargin = (cube_D_mm - cube_Center_D_mm)/2

    print('xyz bounding box of the reconstructed scene: {}, {}, {}'.format(*BB))
    N_along_axis = lambda _min, _max, _resol: int(math.ceil((_max - _min) / _resol))
    N_along_xyz = [N_along_axis( (BB[_axis][0] - safeMargin), (BB[_axis][1] + safeMargin), cube_stride_mm) for _axis in range(3)]   # how many cubes along each axis
    # store the ijk indices of each cube, in order to localize the cube
    cubes_ijk = np.indices(tuple(N_along_xyz))
    N_cubes = cubes_ijk.size / 3   # how many cubes

    cubes_param_np = np.empty((N_cubes,), dtype=[('xyz', np.float32, (3,)), ('ijk', np.uint32, (3,)), ('resol', np.float32)])    # attributes for each CVC (colored voxel cube)
    cubes_param_np['ijk'] = cubes_ijk.reshape([3,-1]).T  # i/j/k grid index
    cubes_xyz_min = cubes_param_np['ijk'] * cube_stride_mm + (BB[:,0][None,:] - safeMargin)
    cubes_param_np['xyz'] = cubes_xyz_min    # x/y/z coordinates (mm)
    cubes_param_np['resol'] = resol

    return cubes_param_np, cube_D_mm

def quantizePts2Cubes(pts_xyz, resol, cube_D, cube_Dcenter, cube_overlapping_ratio, BB = None):
    """
    generate overlapping cubes covering a set of points which is denser, so that we need to quantize the pts' coords.

    --------
    inputs:
        pts_xyz: generate the cubes around these pts
        resol: resolusion of each voxel in the CVC (mm)
        cube_D: size of the CVC (Colored Voxel Cube)
        cube_Dcenter: only keep the center part of the CVC, because of the boundery effect of ConvNet.
        cube_overlapping_ratio: pertantage of the CVC are covered by the neighboring ones
        BB: bounding box, numpy array: [[x_min,x_max],[y_min,y_max],[z_min,z_max]]

    --------
    outputs: 
        cubes_param_np: (N_cubes, N_params) np.float32
        cube_D_mm: scalar

    --------
    examples:
    >>> pts_xyz = np.array([[-1, 2, 0], [0, 2, 0], [1, 2, 0], [0,1,0], [0,0,0], [1,0,0], [2.1,0,0]])
    >>> #TODO quantizePts2Cubes(pts_xyz, resol=2, cube_D=3, cube_Dcenter = 2, cube_overlapping_ratio = 0.5)
    """

    cube_D_mm = resol * cube_D   # D size of each cube along each axis, 
    cube_Center_D_mm = resol * cube_Dcenter   # D size of each cube's center that is finally remained 
    cube_stride_mm = cube_Center_D_mm * cube_overlapping_ratio # the distance between adjacent cubes, 
    if BB is not None:
        safeMargin = cube_D_mm/2 # a little bit bigger than the BB
        inBB = np.array([np.logical_and(pts_xyz[:, _axis] >= (BB[_axis, 0] - safeMargin), pts_xyz[:, _axis] <= (BB[_axis, 1] + safeMargin)) for _axis in range(3)]).all(axis=0)
        pts_xyz = pts_xyz[inBB]
    shift = pts_xyz.min(axis=0)[None,...] # (1, 3), make sure the cube_ijk is non-negative, and try to cover the pts in the middle of the cubes.
    cubes_ijk_floor = (pts_xyz - shift) // cube_stride_mm # (N_pts, 3)
    cubes_ijk_ceil = ((pts_xyz - shift) // cube_stride_mm + 1)  # for each pt consider 2 neighboring cubesalong each axis.
    cubes_ijk = np.vstack([cubes_ijk_floor, cubes_ijk_ceil])  # (2*N_pts, 3)
    cubes_ijk_1d = cubes_ijk.view(dtype=cubes_ijk.dtype.descr * 3)
    cubes_ijk_unique = np.unique(cubes_ijk_1d).view(cubes_ijk_floor.dtype).reshape((-1, 3))  # (N_cubes, 3)
    N_cubes = cubes_ijk_unique.shape[0]   # how many cubes

    cubes_param_np = np.empty((N_cubes,), dtype=[('xyz', np.float32, (3,)), ('ijk', np.uint32, (3,)), ('resol', np.float32)])    # attributes for each CVC (colored voxel cube)
    cubes_param_np['ijk'] = cubes_ijk_unique  # i/j/k grid index
    cubesCenter_xyz = cubes_param_np['ijk'] * cube_stride_mm + shift
    cubes_param_np['xyz'] = cubesCenter_xyz - cube_D_mm/2    # (N_cubes, 3) min of x/y/z coordinates (mm)
    cubes_param_np['resol'] = resol

    return cubes_param_np, cube_D_mm


def readPointCloud_xyz(pointCloudFile = 'xx/xx.ply'):
    pcd = PlyData.read(pointCloudFile)  # pcd for Point Cloud Data
    pcd_xyz = np.c_[pcd['vertex']['x'], pcd['vertex']['y'], pcd['vertex']['z']]
    return pcd_xyz

def readBB_fromModel(objFile = 'xx/xx.obj'):
    mesh = mesh_util.load_obj(filename= objFile)
    BB = np.c_[mesh.v.min(axis=0), mesh.v.max(axis=0)]  # (3, 2)
    return BB


import doctest
doctest.testmod()
