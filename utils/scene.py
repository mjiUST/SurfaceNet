import numpy as np
import math

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
    print('xyz bounding box of the reconstructed scene: {}, {}, {}'.format(*BB))
    N_along_axis = lambda _min, _max, _resol: int(math.ceil((_max - _min) / _resol))
    N_along_xyz = [N_along_axis(BB[_axis][0], BB[_axis][1], cube_stride_mm) for _axis in range(3)]   # how many cubes along each axis
    # store the ijk indices of each cube, in order to localize the cube
    cubes_ijk = np.indices(tuple(N_along_xyz))
    N_cubes = cubes_ijk.size / 3   # how many cubes

    cubes_param_np = np.empty((N_cubes,), dtype=[('xyz', np.float32, (3,)), ('ijk', np.uint32, (3,)), ('resol', np.float32)])    # attributes for each CVC (colored voxel cube)
    cubes_param_np['ijk'] = cubes_ijk.reshape([3,-1]).T  # i/j/k grid index
    cubes_param_np['xyz'] = cubes_param_np['ijk'] * cube_stride_mm + BB[:,0][None,:]    # x/y/z coordinates (mm)
    cubes_param_np['resol'] = resol

    return cubes_param_np, cube_D_mm


import doctest
doctest.testmod()
