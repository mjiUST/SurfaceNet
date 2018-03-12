import os
import math
import utils
import numpy as np



def __readCameraPO_as_np_DTU__(cameraPO_file):
    """ 
    only load a camera PO in the file
    ------------
    inputs:
        cameraPO_file: the camera pose file of a specific view
    outputs:
        cameraPO: np.float64 (3,4)
    ------------
    usage:
    >>> p = __readCameraPO_as_np_DTU__(cameraPO_file = './test/cameraPO/pos_060.txt') 
    >>> p # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    array([[  1.67373847e+03,  -2.15171320e+03,   1.26963515e+03,
        ...
              6.58552305e+02]])
    """
    cameraPO = np.loadtxt(cameraPO_file, dtype=np.float64, delimiter = ' ')
    return cameraPO


def __readCameraPOs_as_np_Middlebury__(cameraPO_file, viewList):
    """ 
    load camera POs of multiple views in one file
    ------------
    inputs:
        cameraPO_file: the camera pose file of a specific view
        viewList: view list 
    outputs:
        cameraPO: np.float64 (N_views,3,4)
    ------------
    usage:
    >>> p = __readCameraPOs_as_np_Middlebury__(cameraPO_file = './test/cameraPO/dinoSR_par.txt', viewList=[3,8]) 
    >>> p # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        array([[[ -1.22933223e+03,   3.08329199e+03,   2.02784015e+02,
        ...
        6.41227584e-01]]])
    """
    with open(cameraPO_file) as f:
        lines = f.readlines() 

    cameraPOs = np.empty((len(lines), 3, 4)).astype(np.float64)
    for _n, _l in enumerate(lines):
        if _n == 0:
            continue
        _params = np.array(_l.strip().split(' ')[1:], dtype=np.float64) 
        _K = _params[:9].reshape((3,3))
        _R = _params[9:18].reshape((3,3))
        _t = _params[18:].reshape((3,1))
        cameraPOs[_n] = np.dot(_K, np.c_[_R,_t])
    return cameraPOs[viewList]




def readCameraPOs_as_np(datasetFolder, datasetName, poseNamePattern, model, viewList):
    """
    inputs:
      datasetFolder: 'x/x/x/middlebury'
      datasetName: 'DTU' / 'Middlebury'
      model: 1..128 / 'dinoxx'
      viewList: [3,8,21,...]
    output:
      cameraPOs (N_views,3,4) np.flost64
    """
    cameraPOs = np.empty((len(viewList),3,4), dtype=np.float64)

    if 'Middlebury' in datasetName:
        cameraPOs = __readCameraPOs_as_np_Middlebury__(cameraPO_file = os.path.join(datasetFolder, poseNamePattern), viewList=viewList)
    else: # cameraPOs are stored in different files
        for _i, _view in enumerate(viewList):
            # if 'DTU' in datasetName:
            _cameraPO = __readCameraPO_as_np_DTU__(cameraPO_file = os.path.join(datasetFolder, poseNamePattern.replace('#', '{:03}'.format(_view)).replace('@', '{}'.format(_view))))
            cameraPOs[_i] = _cameraPO
    return cameraPOs





def __cameraP2T__(cameraPO):
    """
    cameraPO: (3,4)
    return camera center in the world coords: cameraT (3,0)
    >>> P = np.array([[798.693916, -2438.153488, 1568.674338, -542599.034996], \
                  [-44.838945, 1433.912029, 2576.399630, -1176685.647358], \
                  [-0.840873, -0.344537, 0.417405, 382.793511]])
    >>> t = np.array([555.64348632032, 191.10837560939, 360.02470478273])
    >>> np.allclose(__cameraP2T__(P), t)
    True
    """
    homo4D = np.array([np.linalg.det(cameraPO[:,[1,2,3]]), -1*np.linalg.det(cameraPO[:,[0,2,3]]), np.linalg.det(cameraPO[:,[0,1,3]]), -1*np.linalg.det(cameraPO[:,[0,1,2]]) ])
    cameraT = homo4D[:3] / homo4D[3]
    return cameraT

    
def cameraPs2Ts(cameraPOs):
    """
    convert multiple POs to Ts. 
    ----------
    input:
        cameraPOs: list / numpy 
    output:
        cameraTs: list / numpy
    """
    if type(cameraPOs) is list:
        N = len(cameraPOs)
    else:                
        N = cameraPOs.shape[0]
    cameraT_list = []    
    for _cameraPO in cameraPOs:
        cameraT_list.append(__cameraP2T__(_cameraPO))

    return cameraT_list if type(cameraPOs) is list else np.stack(cameraT_list)


def perspectiveProj(projection_M, xyz_3D, return_int_hw = True, return_depth = False):
    """ 
    perform perspective projection from 3D points to 2D points given projection matrix(es)
            support multiple projection_matrixes and multiple 3D vectors
    notice: [matlabx,matlaby] = [width, height]

    ----------
    inputs:
    projection_M: numpy with shape (3,4) / (N_Ms, 3,4), during calculation (3,4) will --> (1,3,4)
    xyz_3D: numpy with shape (3,) / (N_pts, 3), during calculation (3,) will --> (1,3)
    return_int_hw: bool, round results to integer when True.

    ----------
    outputs:
    img_h, img_w: (N_pts,) / (N_Ms, N_pts)

    ----------
    usages:

    inputs: (N_Ms, 3,4) & (N_pts, 3), return_int_hw = False/True

    >>> np.random.seed(201611)
    >>> Ms = np.random.rand(2,3,4)
    >>> pts_3D = np.random.rand(2,3)
    >>> pts_2Dh, pts_2Dw = perspectiveProj(Ms, pts_3D, return_int_hw = False)
    >>> np.allclose(pts_2Dw, np.array([[ 1.35860185,  0.9878389 ],
    ...        [ 0.64522543,  0.76079278 ]]))
    True
    >>> pts_2Dh_int, pts_2Dw_int = perspectiveProj(Ms, pts_3D, return_int_hw = True)
    >>> np.allclose(pts_2Dw_int, np.array([[1, 1], [1, 1]]))
    True

    inputs: (3,4) & (3,)

    >>> np.allclose(
    ...         np.r_[perspectiveProj(Ms[1], pts_3D[0], return_int_hw = False)],
    ...         np.stack((pts_2Dh, pts_2Dw))[:,1,0])
    True
    """

    if projection_M.shape[-2:] != (3,4):
        raise ValueError("perspectiveProj needs projection_M with shape (3,4), however got {}".format(projection_M.shape))

    if xyz_3D.ndim == 1:
        xyz_3D = xyz_3D[None,:]

    if xyz_3D.shape[1] != 3 or xyz_3D.ndim != 2:
        raise ValueError("perspectiveProj needs xyz_3D with shape (3,) or (N_pts, 3), however got {}".format(xyz_3D.shape))
    # perspective projection
    N_pts = xyz_3D.shape[0]
    xyz1 = np.c_[xyz_3D, np.ones((N_pts,1))].astype(np.float64) # (N_pts, 3) ==> (N_pts, 4)
    pts_3D = np.matmul(projection_M, xyz1.T) # (3, 4)/(N_Ms, 3, 4) * (4, N_pts) ==> (3, N_pts)/(N_Ms,3,N_pts)
    # the result is vector: [w,h,1], w is the first dim!!! (matlab's x/y/1')
    pts_2D = pts_3D[...,:2,:]
    pts_2D /= pts_3D[...,2:3,:] # (2, N_pts) /= (1, N_pts) | (N_Ms, 2, N_pts) /= (N_Ms, 1, N_pts)
    if return_int_hw: 
        pts_2D = pts_2D.round().astype(np.int64)  # (2, N_pts) / (N_Ms, 2, N_pts)
    img_w, img_h = pts_2D[...,0,:], pts_2D[...,1,:] # (N_pts,) / (N_Ms, N_pts)
    if return_depth:
        depth = pts_3D[...,2,:]
        return img_h, img_w, depth
    return img_h, img_w



def perspectiveProj_cubesCorner(projection_M, cube_xyz_min, cube_D_mm, return_int_hw = True, return_depth = False):
    """ 
    perform perspective projection from 3D points to 2D points given projection matrix(es)
            support multiple projection_matrixes and multiple 3D vectors
    notice: [matlabx,matlaby] = [width, height]

    ----------
    inputs:
    projection_M: numpy with shape (3,4) / (N_Ms, 3,4), during calculation (3,4) will --> (1,3,4)
    cube_xyz_min: numpy with shape (3,) / (N_pts, 3), during calculation (3,) will --> (1,3)
    cube_D_mm: cube with shape D^3
    return_int_hw: bool, round results to integer when True.
    return_depth: bool

    ----------
    outputs:
    img_h, img_w: (N_Ms, N_pts, 8)

    ----------
    usages:

    inputs: (N_Ms, 3, 4) & (N_pts, 3), return_int_hw = False/True, outputs (N_Ms, N_pts, 8)

    >>> np.random.seed(201611)
    >>> Ms = np.random.rand(2,3,4)
    >>> pts_3D = np.random.rand(2,3)
    >>> pts_2Dh, pts_2Dw = perspectiveProj_cubesCorner(Ms, pts_3D, cube_D_mm = 1, return_int_hw = False)
    >>> np.allclose(pts_2Dw[:,:,0], np.array([[ 1.35860185,  0.9878389 ],
    ...        [ 0.64522543,  0.76079278 ]]))
    True
    >>> pts_2Dh_int, pts_2Dw_int = perspectiveProj_cubesCorner(Ms, pts_3D, cube_D_mm = 1, return_int_hw = True)
    >>> np.allclose(pts_2Dw_int[:,:,0], np.array([[1, 1], [1, 1]]))
    True

    inputs: (3,4) & (3,), outputs (1,1,8)

    >>> np.allclose(
    ...         perspectiveProj_cubesCorner(Ms[1], pts_3D[0], cube_D_mm = 1, return_int_hw = False)[0],
    ...         pts_2Dh[1,0])        # (1,1,8)
    True
    """

    if projection_M.shape[-2:] != (3,4):
        raise ValueError("perspectiveProj needs projection_M with shape (3,4), however got {}".format(projection_M.shape))

    if cube_xyz_min.ndim == 1:
        cube_xyz_min = cube_xyz_min[None,:]     # (3,) --> (N_pts, 3)

    if cube_xyz_min.shape[1] != 3 or cube_xyz_min.ndim != 2:
        raise ValueError("perspectiveProj needs cube_xyz_min with shape (3,) or (N_pts, 3), however got {}".format(cube_xyz_min.shape))

    N_pts = cube_xyz_min.shape[0]
    cubeCorner_shift = np.indices((2, 2, 2)).reshape((3, -1)).T[None,:,:] * cube_D_mm    # (3,2,2,2) --> (1,8,3)
    cubeCorner = cube_xyz_min[:,None,:] + cubeCorner_shift      # (N_pts, 1, 3) + (1,8,3) --> (N_pts, 8, 3)
    img_h, img_w = perspectiveProj(projection_M = projection_M, xyz_3D = cubeCorner.reshape((N_pts*8, 3)), return_int_hw = return_int_hw, return_depth = return_depth)    # img_w/h: (N_Ms, N_pts*8) 
    img_w = img_w.reshape((-1, N_pts, 8))
    img_h = img_h.reshape((-1, N_pts, 8))
    return img_h, img_w


def calculate_angle_p1_p2_p3(p1,p2,p3,return_angle=True, return_cosine=True):
    """
    calculate angle <p1,p2,p3>, which is the angle between the vectors p2p1 and p2p3 

    Parameters
    ----------
    p1/p2/p3: numpy with shape (3,)
    return_angle: return the radian angle
    return_cosine: return the cosine value

    Returns
    -------
    angle, cosine

    Examples
    --------
    """
    unit_vector = lambda v: v / np.linalg.norm(v)
    angle = lambda v1,v2: np.arccos(np.clip(np.dot(unit_vector(v1), unit_vector(v2)), -1.0, 1.0))
    cos_angle = lambda v1,v2: np.clip(np.dot(unit_vector(v1), unit_vector(v2)), -1.0, 1.0)

    vect_p2p1 = p1-p2
    vect_p2p3 = p3-p2
    return angle(vect_p2p1, vect_p2p3) if return_angle else None , \
            cos_angle(vect_p2p1, vect_p2p3) if return_cosine else None


def viewPairAngles_wrt_pts(cameraTs, pts_xyz):
    """
    given a set of camera positions and a set of points coordinates, output the angle between camera pairs w.r.t. each 3D point.

    -----------
    inputs:
        cameraTs: (N_views, 3) camera positions
        pts_xyz: (N_pts, 3) 3D points' coordinates

    -----------
    outputs:
        viewPairAngle_wrt_pts: (N_pts, N_viewPairs) angle 

    -----------
    usages:
    >>> pts_xyz = np.array([[0,0,0],[1,1,1]], dtype=np.float32)     # p1 / p2
    >>> cameraTs = np.array([[0,0,1], [0,1,1], [1,0,1]], dtype=np.float32)      # c1/2/3
    >>> viewPairAngles_wrt_pts(cameraTs, pts_xyz) * 180 / math.pi    # output[i]: [<c1,pi,c2>, <c1,pi,c3>, <c2,pi,c3>]
    array([[ 45.,  45.,  60.],
           [ 45.,  45.,  90.]], dtype=float32)
    """

    unitize_array = lambda array, axis: array/np.linalg.norm(array, axis=axis, ord=2, keepdims=True)
    calc_arccos = lambda cos_values: np.arccos(np.clip(cos_values, -1.0, 1.0))  # TODO does it need clip ?
    N_views = cameraTs.shape[0]
    vector_pts2cameras = pts_xyz[:,None,:] - cameraTs[None,...]   # (N_pts, 1, 3) - (1, N_views, 3) ==> (N_pts, N_views, 3)
    unit_vector_pts2cameras = unitize_array(vector_pts2cameras, axis = -1)    # (N_pts, N_views, 3)  unit vector along axis=-1

    # do the matrix multiplication for the (N_pats,) tack of (N_views, 3) matrixs 
    ## (N_pts, N_views, 3) * (N_pts, 3, N_views) ==> (N_pts, N_views, N_views)
    # viewPairCosine_wrt_pts = np.matmul(unit_vector_pts2cameras, unit_vector_pts2cameras.transpose((0,2,1)))
    viewPairs = utils.k_combination_np(range(N_views), k = 2)     # (N_combinations, 2)
    viewPairCosine_wrt_pts = np.sum(np.multiply(unit_vector_pts2cameras[:, viewPairs[:,0]], unit_vector_pts2cameras[:, viewPairs[:,1]]), axis=-1)    # (N_pts, N_combinations, 3) elementwise multiplication --> (N_pts, N_combinations) sum over the last axis
    viewPairAngle_wrt_pts = calc_arccos(viewPairCosine_wrt_pts)     # (N_pts, N_combinations)
    return viewPairAngle_wrt_pts





import doctest
doctest.testmod()




