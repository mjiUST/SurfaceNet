import numpy as np
import sys
import time


sys.path.append("../../utils")
import camera


#----------------------------------------------------------------------
def rayPooling_1cube_numpy_old(cameraPOs, cameraTs, cube_prediction, viewPair_viewIndx, param):
    """ given multiple cameraMs and one cube_prediction, perform rayPooling. 

    * map the orig_cube from the world coordinates to `self-defined` Normalized Device Coordinates (NDC), 
            so that the rays are parallel with one axis, along which the argmax will be performed.
            i.e. map from the external cutting pyramid frustum of the orig_cube to a cube.
        * perspective projection, for each viewIndx: 
                from 3D to 2D (img_h, img_w) ==> (img_h_relative, img_w_relative) ==> (D_NDC1,D_NDC2)
                '_relative' means shift the minimum in each view channel to (0,0), so that img_h/w_relative.min()=0
                D_NDC1 = img_h_relative.max
                D_NDC2 = img_w_relative.max
        * The 3rd dimention could be `euclidian dist(pt, orig)` or `pt_z in eye coordinates`, 
                in which the camera is in the origon.
                There is no big difference between the two choices 
                when the angle<vector(pt,orig), 2D plane' normal> is not large.
            * depth_resol = resol param of the cube 
                    {Small / Large} depth_resol leads to {poor geometry detail / huge memory assumption}.
                    D_NDC3 = (depth.max() - depth.min()) / depth_resol 
                    
    * In order to get rid of inverse transformation, which relies on invertable transformation matrix 
            which we don't have yet for the `self-defined 3D-3D tranformation`,
        * add a channel dimention to embedd the indices of the orignal coordinates, (N_channels,)+(cubeD,)*3 ==> (N_channels,D_NDC1,D_NDC2,D_NDC3)
                N_channels = 4: channel information are (prediction, index_Dim1, index_Dim2, index_Dim3)
        * argmax along NDC3 could read the orig_indexes of the reserved voxels with max predition. 
    
    TODO:
    ---------------
    * We can map the cube to the tightest bounded cube in NDC using projective projection, 
            1. don't how to define the tranformation matrix 
    * Currently in order to be simple and readable, don't optimize for the repeating view indexes, 
            for example: the 3rd view index will be calculated twice in the case of: viewPair_index = [(1,3),(15,3),(5,8)]

    inputs:
    ---------------
    cameraPOs: np.float, (N_views, 3, 4)
            the camera matrixes of all views
    cameraTs: np.float, (N_views, 3)
            the camera position of all views
    cube_prediction: np.float, (cubeD,)*3 
            each element is voxel's prediction value, based on which the max rayPooling is performed.
    viewPair_viewIndx: np.int, (N_viewPair, 2) 
            viewPair_viewIndx for a cube. Different cubes will have different selected view pairs.
    param: np.float, (Dim_param,)
            cube params including: x,y,z,resol,modelIndx...

    return: 
    ---------------
    cube_N_votes: np.int, (cubeD,)*3 
            with max = N_viewPair*2

    usage:
    ---------------

    """
    startT = time.time()
    cube_prediction = cube_prediction.squeeze()
    if cube_prediction.ndim != 3:
        raise ValueError('rayPooling method argument cube_prediction has {} dims'.format(cube_prediction.ndim))
    ## print("time 0 : {}".format(time.time() - startT)); startT = time.time()
    cubeD = cube_prediction.shape[-1]
    # D_NDC3 = 30 * cubeD
    N_pts = cubeD ** 3
    N_views = viewPair_viewIndx.size
    N_channels = 4 # channel information are (prediction, index_Dim1, index_Dim2, index_Dim3)
    viewIndx_np = viewPair_viewIndx.flatten()
    ## print("time 1 : {}".format(time.time() - startT)); startT = time.time()

    view_POs = cameraPOs[viewIndx_np] # (N_views, 3, 4)
    view_Ts = cameraTs[viewIndx_np] # (N_views, 3)
    min_x,min_y,min_z,resol,_ = param
    ## print("time 2 : {}".format(time.time() - startT)); startT = time.time()

    ijk_indices = np.indices((cubeD,cubeD,cubeD)) # (3,)+(cubeD,)*3
    pts_xyz = (ijk_indices * resol).reshape((3,-1)) + np.array([min_x,min_y,min_z])[:,None] # (3, N_pts), N_pts = cubeD**3
    # perspective projection of multiple views.
    # Note: the img_w/h_views are absolute index on different views. 
    #       img_w/h_views.max/min will not be meaningful before convert to relative index (with min=0 in each view's channel).
    ## print("time 3 : {}".format(time.time() - startT)); startT = time.time()
    img_h_views_absolute, img_w_views_absolute, depth = camera.perspectiveProj(projection_M = view_POs, xyz_3D = pts_xyz.T, return_int_hw = True, return_depth = True) # (N_views, N_pts)
    ## print("time 3.1 : {}".format(time.time() - startT)); startT = time.time()
    img_w_views_relative = img_w_views_absolute - img_w_views_absolute.min(axis=1, keepdims=True) # (N_views, N_pts)
    img_h_views_relative = img_h_views_absolute - img_h_views_absolute.min(axis=1, keepdims=True) # (N_views, N_pts)
    ## print("time 4 : {}".format(time.time() - startT)); startT = time.time()

    ## depth = np.linalg.norm(view_Ts[:,:,None] - pts_xyz[None,:,:], ord=2, axis=1, keepdims=False) # (N_views, 3, 1) - (1, 3, N_pts) ==> (N_views, 3, N_pts) ==> (N_views, N_pts)
    ## print("time 4.1 : {}".format(time.time() - startT)); startT = time.time()
    depth_resol = resol #(depth.max() - depth.min()) / (D_NDC3 - 1)
    depth_int = (depth / depth_resol).round().astype(np.int32) # (N_views, N_pts)
    D_NDC3 = depth_int.max() - depth_int.min() + 1
    # in order to do the operation for all the viewIndx, we can simply let all the viewIndx share the same (D_NDC1,D_NDC2,D_NDC3), 
    # readable but waste memory
    ## print("time 5 : {}".format(time.time() - startT)); startT = time.time()
    D_NDC1, D_NDC2 = img_h_views_relative.max() + 1, img_w_views_relative.max() + 1
    views_prediction_NDC = np.zeros((N_channels, N_views, D_NDC1, D_NDC2, D_NDC3))
    ## print("time 6 : {}".format(time.time() - startT)); startT = time.time()
    view_dimIndx = np.repeat(np.arange(N_views)[:,None], N_pts, axis=1) # (N_views,1) ==> (N_views,N_pts)
    # embedd the prediction and indexes_in_orig_coordinates into 4D tuples. (4,)+(cubeD,)*3 ==> (4,N_views,)+(cubeD,)*3 ==> (4, N_views*cubeD**3)
    channels_infor = np.repeat(np.vstack([cube_prediction[None,...], ijk_indices])[:,None,...], N_views, axis=0).reshape((4,-1))
    ## print("time 7 : {}".format(time.time() - startT)); startT = time.time()
    
    indx_NDC_view = view_dimIndx.flatten()
    indx_NDC1 = img_h_views_relative.flatten()
    indx_NDC2 = img_w_views_relative.flatten()
    ## print("time 8 : {}".format(time.time() - startT)); startT = time.time()
    indx_NDC3 = depth_int.flatten() - depth_int.min()
    ## print("time 8.1 : {}".format(time.time() - startT)); startT = time.time()

    views_prediction_NDC[:, indx_NDC_view, indx_NDC1, indx_NDC2, indx_NDC3] = channels_infor # (N_channels, N_views, D_NDC1, D_NDC2, D_NDC3)
    # argmax along the depth dimention, for each element in NDC1/2, there is a argmax value, even though some of them are not mapped by the voxel.
    ## print("time 9 : {}".format(time.time() - startT)); startT = time.time()
    argmax_NDC3 = np.argmax(views_prediction_NDC[0], axis=-1) # (N_views, D_NDC1, D_NDC2) 
    ## print("time 9.1 : {}".format(time.time() - startT)); startT = time.time()
    argmax_indx = np.vstack([np.indices((N_views, D_NDC1, D_NDC2)), argmax_NDC3[None,...]]).reshape((4,-1)) # (4, N_views, D_NDC1, D_NDC2) ==> (4, N_views*D_NDC1*D_NDC2)
    # if all the elements along the depth dimention (NDC3) are not mapped by any voxel. Filter out the corresponding pixels:
    ## print("time 10 : {}".format(time.time() - startT)); startT = time.time()
    argmax_indx_filter = np.any(views_prediction_NDC[0], axis = -1).flatten() # (N_views*D_NDC1*D_NDC2,) 
    ## print("time 10.1 : {}".format(time.time() - startT)); startT = time.time()
    argmax_indx = argmax_indx[:, argmax_indx_filter]
    # for each pixel, only one voxel can get ray vote. So finnally there are N_pixels voxels will be left by rayPooling.
    ## print("time 11 : {}".format(time.time() - startT)); startT = time.time()
    rayPooling_indx = views_prediction_NDC[-3:, argmax_indx[0], argmax_indx[1], argmax_indx[2], argmax_indx[3]] # (3, N_pixels), N_pixels < N_views*D_NDC1*D_NDC2 
    rayPooling_indx = np.vstack([argmax_indx[0], rayPooling_indx.astype(np.int32)]) # (4, N_pixels)
    ## print("time 12 : {}".format(time.time() - startT)); startT = time.time()
    cube_eachView_vote = np.zeros((N_views,)+(cubeD,)*3).astype(np.bool) # (N_views,)+(cubeD,)*3
    cube_eachView_vote[tuple(rayPooling_indx)] = True
    ## print("time 13 : {}".format(time.time() - startT)); startT = time.time()
    cube_N_votes = np.sum(cube_eachView_vote, axis=0) # (N_views,)+(cubeD,)*3 ==> (cubeD,)*3
    ## print("time 14 : {}".format(time.time() - startT)); startT = time.time()
    return cube_N_votes
    

#----------------------------------------------------------------------
def rayPooling_1cube_numpy(cameraPOs, cameraTs, cube_prediction, viewPair_viewIndx, xyz, resol, prediction_thresh=None):
    """ given multiple cameraMs and one cube_prediction, perform rayPooling. 

    * map the orig_cube from the world coordinates to `self-defined` Normalized Device Coordinates (NDC), 
            so that the rays are parallel with one axis, along which the argmax will be performed.
            i.e. map from the external cutting pyramid frustum of the orig_cube to a cube.
            And the 3D NDC can be simplified into 2D, one dim is (img_h, img_w) tuple, the other axis is depth.
        * perspective projection, for each viewIndx: 
                from 3D to 2D (img_h, img_w) ==> (D_NDC1,)
        * The other dimention could be `euclidian dist(pt, orig)` or 
                `pt_z in eye coordinates` in which the camera is in the origon.
                There is no big difference between the two choices 
                when the angle<vector(pt,orig), 2D plane' normal> is not large.
            * depth_resol = resol param of the cube 
                    {Small / Large} depth_resol leads to {poor geometry detail / huge memory assumption}.
                    D_NDC2 = (depth.max() - depth.min()) / depth_resol 
                    
    * In order to get rid of inverse mapping, which relies on invertable transformation matrix 
            which is hard to get for the `self-defined 3D-2D tranformation`,
        * add a channel dimention to embedd the indices of the orignal coordinates, (N_channels,)+(cubeD,)*3 ==> (N_channels,D_NDC1,D_NDC2)
                N_channels = 4: channel information are (prediction, index_Dim1, index_Dim2, index_Dim3)
        * argmax along NDC2 could read the orig_indexes of the reserved voxels with max predition. 
        inputs:
    ---------------
    cameraPOs: np.float, (N_views, 3, 4)
            the camera matrixes of all views
    cameraTs: np.float, (N_views, 3)
            the camera position of all views
    cube_prediction: np.float, (cubeD,)*3 
            each element is voxel's prediction value, based on which the max rayPooling is performed.
    viewPair_viewIndx: np.int, (N_viewPair, 2) 
            viewPair_viewIndx for a cube. Different cubes will have different selected view pairs.
    param: np.float, (Dim_param,)
            cube params including: x,y,z,resol
    prediction_thresh: None / scalar
            Filter the prediction before the expensive operations.

    return: 
    ---------------
    cube_N_votes: np.int, (cubeD,)*3 
            with max = N_viewPair*2

 
    log:
    ---------------
    * We can map the cube to the tightest bounded cube in NDC using projective projection, 
            1. don't how to define the tranformation matrix 
            [solution] use the 2D NDC, using (img_h,img_w) tuple as one dimention!
    * don't compute the repeating views
            for example: the 3rd view index will not be calculated twice in the case of: viewPair_index = [(1,3),(15,3),(5,8)]
    * only compute valid voxels, whoses prediction is > 1.0/threshold

    usage:
    ---------------

    """
    startT = time.time()
    cube_prediction = cube_prediction.squeeze()
    if cube_prediction.ndim != 3:
        raise ValueError('rayPooling method argument cube_prediction has {} dims'.format(cube_prediction.ndim))
    ## print("time 0 : {}".format(time.time() - startT)); startT = time.time()
    cube_shape = cube_prediction.shape[-3:]
    cubeD = cube_prediction.shape[-1]
    # D_NDC3 = 30 * cubeD
    N_pts = cubeD ** 3
    N_views = viewPair_viewIndx.size
    N_channels = 2 # channel information are (prediction, flatten_index)
    viewIndx_set, viewIndx_inverseIndx = np.unique(viewPair_viewIndx.flatten(), return_inverse=True)
    ## print("time 1 : {}".format(time.time() - startT)); startT = time.time()
    N_views_set = viewIndx_set.size
    view_POs = cameraPOs[viewIndx_set] # (N_views_set, 3, 4)
    view_Ts = cameraTs[viewIndx_set] # (N_views_set, 3)
    min_x, min_y, min_z = xyz
    ## print("time 2 : {}".format(time.time() - startT)); startT = time.time()

    pts_select = np.arange(cube_prediction.size) if prediction_thresh is None else \
                np.where(cube_prediction.flatten() > prediction_thresh)[0] # (N_pts,)
#     ijk_indices = np.indices(cube_shape) # (3,)+cube_shape
#     ijk_select = ijk_indices.reshape((3,-1))[:,pts_select]# (3, N_pts)
    ijk_select = np.asarray(np.unravel_index(pts_select, dims=cube_shape)) # flatten index --> nD array index
    pts_xyz = ijk_select * resol + np.array([min_x,min_y,min_z])[:,None] # (3, N_pts)
    # perspective projection of multiple views.
    # Note: the img_w/h_views are absolute index on different views. 
    #       img_w/h_views.max/min will not be meaningful before convert to relative index (with min=0 in each view's channel).
    ## print("time 3 : {}".format(time.time() - startT)); startT = time.time()
    img_h_views_absolute, img_w_views_absolute, depth = camera.perspectiveProj(projection_M = view_POs, xyz_3D = pts_xyz.T, return_int_hw = True, return_depth = True) # (N_views_set, N_pts)
    ## print("time 3.1 : {}".format(time.time() - startT)); startT = time.time()
    ## depth = np.linalg.norm(view_Ts[:,:,None] - pts_xyz[None,:,:], ord=2, axis=1, keepdims=False) # (N_views_set, 3, 1) - (1, 3, N_pts) ==> (N_views_set, 3, N_pts) ==> (N_views_set, N_pts)
    ## print("time 4.1 : {}".format(time.time() - startT)); startT = time.time()
    depth_resol = resol #(depth.max() - depth.min()) / (D_NDC3 - 1)
    depth_int = (depth / depth_resol).round().astype(np.int32) # (N_views_set, N_pts)
    channels_infor = np.vstack([cube_prediction.flatten()[pts_select][None,...], pts_select]) # (N_channels, N_pts)
    cube_eachView_vote = np.zeros((N_views_set,)+cube_shape).astype(np.bool) # (N_views_set,)+cube_shape
    
    for _view in range(N_views_set):
        _depth_int = depth_int[_view]# (N_pts,)
        if _depth_int.size == 0:
            continue            
        D_NDC2 = _depth_int.max() - _depth_int.min() + 1
        _img_w_abs, _img_h_abs = img_w_views_absolute[_view], img_h_views_absolute[_view] # (N_pts,)
        _img_wh_abs = np.c_[_img_w_abs, _img_h_abs]# (N_pts, 2)
        _dtype_wh = _img_w_abs.dtype.descr * 2
        _wh_tpl = _img_wh_abs.view(_dtype_wh) # (N_pts,), dtype=[(.,.)]
        _wh_tpl_set, _wh_tpl_indx = np.unique(_wh_tpl, return_inverse = True) # (N_wh,) (N_pts,)
        D_NDC1 = len(_wh_tpl_set) # NO. of pixels which the N_pts voxels are corresponding to.

        views_prediction_NDC = np.zeros((N_channels, D_NDC1, D_NDC2)) # TODO: sparse matrix to replace, but need to find argmax(sparse)
        indx_NDC2 = _depth_int.flatten() - _depth_int.min()
        views_prediction_NDC[:, _wh_tpl_indx, indx_NDC2] = channels_infor # (N_channels, D_NDC1, D_NDC2)
        # argmax along the depth dimention, 
        argmax_NDC2 = np.argmax(views_prediction_NDC[0], axis=-1) # (D_NDC1,) 
        # for each pixel, only one voxel can get ray vote. So finnally there are N_pixels voxels will be left by rayPooling.
        rayPooling_indx = views_prediction_NDC[-1:, np.arange(D_NDC1), argmax_NDC2].astype(np.int32) # (3, N_pixels), N_pixels = D_NDC1
        cube_eachView_vote[_view][np.unravel_index(rayPooling_indx, dims=cube_shape)] = True
    
    cube_N_votes = np.sum(cube_eachView_vote[viewIndx_inverseIndx], axis=0) # (N_views,)+cube_shape ==> cube_shape
    ## print("time 14 : {}".format(time.time() - startT)); startT = time.time()
    return cube_N_votes




#----------------------------------------------------------------------
#import theano
#import theano.tensor as T

#def rayPooling_1cube_theano(cameraPOs, cameraTs, cube_prediction, viewPair_viewIndx, param):
#    pass

