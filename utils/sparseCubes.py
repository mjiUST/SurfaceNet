import numpy as np
import os
import cPickle as pickle
import rayPooling
import sys
import camera
from plyfile import PlyData, PlyElement

def dense2sparse(prediction, rgb, param, viewPair, min_prob = 0.5, rayPool_thresh = 0, \
        enable_centerCrop = False, cube_Dcenter = None, \
        enable_rayPooling = False, cameraPOs = None, cameraTs = None):
    """
    convert dense prediction / rgb to sparse representation
    using rayPooling & prob_thresholding & center crop

    Note:
        rayPooling: threshold of max_votes = rayPool_thresh 
        after center crop: the min_xyz should be shifted to the new position
   
    --------------
    inputs:
        prediction: np.float16(N_cubes,D,D,D)
        rgb: np.uint8(N_cubes,D,D,D,3)
        param: np.float32(N_cubes, N_params): 'ijk'/'xyz'/'resol'
        viewPair: np.uint16(N_cubes, N_viewPairs, 2)
        min_prob = 0.5

        enable_centerCrop = False # used for center crop
        cube_Dcenter = None

        enable_rayPooling = False # used for rayPooling
        cameraPOs = None
        cameraTs = None
    ---------------
    outputs:
        nonempty_cube_indx: np.uint32 (N_nonempty_cubes,)
        vxl_ijk_list[i]: np.uint8 (iN_voxels, 3)
        prediction_list[i]: np.float16 (iN_voxels,) # nth element corresponds to cube_n
        rgb_list[i]: np.uint8 (iN_voxels, 3)
        rayPooling_votes_list[i]: np.uint8 (iN_voxels,)
        param_new: np.float32(N_nonempty_cubes, 4): after center crop
    """

    N_cubes, D_orig, _, _ = prediction.shape # [:2]
    nonempty_cube_indx, vxl_ijk_list, prediction_list, rgb_list, rayPooling_votes_list =\
            [], [], [], [], []
    param_new = np.copy(param)
    
    slc = np.s_[:,:,:] # select all in first 3 dims
    if enable_centerCrop:
        _Cmin, _Cmax = (D_orig-cube_Dcenter)/2, (D_orig-cube_Dcenter)/2 + cube_Dcenter 
        # np.s_[_Cmin:_Cmax,_Cmin:_Cmax,_Cmin:_Cmax]
        slc = (slice(_Cmin, _Cmax, 1),)*3 # np.s_[1:6] = slice(1,6)
        # shift the min_xyz of the center_cropped cubes
        param_new['xyz'] += param_new['resol'][:, None] * _Cmin # (N_cubes, 3) + (N_cubes,1) = (N_cubes, 3)

    for _n in range(N_cubes):
        if enable_rayPooling:
            # rayPooling function has already done the prob_thresholding
            rayPool_votes = rayPooling.rayPooling_1cube_numpy(cameraPOs, cameraTs, \
                    viewPair_viewIndx = viewPair[_n], xyz = param[_n]['xyz'], resol = param[_n]['resol'],\
                    cube_prediction = prediction[_n], prediction_thresh = min_prob).astype(np.uint8)
            # 2n view pairs, only reserve the voxel with raypooling votes >= n
            vxl_ijk_tuple = np.where(rayPool_votes[slc] >= rayPool_thresh) 
        if (not enable_rayPooling) or rayPool_thresh == 0: # only filter out voxels with low prob
            vxl_ijk_tuple = np.where(prediction[_n][slc] > min_prob)
        if vxl_ijk_tuple[0].size == 0:
            continue # empty cube

        nonempty_cube_indx.append(_n)
        vxl_ijk_list.append(np.c_[vxl_ijk_tuple].astype(np.uint8)) # (iN_vxl,3)
        prediction_list.append(prediction[_n][slc][vxl_ijk_tuple].astype(np.float16)) # (D,D,D)-->(iN_vxl,)
        rgb_list.append(rgb[_n][slc][vxl_ijk_tuple].astype(np.uint8)) # (D,D,D,3)-->(iN_vxl,3)
        if enable_rayPooling:
            rayPooling_votes_list.append(rayPool_votes[slc][vxl_ijk_tuple].astype(np.uint8)) # (cube_Dcenter,)*3 --> (iN_voxel,)
        
    return nonempty_cube_indx, vxl_ijk_list, prediction_list, rgb_list, rayPooling_votes_list, param_new




def append_dense_2sparseList(prediction_sub, rgb_sub, param_sub, viewPair_sub, min_prob = 0.5, rayPool_thresh = 0, \
        enable_centerCrop = False, cube_Dcenter = None, \
        enable_rayPooling = False, cameraPOs = None, cameraTs = None, \
        prediction_list = [], rgb_list = [], vxl_ijk_list = [], rayPooling_votes_list = [], \
        cube_ijk_np = None, param_np = None, viewPair_np = None):
    """
    append the sparse lists/nps results to empty or non-empty lists/nps.
  
    --------------
    inputs:
        prediction_sub: np.float16(N_cubes,1,D,D,D)/(N_cubes,D,D,D)
        rgb_sub: np.uint8(N_cubes,3,D,D,D)
        param_sub: np.float32(N_cubes, N_params): 'ijk'/'xyz'/'resol'
        viewPair_sub: np.uint16(N_cubes, N_viewPairs, 2)
        min_prob = 0.5

        enable_centerCrop = False # used for center crop
        cube_Dcenter = None

        enable_rayPooling = False # used for rayPooling
        cameraPOs = None
        cameraTs = None

        prediction_list, rgb_list, vxl_ijk_list, rayPooling_votes_list: orignal lists before append
        cube_ijk_np, param_np, viewPair_np: orignal np before append

    --------------
    outputs:
        prediction_list, rgb_list, vxl_ijk_list, rayPooling_votes_list: updated lists after append
        cube_ijk_np, param_np, viewPair_np: updated np after append
    """

    if prediction_sub.ndim == 5:
        prediction_sub = prediction_sub.astype(np.float16)[:,0]  # (N,1,D,D,D)-->(N,D,D,D)
    rgb_sub = np.transpose(rgb_sub.astype(np.uint8), axes=(0,2,3,4,1)) #{N,3,D,D,D} --> {N,D,D,D,3}
    # finnally, only the xyz/resol/modelIndx will be stored. In case the entire param_sub will be saved in memory, we deep copy it.
    cube_ijk_sub = param_sub['ijk']
    viewPair_sub = viewPair_sub.astype(np.uint16) # (N,N_viewPairs,2)
    sparse_output = dense2sparse(prediction = prediction_sub, rgb = rgb_sub, param = param_sub,\
            viewPair = viewPair_sub, min_prob = min_prob, rayPool_thresh = rayPool_thresh,\
            enable_centerCrop = enable_centerCrop, cube_Dcenter = cube_Dcenter,\
            enable_rayPooling = enable_rayPooling, cameraPOs = cameraPOs, cameraTs = cameraTs)
    nonempty_cube_indx_sub, vxl_ijk_sub_list, prediction_sub_list, \
            rgb_sub_list, rayPooling_sub_votes_list, param_new_sub = sparse_output
    param_sub = param_new_sub[nonempty_cube_indx_sub]
    viewPair_sub = viewPair_sub[nonempty_cube_indx_sub]
    cube_ijk_sub = cube_ijk_sub[nonempty_cube_indx_sub]
    if not len(prediction_sub_list) == len(rgb_sub_list) == len(vxl_ijk_sub_list) == \
            param_sub.shape[0] == viewPair_sub.shape[0] == cube_ijk_sub.shape[0]:
        raise Warning('load dense data, # of cubes is not consistent.')
    prediction_list.extend(prediction_sub_list)
    rgb_list.extend(rgb_sub_list)
    vxl_ijk_list.extend(vxl_ijk_sub_list)
    rayPooling_votes_list.extend(rayPooling_sub_votes_list)
    param_np = param_sub if param_np is None else np.concatenate([param_np, param_sub], axis=0)  # np append / concatenate
    viewPair_np = viewPair_sub if viewPair_np is None else np.vstack([viewPair_np, viewPair_sub])
    cube_ijk_np = cube_ijk_sub if cube_ijk_np is None else np.vstack([cube_ijk_np, cube_ijk_sub])

    return prediction_list, rgb_list, vxl_ijk_list, rayPooling_votes_list, \
            cube_ijk_np, param_np, viewPair_np




def load_dense_as_sparse(files, cube_Dcenter, cameraPOs, min_prob=0.5, rayPool_thresh = 0):
    """
    load multiple dense cube voxels as sparse voxels data

    only reserve the voxels with prediction prob < min_prob
    --------------
    inputs:
        files: file names
        min_prob: 0.5
    --------------
    outputs:
        prediction_list[i]: np.float16 (iN_voxels,) # nth element corresponds to cube_n
        rgb_list[i]: np.uint8 (iN_voxels, 3)
        vxl_ijk_list[i]: np.uint8 (iN_voxels, 3)
        rayPooling_votes_list

        cube_ijk_np: np.uint16 (N,3)
        param_np: np.float32 (N,N_param)
        viewPair_np: np.uint16 (N,N_viewPairs,2)
    """
    prediction_list, rgb_list, vxl_ijk_list, rayPooling_votes_list = [], [], [], []
    cube_ijk_np, param_np, viewPair_np = None, None, None

    cameraT_folder = '/home/mengqi/dataset/MVS/cameraT/'
    cameraPO_folder = '/home/mengqi/dataset/MVS/pos/'

    # TODO: the new load_selected_POs hide the view index
    # cameraPOs = camera.load_selected_cameraPO_files_f64(dataset_name=param_volum.__datasetName, view_list=param_volum.__view_set)
    # cameraPOs = prepare_data.load_cameraPos_as_np(cameraPO_folder)
    cameraTs = camera.cameraPs2Ts(cameraPOs)


    for file_name in files: 
        print file_name
        try:
            with open(file_name) as f:
                npz_file = np.load(f)
                """
                prediction_sub: {N,1,D,D,D} float16 --> {N,D,D,D}
                rgb_sub: {N,3,D,D,D} uint8 --> {N,D,D,D,3}
                param_sub: {N,8} float64 # x,y,z,resol,modelIndx,indx_d0,indx_d1,indx_d2
                selected_viewPair_viewIndx_sub: {N, No_viewPairs, 2}
                """
                prediction_sub, rgb_sub, param_sub, viewPair_sub = \
                        npz_file["prediction"], npz_file["rgb"], npz_file["param"], npz_file["selected_pairIndx"] 
            prediction_list, rgb_list, vxl_ijk_list, rayPooling_votes_list, \
                    cube_ijk_np, param_np, viewPair_np = \
                    append_dense_2sparseList(prediction_sub = prediction_sub, rgb_sub = rgb_sub, param_sub = param_sub,\
                            viewPair_sub = viewPair_sub, min_prob = min_prob, rayPool_thresh = rayPool_thresh,\
                            enable_centerCrop = True, cube_Dcenter = cube_Dcenter,\
                            enable_rayPooling = True, cameraPOs = cameraPOs, cameraTs = cameraTs, \
                            prediction_list = prediction_list, rgb_list = rgb_list, vxl_ijk_list = vxl_ijk_list, \
                            rayPooling_votes_list = rayPooling_votes_list, \
                            cube_ijk_np = cube_ijk_np, param_np = param_np, viewPair_np = viewPair_np)
        except:
            print('Warning: this file not exist / valid')
    return prediction_list, rgb_list, vxl_ijk_list, rayPooling_votes_list, \
            cube_ijk_np, param_np, viewPair_np

def filter_voxels(vxl_mask_list=[],prediction_list=None, prob_thresh=None,\
        rayPooling_votes_list=None, rayPool_thresh=None):
    """
    thresholding using the prediction or rayPooling 
    consider the given vxl_mask_list

    ---------
    inputs:
        prediction_list[i]: np.float16 (iN_voxels,) # nth element corresponds to cube_n 
        prob_thresh: np.float16 scalar / list
        rayPooling_votes_list[i]: np.uint8 (iN_voxels,)
        rayPool_thresh: np.uint8, scalar
        vxl_mask_list[i]: np.bool (iN_voxels,)
    ---------
    outputs:
        vxl_mask_list[i]: np.bool (iN_voxels,)
    """
    empty_vxl_mask = True if len(vxl_mask_list) == 0 else False
    if prediction_list is not None:
        if prob_thresh is None:
            raise Warning('prob_thresh should not be None.')
        for _c, _prediction in enumerate(prediction_list):
            _prob_thresh = prob_thresh[_c] if isinstance(prob_thresh, list) else prob_thresh
            _surf = _prediction >= _prob_thresh
            if empty_vxl_mask:
                vxl_mask_list.append(_surf)
            else:
                vxl_mask_list[_c] &= _surf
    empty_vxl_mask = True if len(vxl_mask_list) == 0 else False
    if rayPooling_votes_list is not None:
        if rayPool_thresh is None:
            raise Warning('rayPool_thresh should not be None.')
        for _cube, _votes in enumerate(rayPooling_votes_list):
            _surf = _votes >= rayPool_thresh
            if empty_vxl_mask:
                vxl_mask_list.append(_surf)
            else:
                vxl_mask_list[_cube] &= _surf
    return vxl_mask_list


def save2ply(ply_filePath, xyz_np, rgb_np = None, normal_np = None):
    """
    save data to ply file, xyz (rgb, normal)

    ---------
    inputs:
        xyz_np: (N_voxels, 3)
        rgb_np: None / (N_voxels, 3)
        normal_np: None / (N_voxels, 3)

        ply_filePath: 'xxx.ply'
    outputs:
        save to .ply file
    """
    N_voxels = xyz_np.shape[0]
    atributes = [('x', '<f4'), ('y', '<f4'), ('z', '<f4')]
    if normal_np is not None:
        atributes += [('nx', '<f4'), ('ny', '<f4'), ('nz', '<f4')]
    if rgb_np is not None:
        atributes += [('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    saved_pts = np.zeros(shape=(N_voxels,), dtype=np.dtype(atributes))

    saved_pts['x'], saved_pts['y'], saved_pts['z'] = xyz_np[:,0], xyz_np[:,1], xyz_np[:,2] 
    if rgb_np is not None:
        saved_pts['red'], saved_pts['green'], saved_pts['blue'] = rgb_np[:,0], rgb_np[:,1], rgb_np[:,2]
    if normal_np is not None:
        saved_pts['nx'], saved_pts['ny'], saved_pts['nz'] = normal_np[:,0], normal_np[:,1], normal_np[:,2] 

    el_vertex = PlyElement.describe(saved_pts, 'vertex')
    outputFolder = os.path.dirname(ply_filePath)
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)
    PlyData([el_vertex]).write(ply_filePath)
    print('saved ply file: {}'.format(ply_filePath))
    return 1



def save_sparseCubes_2ply(vxl_mask_list, vxl_ijk_list, rgb_list, \
        param, ply_filePath, normal_list=None):
    """
    save sparse cube to ply file

    ---------
    inputs:
        vxl_mask_list[i]: np.bool (iN_voxels,)
        vxl_ijk_list[i]: np.uint8 (iN_voxels, 3)
        rgb_list[i]: np.uint8 (iN_voxels, 3)
        normal_list[i]: np.float16 (iN_voxels, 3)

        param: np.float32(N_nonempty_cubes, 4)
        ply_filePath: 'xxx.ply'
    outputs:
        save to .ply file
    """
    vxl_mask_np = np.concatenate(vxl_mask_list, axis=0) 
    N_voxels = vxl_mask_np.sum()
    vxl_ijk_np = np.vstack(vxl_ijk_list)
    rgb_np = np.vstack(rgb_list)
    if not vxl_mask_np.shape[0] == vxl_ijk_np.shape[0] == rgb_np.shape[0]:
        raise Warning('make sure # of voxels in each cube are consistent.')
    if normal_list is None:
        dt = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'), \
                ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
        normal_np = None
    else:
        dt = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'), \
                ('nx', '<f4'), ('ny', '<f4'), ('nz', '<f4'), \
                ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
        normal_np = np.vstack(normal_list)[vxl_mask_np]
    saved_pts = np.zeros(shape=(N_voxels,), dtype=dt)

    # calculate voxels' xyz 
    xyz_list = []
    for _cube, _select in enumerate(vxl_mask_list):
        resol = param[_cube]['resol']
        xyz_list.append(vxl_ijk_list[_cube][_select] * resol + param[_cube]['xyz'][None,:]) # (iN, 3) + (1, 3)
    xyz_np = np.vstack(xyz_list)
    rgb_np = rgb_np[vxl_mask_np]
    save2ply(ply_filePath, xyz_np, rgb_np, normal_np)
    return 1



def save_sparseCubes(filePath, \
        prediction_list, rgb_list, vxl_ijk_list, rayPooling_votes_list, \
        cube_ijk_np, param_np, viewPair_np):
    """
    save sparse cube voxels using numpy!

    --------------
    inputs:
        prediction_list[i]: np.float16 (iN_voxels,) # nth element corresponds to cube_n
        rgb_list[i]: np.uint8 (iN_voxels, 3)
        vxl_ijk_list[i]: np.uint8 (iN_voxels, 3)
        rayPooling_votes_list[i]: np.uint8 (iN_voxels,)

        cube_ijk_np: np.uint16 (N,3)
        param_np: np.float32 (N,N_param)
        viewPair_np: np.uint16 (N,N_viewPairs,2)    
    --------------
    outputs:
    """
    prediction_np = np.concatenate(prediction_list, axis=0)
    rgb_np = np.vstack(rgb_list)
    vxl_ijk_np = np.vstack(vxl_ijk_list)
    rayPooling_votes_np = np.empty((0,), np.uint8) if len(rayPooling_votes_list) == 0 else \
            np.concatenate(rayPooling_votes_list, axis=0) 

    N_cube = cube_ijk_np.shape[0]
    # cube_1st_vxlIndx_np: record the start voxel index of ith cube in the (i+1)th position, in order to recover the lists.
    cube_1st_vxlIndx_np = np.zeros((N_cube+1,)).astype(np.uint32)      
    for _n_cube, _prediction in enumerate(prediction_list):
        cube_1st_vxlIndx_np[_n_cube + 1] = _prediction.size + cube_1st_vxlIndx_np[_n_cube] 
    if not cube_1st_vxlIndx_np[-1] == prediction_np.shape[0] == rgb_np.shape[0] == vxl_ijk_np.shape[0]:
        raise Warning("# of voxels is not consistent while saving sparseCubes.")
    with open(filePath, 'wb') as f:
        np.savez_compressed(f, cube_1st_vxlIndx_np = cube_1st_vxlIndx_np, prediction_np = prediction_np, \
                rgb_np = rgb_np, vxl_ijk_np = vxl_ijk_np, rayPooling_votes_np = rayPooling_votes_np, \
                cube_ijk_np = cube_ijk_np, param_np = param_np, viewPair_np = viewPair_np)
        print("saved sparseCubes to file: {}".format(filePath))


def load_sparseCubes(filePath):
    """
    load sparse cube voxels from saved numpy npz!

    --------------
    outputs:
        prediction_list[i]: np.float16 (iN_voxels,) # nth element corresponds to cube_n
        rgb_list[i]: np.uint8 (iN_voxels, 3)
        vxl_ijk_list[i]: np.uint8 (iN_voxels, 3)
        rayPooling_votes_list[i]: np.uint8 (iN_voxels,)

        cube_ijk_np: np.uint16 (N,3)
        param_np: np.float32 (N,N_param)
        viewPair_np: np.uint16 (N,N_viewPairs,2)    
    """
    with open(filePath) as f:
        npz = np.load(f)
        cube_1st_vxlIndx_np, prediction_np, rgb_np, vxl_ijk_np, rayPooling_votes_np, cube_ijk_np, param_np, viewPair_np = \
                npz['cube_1st_vxlIndx_np'], npz['prediction_np'], npz['rgb_np'], npz['vxl_ijk_np'], npz['rayPooling_votes_np'], \
                npz['cube_ijk_np'], npz['param_np'], npz['viewPair_np']
        print("loaded sparseCubes to file: {}".format(filePath))
    
    if not cube_1st_vxlIndx_np[-1] == prediction_np.shape[0] == rgb_np.shape[0] == vxl_ijk_np.shape[0]:
        raise Warning("# of voxels is not consistent while saving sparseCubes.")
    if not rayPooling_votes_np.shape[0] in [0, cube_1st_vxlIndx_np[-1]]:
        raise Warning("rayPooling_votes_np.shape[0] != 0 / # of voxels.")
 
    prediction_list, rgb_list, vxl_ijk_list, rayPooling_votes_list = [], [], [], []
    N_cube = cube_ijk_np.shape[0]
    for _n_cube in range(N_cube):
        slc = np.s_[cube_1st_vxlIndx_np[_n_cube]: cube_1st_vxlIndx_np[_n_cube + 1]]
        prediction_list.append(prediction_np[slc])
        rgb_list.append(rgb_np[slc])
        vxl_ijk_list.append(vxl_ijk_np[slc])
        rayPooling_votes_list.append(rayPooling_votes_np[slc])
    return prediction_list, rgb_list, vxl_ijk_list, rayPooling_votes_list, \
        cube_ijk_np, param_np, viewPair_np
    





def __debug():
    # N_batchs=75
    # tmp = 8
    # batch_range = range(10, 15) # (13,16) / (0,N_batchs)
    # dataFolder='/home/mengqi/dataset/MVS/lasagne/save_reconstruction_result/saved_prediction_rgb_params_modelBB/model17-3viewPairs-resol0.400-strideRatio0.500'
    # file_path_lambda = lambda i: os.path.join(dataFolder,'batch-{}_{}.npz'.format(i,N_batchs))
    # outputs = load_dense_as_sparse(cube_Dcenter=26, rayPool_thresh=0, min_prob=0.5,files=[file_path_lambda(_b) for _b in batch_range])
    # npz_file = '/home/mengqi/tmp/{}.npz'.format(tmp)
    # save_sparseCubes(npz_file, *outputs)
    # # with open('/home/mengqi/tmp/{}.pkl'.format(tmp),'w') as f:
    # #     pickle.dump(outputs, f)
    # #     print('saved {}'.format(f.name))

    # # with open('/home/mengqi/tmp/{}.pkl'.format(tmp)) as f:
    # #     data = pickle.load(f)
    # #     print('loaded {}'.format(f.name))
    # data = load_sparseCubes(npz_file)
    # prediction_list, rgb_list, vxl_ijk_list, rayPooling_votes_list, \
    #         cube_ijk_np, param_np, viewPair_np = data
    # prob_thresh_list = [0.6] * len(prediction_list)
    # vxl_mask_list = filter_voxels(vxl_mask_list=[],prediction_list=prediction_list, prob_thresh=prob_thresh_list,\
    #         rayPooling_votes_list=rayPooling_votes_list, rayPool_thresh=5)
        
    # save_sparseCubes_2ply(vxl_mask_list, vxl_ijk_list, rgb_list, \
    #         param_np, ply_filePath='/home/mengqi/tmp/{}.ply'.format(tmp), normal_list=None)
    pass        




