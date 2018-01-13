import os
import numpy as np
from numpy.lib.recfunctions import append_fields

import utils
import pointCloud


def __generate_or_load_surfacePts__(modelIndex, npzFile_pattern, modelFile_pattern = None, N_pts_onOffSurface = [100, 100], cube_D = 50, inputDataType = 'pcd', cube_resolutionList = [0.8, 0.4, 0.2], density_dtype = 'float'):
    """
    Given model index and folder to load the voxel informations. 
    If not exist, generate and save.

    -----------
    inputs:
        density_dtype: determine the returned value of density
            'float': float values with max density = 1 / 0; 
            'bool': whether there are more than 1 point in the voxel
            'uint': only return the NO. of pts in the voxel
    """

    npzFile = npzFile_pattern.replace('#', "{:03}-{}on_{}off".format(modelIndex, N_pts_onOffSurface[0], N_pts_onOffSurface[1]))
    if os.path.exists(npzFile):  # load from the saved file
        cube_param, vxl_ijk_list, density_list = pointCloud.read_saved_surfacePts(npzFile)
    else:   # if was not saved, generate new and save
        modelFile = modelFile_pattern.replace("#", "{:03}".format(modelIndex))
        cube_param, vxl_ijk_list, density_list = pointCloud.save_surfacePts_2file(inputFile = modelFile, \
                outputFile = npzFile, \
                N_pts_onSurface = N_pts_onOffSurface[0], \
                N_pts_offSurface = N_pts_onOffSurface[1], \
                cube_D = cube_D, \
                cube_resolutionList = cube_resolutionList, \
                inputDataType = inputDataType, \
                density_dtype = density_dtype)
    return cube_param, vxl_ijk_list, density_list


def load_sparse_surfacePts_asnp(modelIndexList, modelFile_pattern, npzFile_pattern, \
        N_pts_onOffSurface = [100, 100], cube_D = 50, inputDataType = 'pcd', \
        cube_resolutionList = [0.8, 0.4, 0.2], density_dtype = 'float'):
    """
    load model

    -----------
    inputs:
        density_dtype: determine the returned value of density
            'float': float values with max density = 1 / 0; 
            'bool': whether there are more than 1 point in the voxel
            'uint': only return the NO. of pts in the voxel

    outputs:
        cube_param_list, 
        vxl_ijk_list, 
        density_list
    """

    cube_param_list, vxl_ijk_list, density_list = [], [], []
    for _ith_model, _modelIndex in enumerate(modelIndexList):
        # cube params: min_xyz / resolution / cube_D
        _cube_param, _vxl_ijk_list, _density_list = __generate_or_load_surfacePts__(_modelIndex, npzFile_pattern = npzFile_pattern, \
                modelFile_pattern = modelFile_pattern, \
                N_pts_onOffSurface = N_pts_onOffSurface, \
                cube_D = cube_D, \
                inputDataType = inputDataType, \
                cube_resolutionList = cube_resolutionList, \
                density_dtype = density_dtype)

        # cube params: min_xyz / resolution / cube_D / modelIndex
        _cube_param = append_fields(_cube_param, 'modelIndex', dtypes = np.uint8, data = (np.ones(_cube_param.shape) * _ith_model), usemask=False)  # append field to structured array

        cube_param_list.append(_cube_param)
        vxl_ijk_list += list(_vxl_ijk_list)
        density_list += list(_density_list)
    cube_param = np.concatenate(cube_param_list, axis=0)
    return cube_param, vxl_ijk_list, density_list


