import sys
import os
import numpy as np

sys.path.append("../utils")
import sparseCubes


def show_onOffSurfaceSamples(npz_file, save_result_fld, D_cube, \
        init_probThresh, cubeIndexes2visualize = None, cubesRatio = 0.02, printDetails = True):
    """
    In order to check the cube range of a loaded '.npz', add some voxels at the border of selected/random cubes.

    -------
    inputs:
        npz_file, save_result_fld, D_cube, \
        init_probThresh, cubesRatio = 0.2

    ------
    examples:
    >> save_npz_file_path  = '/home/mengqi/fileserver/results/MVS/SurfaceNet/DTU_s64/1_49views_Nv5_resol0.4/49views_resolLevel0.npz'
    >> showCubesBorder(npz_file = save_npz_file_path, save_result_fld = os.path.dirname(save_npz_file_path), D_cube = 52, \
                        init_probThresh = 0.5, cubesRatio = 1)   # {32:26, 64:52}
    """
    data = sparseCubes.load_sparseCubes(npz_file)
    prediction_list, rgb_list, vxl_ijk_list, rayPooling_votes_list, \
            cube_ijk_np, param_np, viewPair_np = data

    vxl_mask_list = sparseCubes.filter_voxels(vxl_mask_list=[],prediction_list=prediction_list, prob_thresh=init_probThresh)

    N_cubes = len(vxl_mask_list)
    if cubeIndexes2visualize is None:
        shownCubes = np.random.rand(N_cubes) < cubesRatio  # bool
    else: 
        shownCubes = cubeIndexes2visualize

    cubes_3MinEdges = ((np.eye(3).astype(np.uint32)[:,None,:] * np.arange(1, D_cube-1, 3)[None,:,None])).reshape((-1, 3))  # (3,3,1) * (1,1,N_edgeVxl) --> (3*N_edgeVxl, 3)
    cubes_3MaxEdges = np.copy(cubes_3MinEdges)
    cubes_3MaxEdges[cubes_3MaxEdges == 0] = D_cube - 1  # (3*N_edgeVxl, 3)
    cubes_6edges = np.vstack([cubes_3MinEdges, cubes_3MaxEdges]) # (6*N_edgeVxl, 3)
    N_appendedVxls = cubes_6edges.shape[0]
    for _cube in np.where(shownCubes)[0]:

        vxl_ijk_list[_cube] = np.append(vxl_ijk_list[_cube], cubes_6edges, axis=0)
        rand_color = np.random.randint(50,255, (1,3)).astype(np.uint8)
        rgb_list[_cube] = np.append(rgb_list[_cube], np.repeat(rand_color, N_appendedVxls, axis=0), axis=0)
        vxl_mask_list[_cube] = np.append(vxl_mask_list[_cube], np.ones((N_appendedVxls,), dtype=np.bool), axis=0)
        if printDetails:
            print("{{cube_ijk: {}}}, {{cube_param: {}}}, {{color: {}}}".format(cube_ijk_np[_cube], param_np[_cube], rand_color))

    if not os.path.exists(save_result_fld):
        os.makedirs(save_result_fld)

    sparseCubes.save_sparseCubes_2ply(vxl_mask_list, vxl_ijk_list, rgb_list, param_np, \
            ply_filePath=npz_file.replace('.npz', '-showCubesBorder.ply'), normal_list=None) 
    return 0



if __name__ == '__main__':
    datasetName = 'Middlebury'
    model = 'dinoSparseRing'
    viewIndex = 11
    cube_D_center, tau, gamma, N_viewPairs4inference = 52, 0.7, 0.8, 3
    init_probThresh, cubeIndexes2visualize = 0.5, [1,2, 3]
    save_npz_file_path  = '/home/mengqi/fileserver/results/MVS/SurfaceNet/Middlebury_s64/dinoSparseRing_15views_Nv3_resol0.00025/15views_1-Finest-1023cubes.npz'

    showCubesBorder(npz_file = save_npz_file_path, save_result_fld = os.path.dirname(save_npz_file_path), D_cube = cube_D_center, \
            init_probThresh = init_probThresh, cubeIndexes2visualize = cubeIndexes2visualize, printDetails = True)   # {32:26, 64:52}

