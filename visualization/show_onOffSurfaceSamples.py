import sys
import os
import numpy as np
from plyfile import PlyData, PlyElement

sys.path.append("../utils")
import sparseCubes
import pointCloud
import utils


def save_sparseCubes_2ply(vxl_ijk_list, rgb_list, \
        cube_param, ply_filePath, normal_list=None):
    """
    save sparse cube to ply file

    ---------
    inputs:
        vxl_ijk_list[i]: np.uint8 (iN_voxels, 3)
        normal_list[i]: np.float16 (iN_voxels, 3)

        cube_param: np.float32(N_nonempty_cubes, 4)
        ply_filePath: 'xxx.ply'
    outputs:
        save to .ply file
    """
    vxl_ijk_np = np.vstack(vxl_ijk_list)
    N_voxels = vxl_ijk_np.shape[0]
    rgb_np = np.vstack(rgb_list)
    if normal_list is None:
        dt = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'), \
                ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    else:
        dt = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'), \
                ('nx', '<f4'), ('ny', '<f4'), ('nz', '<f4'), \
                ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
        normal_np = np.vstack(normal_list)
    saved_pts = np.zeros(shape=(N_voxels,), dtype=dt)

    # calculate voxels' xyz 
    xyz_list = []
    N_cubes = len(vxl_ijk_list)
    for _cube in range(N_cubes):
        resol = cube_param[_cube]['resolution']
        xyz_list.append(vxl_ijk_list[_cube] * resol + cube_param[_cube]['min_xyz'][None,:]) # (iN, 3) + (1, 3)
    xyz_np = np.vstack(xyz_list)

    saved_pts['x'], saved_pts['y'], saved_pts['z'] = xyz_np[:,0], xyz_np[:,1], xyz_np[:,2] 
    saved_pts['red'], saved_pts['green'], saved_pts['blue'] = rgb_np[:,0], rgb_np[:,1], rgb_np[:,2]
    if normal_list is not None:
        saved_pts['nx'], saved_pts['ny'], saved_pts['nz'] = normal_np[:,0], normal_np[:,1], normal_np[:,2] 

    el_vertex = PlyElement.describe(saved_pts, 'vertex')
    outputFolder = '/'.join(ply_filePath.split('/')[:-1])
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)
    PlyData([el_vertex]).write(ply_filePath)
    print('saved ply file: {}'.format(ply_filePath))
    return 1



def show_onOffSurfaceSamples(cube_param, vxl_ijk_list, density_list, save_result_fld, \
        cubesRatio_2showBorder = 0.02, rgb_list = None, printDetails = False):
    """
    In order to check the cube range of a loaded '.npz', add some voxels at the border of selected/random cubes.

    -------
    inputs:

    ------
    examples:
    """

    N_cubes = len(vxl_ijk_list)
    if rgb_list is None:
        rgb_list = [None for _ in range(N_cubes)]
    cubes_2showBorder = np.random.rand(N_cubes) < cubesRatio_2showBorder  # bool

    for _cube in range(N_cubes):
        cube_D = cube_param[_cube]['cube_D']

        # generate vxls on the cube borders
        cubes_3MinEdges = ((np.eye(3).astype(np.uint32)[:,None,:] * np.arange(1, cube_D-1, 3)[None,:,None])).reshape((-1, 3))  # (3,3,1) * (1,1,N_edgeVxl) --> (3*N_edgeVxl, 3)
        cubes_3MaxEdges = np.copy(cubes_3MinEdges)
        cubes_3MaxEdges[cubes_3MaxEdges == 0] = cube_D - 1  # (3*N_edgeVxl, 3)
        cubes_6edges = np.vstack([cubes_3MinEdges, cubes_3MaxEdges]) # (6*N_edgeVxl, 3)
        N_appendedVxls = cubes_6edges.shape[0]

        N_vxls = len(vxl_ijk_list[_cube])
        if N_vxls == 0:
            vxl_ijk_list[_cube] = np.zeros((0, 3)).astype(np.uint32)
        rgb_list[_cube] = (np.ones((N_vxls, 3)) * np.random.randint(50,255, (1, 1))).astype(np.uint8)  # rainbow

        if _cube in np.where(cubes_2showBorder)[0]:  # show the edges of this cube
            if N_vxls == 0:
                edge_color = np.zeros((1,3)).astype(np.uint8)   # black
            else:
                edge_color = np.random.randint(50,255, (1,3)).astype(np.uint8)
            vxl_ijk_list[_cube] = np.append(vxl_ijk_list[_cube], cubes_6edges, axis=0)
            rgb_list[_cube] = np.append(rgb_list[_cube], np.repeat(edge_color, N_appendedVxls, axis=0), axis=0)
            if printDetails:
                print("{{cube_param: {}}}, {{color: {}}}".format(cube_param[_cube], edge_color))

    utils.mkdirs_ifNotExist(save_result_fld)
    save_sparseCubes_2ply(vxl_ijk_list = vxl_ijk_list, rgb_list = rgb_list, cube_param = cube_param, \
            ply_filePath= os.path.join(save_result_fld, 'showCubesBorder.ply'), normal_list=None) 
    return 0



if __name__ == '__main__':

    # load the saved file from the func: pointCloud.save_surfacePts_2file
    # used to varify the saved data (cropped cubes) for training
    save_npz_file_path  = '/home/mengqi/temp/prepare/model009.npz' 
    cube_param, vxl_ijk_list, density_list = pointCloud.read_saved_surfacePts(save_npz_file_path)

    show_onOffSurfaceSamples(cube_param, vxl_ijk_list, density_list, save_result_fld = os.path.dirname(save_npz_file_path), cubesRatio_2showBorder = 0.1, printDetails = True)

