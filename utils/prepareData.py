import os

import utils
import pointCloud




if __name__ == '__main__':
    modelIndex = 9
    cube_D, cube_resolutionList, N_pts_onOffSurface = 50, [0.8, 0.4, 0.2], [100, 100]
    pcdFile = "/home/mengqi/fileserver/datasets/DTU_MVS/Points/stl/stl{:03}_total.ply".format(modelIndex)
    outputFolder = "/home/mengqi/temp/on_off_surfacePts"
    outputFile = os.path.join(outputFolder, "model{:03}.npz".format(modelIndex))
    utils.mkdirs_ifNotExist(outputFolder)
    cube_param, vxl_ijk_list, density_list = pointCloud.save_surfacePts_2file(inputFile = pcdFile, \
            outputFile = outputFile, \
            N_pts_onSurface = N_pts_onOffSurface[0], \
            N_pts_offSurface = N_pts_onOffSurface[1], \
            cube_D = cube_D, \
            cube_resolutionList = cube_resolutionList, \
            inputDataType = 'pcd')
    print cube_param.shape

