import numpy as np
import os
import sys
import math
import scipy.io

sys.path.append("./utils")
import scene

## define the parameters that don't vary in the main function, such as the patch size and the cube size.
## for the parameters that may change in the different main loop through different models in a dataset, we load them use the function `load_modelSpecific_params` defined at the end of this file.

# "reconstruct_model"
whatUWant = "reconstruct_model"

__datasetName = 'Middlebury'  # Middlebury / DTU / people
__GPUMemoryGB = 12  # how large is your GPU memory (GB)
__input_data_rootFld = "./inputs"
__output_data_rootFld = "./outputs"


###########################
#   several modes:
#   "train_xxx" "reconstruct_model"
###########################
__DEBUG_input_data_rootFld = "/home/mengqi/fileserver/datasets"     # used for debug: if exists, use this path
__DEBUG_output_data_rootFld = "/home/mengqi/fileserver/results/MVS/SurfaceNet"
__DEBUG_input_data_rootFld_exists = os.path.exists(__DEBUG_input_data_rootFld)
__DEBUG_output_data_rootFld_exists = os.path.exists(__DEBUG_output_data_rootFld)
__input_data_rootFld = __DEBUG_input_data_rootFld if __DEBUG_input_data_rootFld_exists else __input_data_rootFld
__output_data_rootFld = __DEBUG_output_data_rootFld if __DEBUG_output_data_rootFld_exists else __output_data_rootFld

debug_BB = False
__output_data_rootFld += '_Debug_BB' if debug_BB else ''


if whatUWant is "reconstruct_model": 
    """
    In this mode, reconstruct models using the similarityNet and SurfaceNet.
    """
    #------------ 
    ## params only for reconstruction
    #------------
    # DTU: numbers: 1 .. 128
    # Middlebury: dinoSparseRing
    if __datasetName is 'DTU':
        __modelList = [9]     # [3,18,..]
    elif __datasetName is 'Middlebury':
        __modelList = ["dinoSparseRing"]     # ["dinoSparseRing", "..."]
    elif __datasetName is 'people':
        # frame 0: ["head", "render_07"]
        # frame 1: ["model_20_anim_4", "model_42_anim_8", "model_42_anim_9", "render_11"]
        # frame 11: ["model_20_anim_4", "model_42_anim_8", "model_42_anim_9"]  # may have different resol
        # frame 30: ["T_samba"]
        # frame 40: ["T_samba"]
        # frame 50: ["flashkick", "I_crane"]
        # frame 70: ["T_samba"]  # resol: 0.005
        # frame 100: ["pop", "I_crane"]
        # frame 120: ["I_crane"]
        __frame = 100  # [0, 50, 100, 150]
        __modelList = ["pop", "I_crane"]     # ["D_bouncing", "T_samba", "..."]
        __viewList = range(1, 5)  # range(1, 5)
        __output_data_rootFld = os.path.join(__output_data_rootFld, "people/frame{}_views{}".format(__frame, __viewList))

    __cube_D = 64 #32/64 # size of the CVC = __cube_D ^3, in the paper it is (s,s,s)
    __min_prob = 0.46 # in order to save memory, filter out the voxels with prob < min_prob
    __tau = 0.7     # fix threshold for thinning
    __gamma = 0.8   # used in the ray pooling procedure

    # TODO tune, gpuarray.preallocate=0.95 / -1 
    __batchSize_similNet_patch2embedding_perGB = 100
    __batchSize_similNet_embeddingPair2simil_perGB = 100000
    __batchSize_viewPair_w_perGB = 100000     


    __batchSize_similNet_patch2embedding, __batchSize_similNet_embeddingPair2simil, __batchSize_viewPair_w = np.array([\
            __batchSize_similNet_patch2embedding_perGB, \
            __batchSize_similNet_embeddingPair2simil_perGB, \
            __batchSize_viewPair_w_perGB, \
            ], dtype=np.uint64) * __GPUMemoryGB

    ############# 
    ## similarNet
    #############

    # each patch pair --> features to learn to decide view pairs
    # 2 * 128D/image patch + 1 * (dis)similarity + 1 * angle<v1,v2>
    __D_imgPatchEmbedding = 128
    __D_viewPairFeature = __D_imgPatchEmbedding * 2 + 1 + 1     # embedding / view pair angle / similarity
    __similNet_hidden_dim = 100
    __pretrained_similNet_model_file = os.path.join(__input_data_rootFld, 'SurfaceNet_models/epoch33_acc_tr0.707_val0.791.model') # allDTU
    __imgPatch_hw_size = 64
    __MEAN_IMAGE_BGR = np.asarray([103.939,  116.779,  123.68]).astype(np.float32)
    __triplet_alpha = 100
    __weight_decay = 0.0001
    __DEFAULT_LR = 0 # will be updated during param tuning

    ############
    # SurfaceNet
    ############

    # view index of the considered views
    __use_pretrained_model = True
    if __use_pretrained_model:
        __layerNameList_2_load = ["output_SurfaceNet_reshape","output_softmaxWeights"] ##output_fusionNet/fuse_op_reshape
        __pretrained_SurfaceNet_model_file = os.path.join(__input_data_rootFld, 'SurfaceNet_models/2D_2_3D-19-0.918_0.951.model') # allDTU
    __cube_Dcenter = {32:26, 64:52}[__cube_D] # only keep the center part of the cube because of boundary effect of the convNet.

    ####################
    # adaptive threshold
    ####################
    __beta = 6
    __N_refine_iter = 8
    __cube_overlapping_ratio = 1/2. ## how large area is covered by the neighboring cubes. 
    __weighted_fusion = True # True: weighted average in the fusion layer; False: average

    __batchSize_nViewPair_SurfaceNet_perGB = {32:1.2, 64:0.1667}[__cube_D]  # 0.1667 = 1./6
    __batchSize_nViewPair_SurfaceNet = int(math.floor(__batchSize_nViewPair_SurfaceNet_perGB * __GPUMemoryGB))
  


elif whatUWant is "train_xxx":
    pass


###########################
#   params rarely change
###########################
__MEAN_CVC_RGBRGB = np.asarray([123.68,  116.779,  103.939, 123.68,  116.779,  103.939]).astype(np.float32) # RGBRGB order (VGG mean)
__MEAN_PATCHES_BGR = np.asarray([103.939,  116.779,  123.68]).astype(np.float32)




## print the params in log
for _var in dir():
    if '__' in _var and not (_var[-2:] == '__'): # don't show the uncustomed variables, like '__builtins__'
        exec("print '{} = '.format(_var), "+_var)



def load_modelSpecific_params(datasetName, model):
    """
    In order to loop through the different models in a same dataset.
    This function only assign different params associated with different model in each reconstrction loop.
    ----------
    inputs:
        datasetName: such as "DTU" / "Middlebury" / "people" ... 
        model: such as 3 / "dinoSparseRing" / ...
    outputs:
        datasetFolder: root folder of this dataset
        imgNamePattern: pattern of the img path, replace # to view index
        poseNamePattern: 
        N_viewPairs4inference: how many viewPairs used for reconstruction
        resol: size of voxel
        BB: Bounding Box of this scene.  np(2,3) float32
        viewList: which views are used for reconstruction.
    """
    initialPtsNamePattern = None  # if defined, the cubes position will be initialized according to these points that will be quantizated by $resolution$

    if datasetName is "DTU":
        datasetFolder = os.path.join(__input_data_rootFld, 'DTU_MVS')
        imgNamePattern = "Rectified/scan{}/rect_#_3_r5000.{}".format(model, 'png' if __DEBUG_input_data_rootFld_exists else 'jpg')    # replace # to {:03} 
        poseNamePattern = "SampleSet/MVS Data/Calibration/cal18/pos_#.txt"  # replace # to {:03}
        N_viewPairs4inference = [5]
        resol = np.float32(0.4) #0.4 resolution / the distance between adjacent voxels
        BBNamePattern = "SampleSet/MVS Data/ObsMask/ObsMask{}_10.mat".format(model)
        BB_filePath = os.path.join(datasetFolder, BBNamePattern)
        BB_matlab_var = scipy.io.loadmat(BB_filePath)   # matlab variable
        reconstr_sceneRange = np.asarray([(-40, 40), (80, 160), (630, 680)])
        BB = reconstr_sceneRange if debug_BB else BB_matlab_var['BB'].T   # np(3,2)
        viewList = range(1,50)  # range(1,50)

    if datasetName is "Middlebury":
        datasetFolder = os.path.join(__input_data_rootFld, 'Middlebury')
        N_viewPairs4inference = [3]
        resol = np.float32(0.00025) # 0.00025 resolution / the distance between adjacent voxels
        if model is "dinoSparseRing":
            imgNamePattern = "{}/dinoSR0#.png".format(model)   # replace # to {:03}
            poseNamePattern = "{}/dinoSR_par.txt".format(model)
            BB = np.array([(-0.061897, 0.010897), (-0.018874, 0.068227), (-0.057845, 0.015495)], dtype=np.float32)   # np(3,2)
            viewList = range(7,13) #range(1,16)
        else:
            raise Warning('current model is unexpected: '+model+'.') 

    if datasetName is "people":
        # people dataset website: http://people.csail.mit.edu/drdaniel/mesh_animation/
        datasetFolder = os.path.join(__input_data_rootFld, 'people/samples/mit_format_mvs_example_data_4')
        imgNamePattern = "{}/images/Image@_{:04}.png".format(model, __frame)   # replace # to {:03}, relace @ to {}
        poseNamePattern = "{}/calibration/Camera@.Pmat.cal".format(model)
        BBNamePattern = "{}/meshes/mesh_{:04}.obj".format(model, __frame)
        N_viewPairs4inference = [2]
        resol = np.float32(0.005) # resolution / the distance between adjacent voxels
        BB = scene.readBB_fromModel(objFile = os.path.join(datasetFolder, BBNamePattern)) # None / np.array([(0.2091, 0.5904), (0.0327, 1.7774), (-0.3977, 0.3544)], dtype=np.float32)   # np(3,2)
        initialPtsNamePattern = None # None / "{}/visualHull/vhull_4_views_1346/{:04}.ply".format(model, __frame)
        viewList = __viewList # range(1,5) # range(4,9) + [1] #range(1,9)
    return datasetFolder, imgNamePattern, poseNamePattern, initialPtsNamePattern, N_viewPairs4inference, resol, BB, viewList


