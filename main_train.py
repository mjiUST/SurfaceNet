import os
import sys
import random
import numpy as np
import time
from prefetch_generator import BackgroundGenerator, background  # https://github.com/justheuristic/prefetch_generator


import params
sys.path.append("./utils")
import CVC
import image
import camera
import prepareData
import sparseCubes
sys.path.append("./nets")
import SurfaceNet
import SimilarityNet


def load_dnn_fns(with_relativeImpt, SurfaceNet_model_path = None):
    """
    define / load all the dnn functions for training / finetuning
    """

    # define or load SurfaceNet
    train_fn, val_fn, lr_tensor = SurfaceNet.SurfaceNet_trainVal(with_relativeImpt, pretrained_model_path = SurfaceNet_model_path)

    # TODO: define and load SimilarityNet

    return {'train_fn': train_fn, 'val_fn': val_fn, 'lr_tensor': lr_tensor}


def loadFixedVar_4training():
    """
    Load camera params ..., that don't change during training
    Images are loaded while being used to save memory!
    """

    cameraPOs_np = camera.readCameraPOs_as_np(datasetFolder = params.__datasetFolder, datasetName = params.__datasetName, poseNamePattern = params.__poseNamePattern, viewList = params.__viewList)  # (N_views, 3, 4) np
    cameraTs_np = camera.cameraPs2Ts(cameraPOs = cameraPOs_np)  # (N_views, 3) np

    return {'cameraPOs_np': cameraPOs_np, 'cameraTs_np': cameraTs_np}


def load_sparseSurfacePts(N_onSurfacePts_train, N_offSurfacePts_train, N_onSurfacePts_val, N_offSurfacePts_val, cube_D_loaded):
    """
    load candidate on/off surface cubes
    """

    cube_param_train, vxl_ijk_list_train, density_list_train = prepareData.load_sparse_surfacePts_asnp( \
            modelIndexList = params.__modelList_train, \
            modelFile_pattern = os.path.join(params.__datasetFolder, params.__modelFile_pattern), \
            npzFile_pattern = os.path.join(params.__input_data_rootFld, params.__npzFile_pattern), \
            N_pts_onOffSurface = [N_onSurfacePts_train, N_offSurfacePts_train], cube_D = cube_D_loaded, inputDataType = 'pcd', \
            cube_resolutionList = [0.8, 0.4, 0.2], density_dtype = 'float' if params.__soft_label else 'bool')

    cube_param_val, vxl_ijk_list_val, density_list_val = prepareData.load_sparse_surfacePts_asnp( \
            modelIndexList = params.__modelList_val, \
            modelFile_pattern = os.path.join(params.__datasetFolder, params.__modelFile_pattern), \
            npzFile_pattern = os.path.join(params.__input_data_rootFld, params.__npzFile_pattern), \
            N_pts_onOffSurface = [N_onSurfacePts_val, N_offSurfacePts_val], cube_D = cube_D_loaded, inputDataType = 'pcd', \
            cube_resolutionList = [0.8, 0.4, 0.2], density_dtype = 'float' if params.__soft_label else 'bool')

    return {'cube_param_train': cube_param_train, 'vxl_ijk_list_train': vxl_ijk_list_train, 
            'density_list_train': density_list_train, 'cube_param_val': cube_param_val, 
            'vxl_ijk_list_val': vxl_ijk_list_val, 'density_list_val': density_list_val, 
            'cube_D_loaded': cube_D_loaded}


@background(max_prefetch=2)
def iterate_minibatches(N_batches, batchSize, N_viewPairs, cube_param, cameraPOs_np, images_list,
        dense_gt):
    """
    fetch minibatches
    """

    N_cubes = cube_param.shape[0]
    for _n in range(N_batches):
        selector = random.sample(range(N_cubes), batchSize) # randomly select = shuffle the samples
        # generate CVC
        rand_viewPairs = np.random.randint(0, len(params.__viewList), (batchSize, N_viewPairs, 2)) # (params.__chunk_len, N_viewPair, 2) randomly select viewPairs for each cube
        # dtype = uint8
        _CVCs1_sub = CVC.gen_models_coloredCubes( \
                viewPairs = rand_viewPairs,  \
                cube_params = cube_param[selector], \
                cameraPOs = cameraPOs_np, \
                models_img_list = images_list, \
                cube_D = cube_param['cube_D'][0] \
                ) # ((N_cubeSub * __N_viewPairs4train, 3 * 2) + (D_CVC,) * 3) 5D
        _gt_sub = dense_gt[selector][:, None]  # (N, D,D,D) --> (N, 1, D,D,D)
        _gt_sub, _CVCs2_sub = CVC.preprocess_augmentation(_gt_sub, _CVCs1_sub, mean_rgb = params.__MEAN_CVC_RGBRGB[None,:,None,None,None], augment_ON=True, crop_ON = True, cube_D = params.__cube_D)


        yield _CVCs2_sub, _gt_sub



def iterate_models_and_lightConditions(N_epoches, N_randomModels, modelList, lightConditions,
        N_on_off_surfacePts, cube_D_loaded, random_model = True, random_lightCondition = True):
    """
    Load in background
    Only load the data of few models to save memory consumption

    If random_model: load model one by one (used for validation)
        else: randomly load few model at the same time (used for training)
    If random_lightCondition: load random light condition for each view of each model
        else: load specific light condition for validation
    """

    N_epoches = N_epoches if random_model else len(modelList)  
    for epoch in range(N_epoches):
        if random_model:    # Try to avoid redundant modelIndexes
            modelList_2load = random.sample(modelList, N_randomModels) if len(modelList) > N_randomModels else modelList
        else:   # load only one model in each loop to save GPU time
            modelList_2load = modelList[epoch: epoch+1]
        cube_param, vxl_ijk_list, density_list = prepareData.load_sparse_surfacePts_asnp( \
                modelIndexList = modelList_2load, \
                modelFile_pattern = os.path.join(params.__datasetFolder, params.__modelFile_pattern), \
                npzFile_pattern = os.path.join(params.__input_data_rootFld, params.__npzFile_pattern), \
                N_pts_onOffSurface = N_on_off_surfacePts, cube_D = cube_D_loaded, inputDataType = 'pcd', \
                cube_resolutionList = [0.8, 0.4, 0.2], density_dtype = 'float' if params.__soft_label else 'bool')
        dense_gt = sparseCubes.sparse2dense(vxl_ijk_list, density_list, coords_shape = cube_D_loaded, dt = np.float32)

        images_list = image.readImages_models_views_lights(datasetFolder = params.__datasetFolder, 
                modelList = modelList,
                viewList = params.__viewList, 
                lightConditions = lightConditions, 
                imgNamePattern_fn = params.imgNamePattern_fn,
                random_lightCondition = random_lightCondition)
        yield images_list, cube_param, dense_gt


def train(cameraPOs_np, cameraTs_np, lr_tensor = None, 
        train_fn = None, val_fn = None,
        N_on_off_surfacePts_train = [100, 100], N_on_off_surfacePts_val = [100, 100]):
    """

    inputs
    ----------
    lr_tensor: used for weight decay
    """


    #######################
    # SurfaceNet training #
    #######################

    N_viewPairs = params.__N_viewPairs4train

    start_time_epoch = time.time()
    for epoch, (images_list_train, cube_param_train, dense_gt_train) in \
            enumerate(BackgroundGenerator(iterate_models_and_lightConditions( \
                    N_epoches = params.__N_epoches, 
                    N_on_off_surfacePts = N_on_off_surfacePts_train,
                    N_randomModels = 7,     # randomly load data of few models to save memory consumption!
                    cube_D_loaded = params.__cube_D_loaded,
                    random_model = True,
                    random_lightCondition = True,
                    lightConditions = params.__random_lightConditions,
                    modelList = params.__modelList_train))):
        # TODO: load partial models; partial views/lightings in another thread / process?
        N_cubes_train = dense_gt_train.shape[0]
        if (epoch%params.__lr_decay_per_N_epoch == 0) and (epoch > 1):
            lr_tensor.set_value(lr_tensor.get_value() * params.__lr_decay)        
            print 'current updated lr_tensor = {}'.format(lr_tensor.get_value())

        acc_train_batches, acc_guess_all0 = [], []
        N_batches = 2 #N_cubes_train / params.__chunk_len_train
        for _batch, (_CVCs2_sub, _gt_sub) in enumerate(BackgroundGenerator(iterate_minibatches( \
                N_batches = N_batches, 
                batchSize = params.__chunk_len_train, N_viewPairs = N_viewPairs, cube_param = cube_param_train,
                cameraPOs_np = cameraPOs_np, images_list = images_list_train, dense_gt = dense_gt_train))):

            start_time_batch = time.time()

            # TODO: eliminate the 'if' condition
            # TODO: train_fn have different setting for different training procedures.
            _loss, acc, surfacePrediction = train_fn(_CVCs2_sub, _gt_sub)
                                    # if params.__N_viewPairs4train == 1 \
                                    # else nViewPair_SurfaceNet_fn(_CVCs2_sub, w_viewPairs4Reconstr[_batch[validCubes]])
            print("batch / epoch time {} / {}".format(time.time() - start_time_batch, time.time() - start_time_epoch))


            # if params.__train_SurfaceNet_with_SimilarityNet:
            #     selected_viewPairs, similNet_features = perform_similNet(similNet_fn=similNet_fn, \
            #             occupiedCubes_param = train_gt_param[selected], N_select_viewPairs = params.__N_viewPairs2train, models_img=models_img_train, \
            #             view_set = params.__view_set, cameraPOs=cameraPOs, cameraTs=cameraTs, patch_r=32, batch_size=100, similNet_features_dim = params.__similNet_features_dim)
            #     
            #     train_X_sub = gen_coloredCubes(selected_viewPairs = selected_viewPairs, occupiedCubes_param = train_gt_param[selected], \
            #             cameraPOs=cameraPOs, models_img=models_img_train, visualization_ON = False, occupiedCubes_01 = train_gt_sub)
            #     train_gt_sub, train_X_sub, train_X_rgb_sub = preprocess_augmentation(train_gt_sub, train_X_sub, augment_ON=True, color2grey = input_is_grey)
            #     _loss, acc, predict_train, similFeature_softmax_output = train_fn(train_X_sub, similNet_features.reshape(-1,similNet_features.shape[-1]), train_gt_sub)

            # else:
            #     selected_viewPairs = select_M_viewPairs_from_N_randViews_for_N_samples(params.__view_set, params.__N_randViews4train, \
            #             params.__N_viewPairs2train, N_samples = params.__chunk_len_train, \
            #             cubes_visib = train_gt_visib[selected] if params.__only_nonOcclud_cubes else None)
            #     # selected_viewPairs = np.random.choice(params.__view_set, (params.__chunk_len_train,params.__N_viewPairs2train,2))
            #     train_X_sub = gen_coloredCubes(selected_viewPairs = selected_viewPairs, occupiedCubes_param = train_gt_param[selected], \
            #             cameraPOs=cameraPOs, models_img=models_img_train, visualization_ON = False, occupiedCubes_01 = train_gt_sub)
            #     train_gt_sub, train_X_sub, train_X_rgb_sub = preprocess_augmentation(train_gt_sub, train_X_sub, augment_ON=True, color2grey = input_is_grey)
            #     _loss, acc, predict_train = train_fn(train_X_sub, train_gt_sub)
            #     

            acc_train_batches.append(float(acc))
            acc_guess_all0.append(1-float(_gt_sub.sum())/_gt_sub.size)
            print("Epoch %d, _batch %d: Loss %g, acc %g, acc_guess_all0 %g" % \
                                          (epoch, _batch, np.sum(_loss), np.asarray(acc_train_batches).mean(), np.asarray(acc_guess_all0).mean()))


        if params.__val_ON and ((epoch % 1) == 0):    # every N epoch
            print "starting validation..."    
            for epoch, (images_list_val, cube_param_val, dense_gt_val) in \
                    enumerate(BackgroundGenerator(iterate_models_and_lightConditions( \
                            N_epoches = params.__N_epoches, 
                            N_on_off_surfacePts = N_on_off_surfacePts_val,
                            N_randomModels = len(params.__modelList_val),     # select all models
                            cube_D_loaded = params.__cube_D_loaded,
                            random_model = False,
                            random_lightCondition = False,
                            lightConditions = params.__lightConditions,
                            modelList = params.__modelList_val))):
                N_cubes_val = dense_gt_val.shape[0]
                acc_val_batches = []
                N_batches = N_cubes_val/params.__chunk_len_val
                for _batch, (_CVCs2_sub, _gt_sub) in enumerate(BackgroundGenerator(iterate_minibatches( \
                    N_batches = N_batches, 
                    batchSize = params.__chunk_len_val, N_viewPairs = N_viewPairs, cube_param = cube_param_val,
                    cameraPOs_np = cameraPOs_np, images_list = images_list_val, dense_gt = dense_gt_val))):
                    acc_val, predict_val = val_fn(_CVCs2_sub, _gt_sub)
                    acc_val_batches.append(acc_val)   

                    # if params.__train_SurfaceNet_with_SimilarityNet:
                    #     selected_viewPairs, similNet_features = perform_similNet(similNet_fn=similNet_fn, \
                    #             occupiedCubes_param = val_gt_param[selected], N_select_viewPairs = params.__N_select_viewPairs2val, models_img=images_list, \
                    #             view_set = params.__view_set, cameraPOs=cameraPOs, cameraTs=cameraTs, patch_r=32, batch_size=100, similNet_features_dim = params.__similNet_features_dim)

                    #     val_X_sub = gen_coloredCubes(selected_viewPairs = selected_viewPairs, occupiedCubes_param = val_gt_param[selected], \
                    #             cameraPOs=cameraPOs, models_img=images_list, visualization_ON = False, occupiedCubes_01 = val_gt_sub)                    
                    #     val_gt_sub, val_X_sub, val_X_rgb_sub = preprocess_augmentation(val_gt_sub, val_X_sub, augment_ON=False, color2grey = input_is_grey)
                    #     acc_val, predict_val = val_fn(val_X_sub, similNet_features.reshape(-1,similNet_features.shape[-1]), val_gt_sub)
                    #     # acc_val, predict_val = fuseNet_val_fn(val_X_sub, val_gt_sub)
                    # else:
                    #     selected_viewPairs = select_M_viewPairs_from_N_randViews_for_N_samples(params.__view_set, params.__N_randViews4val, \
                    #             params.__N_select_viewPairs2val, cubes_visib = val_gt_visib[selected] if params.__only_nonOcclud_cubes else None)
                    #     # selected_viewPairs = np.random.choice(params.__view_set, (params.__chunk_len_val,params.__N_select_viewPairs2val,2))
                    #     val_X_sub = gen_coloredCubes(selected_viewPairs = selected_viewPairs, occupiedCubes_param = val_gt_param[selected], \
                    #             cameraPOs=cameraPOs, models_img=images_list, visualization_ON = False, occupiedCubes_01 = val_gt_sub)                    
                    #     val_gt_sub, val_X_sub, val_X_rgb_sub = preprocess_augmentation(val_gt_sub, val_X_sub, augment_ON=False, color2grey = input_is_grey)
                    #     acc_val, predict_val = val_fn(val_X_sub, val_gt_sub)
                        
                        
                    # if params.__val_visualize_ON :
                    #     X_1, X_2 = [val_X_rgb_sub[0:params.__N_select_viewPairs2val], val_gt_sub[0:1]]
                    #     result = predict_val[0]
                    #     X_1 += 0 #params.__CHANNEL_MEAN # [None,:,None,None,None]   #(X_1+.5)*255. 
                    #     tmp_5D = np.copy(X_1) # used for visualize the surface part of the colored cubes
                    #     # if want to visualize the result, just stop at the following code, and in the debug probe run the visualize_N_densities_pcl func.
                    #     # rimember to 'enter' before continue the program~
                    #     tmp_5D[:,:,X_2[0].squeeze()==0]=0 
                    #     if not params.__train_ON:
                    #         visualize_N_densities_pcl([X_2[0]*params.__surfPredict_scale4visual, result*params.__surfPredict_scale4visual, tmp_5D[0,3:], tmp_5D[0,:3], X_1[0,3:], X_1[0,:3]])
                acc_val = np.asarray(acc_val_batches).mean()
                print("val_acc %g" %(acc_val))

            # if (epoch % params.__every_N_epoch_2saveModel) == 0:
            #     save_entire_model(net[params.__layer_2_save_model], '2D_2_3D-{}-{:0.3}_{:0.3}.model'.format(epoch, np.asarray(acc_train_batches).mean(), acc_val))             



