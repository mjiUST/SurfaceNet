import os
import sys
import copy
import math
import random
random.seed(201801)
import numpy as np
np.random.seed(201801)
import time
from prefetch_generator import BackgroundGenerator, background  # https://github.com/justheuristic/prefetch_generator


import params
sys.path.append("./utils")
import CVC
import utils
import image
import camera
import prepareData
import sparseCubes
sys.path.append("./nets")
import utils_nets
import SurfaceNet
import SimilarityNet


def load_dnn_fns(with_relativeImpt, SurfaceNet_model_path = None, SimilarityNet_model_path = 'N/A'):
    """
    define / load all the dnn functions for training / finetuning
    """

    # define or load SurfaceNet
    net, train_fn, val_fn, lr_tensor = SurfaceNet.SurfaceNet_trainVal(with_relativeImpt, pretrained_model_path = SurfaceNet_model_path)
    outputDic = {'net': net, 'train_fn': train_fn, 'val_fn': val_fn, 'lr_tensor': lr_tensor}

    # TODO: define and load SimilarityNet
    if not SimilarityNet_model_path == 'N/A':
        patch2embedding_fn, embeddingPair2simil_fn = SimilarityNet.SimilarityNet_inference(
                model_path = SimilarityNet_model_path,
                imgPatch_hw_size = (params.__imgPatch_hw_size, )*2 )
        outputDic.update({'patch2embedding_fn': patch2embedding_fn, 'embeddingPair2simil_fn': embeddingPair2simil_fn})

    return outputDic


def loadFixedVar_4training():
    """
    Load camera params ..., that don't change during training
    Images are loaded while being used to save memory!
    """

    cameraPOs_np = camera.readCameraPOs_as_np(datasetFolder = params.__datasetFolder, datasetName = params.__datasetName, poseNamePattern = params.__poseNamePattern, viewList = params.__viewList)  # (N_views, 3, 4) np
    cameraTs_np = camera.cameraPs2Ts(cameraPOs = cameraPOs_np)  # (N_views, 3) np

    return {'cameraPOs_np': cameraPOs_np, 'cameraTs_np': cameraTs_np}


def prepare_minibatches(batchSize, N_viewPairs, cube_param, cameraPOs_np, images_list,
        dense_gt, N_batches = None, augment_ON = True, rand_viewPairs_ON = True):
    """
    fetch minibatches
    If N_batches is None: loop through each cube param onece. (for validation / testing)
        else: randomly select cubes in each batch. (for training)
    If rand_viewPairs_ON is True: randomly select view pairs; otherwise, evenly sample from the view pairs set.
    """

    N_cubes = cube_param.shape[0]
    N_views = cameraPOs_np.shape[0]
    if N_batches is not None: # train: randomly select
        selectors_np = np.random.randint(0, N_cubes, (N_batches, batchSize))
    else: # validation / test: select in order
        selectors_np = utils.gen_batch_npBool(N_all = N_cubes, batch_size = batchSize)
    viewPairs_all = utils.k_combination_np(range(N_views), k = 2)     # (N_combinations, 2)
    N_combinations = viewPairs_all.shape[0]

    for selector in selectors_np:
        N_selector = batchSize if N_batches else selector.sum()
        # generate CVC
        viewPairs_index = np.random.randint(0, N_combinations, (N_selector, N_viewPairs)) if rand_viewPairs_ON \
                else np.arange(N_combinations)[::N_combinations/N_viewPairs][:N_viewPairs][None].repeat(N_selector, axis=0)
        viewPairs = viewPairs_all[viewPairs_index] # (N_selector, N_viewPair, 2) randomly select viewPairs for each cube
        # viewPairs = np.random.randint(0, len(params.__viewList), (N_selector, N_viewPairs, 2)) # (N_selector, N_viewPair, 2) randomly select viewPairs for each cube
        # dtype = uint8
        _CVCs1_sub, images_slice = CVC.gen_models_coloredCubes( \
                viewPairs = viewPairs,
                cube_params = cube_param[selector],
                cameraPOs = cameraPOs_np,
                models_img_list = images_list,
                cube_D = cube_param['cube_D'][0],
                random_colorCondition = augment_ON # If True: randomly select light conditions for different views
                ) # ((N_selector * __N_viewPairs4train, 3 * 2) + (D_CVC,) * 3) 5D
        _gt_sub = dense_gt[selector][:, None]  # (N, D,D,D) --> (N, 1, D,D,D)
        _gt_sub, _CVCs2_sub = CVC.preprocess_augmentation(_gt_sub, _CVCs1_sub, mean_rgb = params.__MEAN_CVC_RGBRGB[None,:,None,None,None], augment_ON=augment_ON, crop_ON = True, cube_D = params.__cube_D)

        _nRGB_CVCs_sub = _CVCs1_sub.reshape((N_selector, N_viewPairs, 2, 3) + _CVCs1_sub.shape[-3:])
        _nRGB_CVCs_sub = _nRGB_CVCs_sub.mean(axis = 2) # (N_cubes, N_viewPairs, 3, D,D,D)

        yield selector, viewPairs, viewPairs_index, _nRGB_CVCs_sub, _CVCs2_sub, _gt_sub, images_slice



def traverse_models_and_select_lightConditions(N_models_inBatch, modelList, lightConditions,
        N_on_off_surfacePts, cube_D_loaded, random_modelOrder = True, random_lightCondition = True):
    """
    Load in background
    Only load the data of few models to save memory consumption

    If random_modelOrder: shuffle the modelIndex order (can be used for training)
        else: keep the modelIndex order as origin (used for validation)
    If random_lightCondition: load random light condition for each view of each model
        else: load specific light condition for validation

    inputs:
    --------

    outputs:
    --------
    modelList_2load, record_lastLightCondition4models: try not to print log in other threads, otherwise it is hard for file comparison.
    """

    modelList_copy = copy.deepcopy(modelList)
    if random_modelOrder:
        random.shuffle(modelList_copy)
    modelList_np = np.asarray(modelList_copy).astype(np.int)
    # Each time return N_models_inBatch models
    selectors_list = utils.gen_batch_npBool(N_all = len(modelList), batch_size = N_models_inBatch)
    N_iters = len(selectors_list)
    for _iter, _selector in enumerate(selectors_list):
        modelList_2load = list(modelList_np[_selector])  # Try to avoid redundant modelIndexes
        cube_param, vxl_ijk_list, density_list = prepareData.load_sparse_surfacePts_asnp( \
                modelIndexList = modelList_2load,
                modelFile_pattern = os.path.join(params.__datasetFolder, params.__modelFile_pattern),
                npzFile_pattern = os.path.join(params.__input_data_rootFld, params.__npzFile_pattern),
                N_pts_onOffSurface = N_on_off_surfacePts, cube_D = cube_D_loaded, inputDataType = 'pcd',
                silentLog = params.__silentLog,
                cube_resolutionList = [0.8, 0.4, 0.2], density_dtype = 'float' if params.__soft_label else 'bool')
        dense_gt = sparseCubes.sparse2dense(vxl_ijk_list, density_list, coords_shape = cube_D_loaded, dt = np.float32)

        images_list, record_lastLightCondition4models = image.readImages_models_views_lights(datasetFolder = params.__datasetFolder, 
                modelList = modelList_2load,
                viewList = params.__viewList, 
                lightConditions = lightConditions, 
                imgNamePattern_fn = params.imgNamePattern_fn,
                silentLog = params.__silentLog,
                random_lightCondition = random_lightCondition)
        yield _iter, N_iters, images_list, cube_param, dense_gt, modelList_2load, record_lastLightCondition4models


def train(cameraPOs_np, cameraTs_np, lr_tensor = None, trainingStage = 0,
        net = None, train_fn = None, val_fn = None, layer_2_save_model = '', N_epoch = 2,
        patch2embedding_fn = None, embeddingPair2simil_fn = None,
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
    N_views = cameraPOs_np.shape[0]

    start_time_epoch = time.time()
    for epoch in range(N_epoch):  # how many times to iterate the entire modelList

        loss_batches, acc_train_batches, acc_guess_all0 = [], [], []
        print("\nEpoch {}".format(epoch))
        if (epoch%params.__lr_decay_per_N_epoch == 0) and (epoch > 1):
            lr_tensor.set_value(lr_tensor.get_value() * params.__lr_decay)        
            print 'current updated lr_tensor = {}'.format(lr_tensor.get_value())

        for _i_modelSet, (_iter, N_iters, images_list_train, cube_param_train, dense_gt_train, modelList_2load, record_lastLightCondition4models) in \
                enumerate(BackgroundGenerator(traverse_models_and_select_lightConditions( \
                        N_on_off_surfacePts = N_on_off_surfacePts_train,
                        N_models_inBatch = 1,     # randomly load data of few models to save memory consumption!
                        cube_D_loaded = params.__cube_D_loaded,
                        random_modelOrder = True,
                        random_lightCondition = True,
                        lightConditions = params.__random_lightConditions,
                        modelList = params.__modelList_train), max_prefetch=1)):
                # images_list_train: [(N_views, 1, h, w, ...), ] * N_models=1

            if params.__train_ON:
                N_cubes_train = dense_gt_train.shape[0]
                N_batches_train = N_cubes_train / params.__chunk_len_train
                print("Training iter {}: N_on_off_surfacePts_train: {}; ModelList_2load: {}; Record_lastLightCondition4models: {}".format( \
                        _iter, N_on_off_surfacePts_train, modelList_2load, record_lastLightCondition4models))

                if patch2embedding_fn is not None: # calculate the SimilarityNet terms before loop to speed up
                    theta_viewPairs = camera.viewPairAngles_wrt_pts( \
                            cameraTs = cameraTs_np, 
                            pts_xyz = cube_param_train['min_xyz'] + (params.__cube_D_loaded * cube_param_train['resolution'])[:, None] / 2.    # cube_center_mm (N_cubes, 3)
                            )  # (N_cubes, N_viewPairs)
                    patches_embedding = utils.generate_1model_patches_embedding( \
                            images_list = [images_list_train[0][_v, 0] for _v in range(N_views)],  # [(N_views, N_lights=1, h, w, 3/1), ] * N_models=1 --> [(h, w, 3/1), ] * N_views
                            cube_param = cube_param_train,
                            cameraPOs_np = cameraPOs_np,
                            patches_mean_bgr = params.__MEAN_PATCHES_BGR,
                            D_embedding = params.__D_imgPatchEmbedding,
                            cube_D = params.__cube_D_loaded,
                            patchSize = params.__imgPatch_hw_size,
                            batchSize_patch2embedding = params.__batchSize_similNet_patch2embedding,
                            patch2embedding_fn = patch2embedding_fn) # (N_cubes, N_views, D_embedding)

                for _batch, (selector, viewPairs, viewPairs_index, _nRGB_CVCs_sub, _CVCs2_sub, _gt_sub, images_slice) in enumerate(BackgroundGenerator(prepare_minibatches( \
                        N_batches = N_batches_train, # random selection in each iteration
                        batchSize = params.__chunk_len_train, N_viewPairs = N_viewPairs, cube_param = cube_param_train,
                        rand_viewPairs_ON = True,
                        cameraPOs_np = cameraPOs_np, images_list = images_list_train, dense_gt = dense_gt_train), max_prefetch=1)):

                    start_time_batch = time.time()

                    if patch2embedding_fn is not None:
                        viewPairs_featureVec = utils.generate_viewPairs_featureVec( \
                                theta_viewPairs_all = theta_viewPairs[selector],
                                patches_embedding = patches_embedding[selector],
                                viewPairs = viewPairs, # (N_cubes, N_viewPairs, 2)
                                viewPairs_index = viewPairs_index, # (N_cubes, N_viewPairs)
                                D_featureVec = params.__D_viewPairFeature,
                                batchSize_embeddingPair2simil = params.__batchSize_similNet_embeddingPair2simil,
                                embeddingPair2simil_fn = embeddingPair2simil_fn
                                )
                        train_loss, acc, train_predict, output_softmaxWeights = train_fn(_CVCs2_sub, viewPairs_featureVec, _gt_sub)
                                            # if params.__N_viewPairs4train == 1 \
                                            # else nViewPair_SurfaceNet_fn(_CVCs2_sub, w_viewPairs4Reconstr[_batch[validCubes]])
                    else:  # without relative importance
                        train_loss, acc, train_predict = train_fn(_CVCs2_sub, _gt_sub)
                    ## print("batch / epoch time {} / {}".format(time.time() - start_time_batch, time.time() - start_time_epoch))
                    if math.isnan(train_loss.mean()):
                        print('Loss is nan!')


                    loss_batches.append(np.sum(train_loss))
                    acc_train_batches.append(float(acc))
                    acc_guess_all0.append(1-float(_gt_sub.sum())/_gt_sub.size)
                    if (_batch % (N_batches_train / 3 + 1)) == 0:     # only print results of few batches
                        train_acc = np.asarray(acc_train_batches).mean()
                        print("Batch %d: Loss %g, train_acc %g, acc_guess_all0 %g" % \
                                (_batch, np.asarray(loss_batches).mean(), train_acc, np.asarray(acc_guess_all0).mean()))
                        # move these print log from other threads to the main thread

            val_acc = 0
            if params.__val_ON and (_iter == N_iters-1): # at the end of epoches. ((_iter % 2) == 0): every N modelSet
                val_acc_batches = []
                # loop through all the models in the modelList_val. Load & process one-by-one to save memory.
                for _, (_iter_val, N_iters_val, images_list_val, cube_param_val, dense_gt_val, modelList_2load, record_lastLightCondition4models) in \
                        enumerate(traverse_models_and_select_lightConditions( \
                                N_on_off_surfacePts = N_on_off_surfacePts_val,
                                N_models_inBatch = 1,     # select all models
                                cube_D_loaded = params.__cube_D_loaded,
                                random_modelOrder = False,
                                random_lightCondition = False,
                                lightConditions = params.__lightConditions,
                                modelList = params.__modelList_val)):

                    if params.__visualizeValModel:
                        prediction_list, rgb_list, vxl_ijk_list, rayPooling_votes_list = [], [], [], []
                        param_np, viewPair_np = None, None

                    if patch2embedding_fn is not None: # calculate the SimilarityNet terms before loop to speed up
                        theta_viewPairs = camera.viewPairAngles_wrt_pts( \
                                cameraTs = cameraTs_np, 
                                pts_xyz = cube_param_val['min_xyz'] + (params.__cube_D_loaded * cube_param_val['resolution'])[:, None] / 2.    # cube_center_mm (N_cubes, 3)
                                )  # (N_cubes, N_viewPairs)
                        patches_embedding = utils.generate_1model_patches_embedding( \
                                images_list = [images_list_val[0][_v, 0] for _v in range(N_views)],  # [(N_views, N_lights=1, h, w, 3/1), ] * N_models=1 --> [(h, w, 3/1), ] * N_views
                                cube_param = cube_param_val,
                                cameraPOs_np = cameraPOs_np,
                                patches_mean_bgr = params.__MEAN_PATCHES_BGR,
                                D_embedding = params.__D_imgPatchEmbedding,
                                cube_D = params.__cube_D_loaded,
                                patchSize = params.__imgPatch_hw_size,
                                batchSize_patch2embedding = params.__batchSize_similNet_patch2embedding,
                                patch2embedding_fn = patch2embedding_fn) # (N_cubes, N_views, D_embedding)

                    N_cubes_val = dense_gt_val.shape[0]
                    for _batch, (selector, viewPairs, viewPairs_index, _nRGB_CVCs_sub, _CVCs2_sub, _gt_sub, images_slice) in enumerate(BackgroundGenerator(prepare_minibatches( \
                            N_batches = None,  # test all the samples rather than random selection
                            augment_ON = False,
                            rand_viewPairs_ON = False, # evenly sample from viewPairs set, rather than randomly select
                            batchSize = params.__chunk_len_val, N_viewPairs = N_viewPairs, cube_param = cube_param_val,
                            cameraPOs_np = cameraPOs_np, images_list = images_list_val, dense_gt = dense_gt_val), max_prefetch=1)):

                        if patch2embedding_fn is not None:
                            viewPairs_featureVec = utils.generate_viewPairs_featureVec( \
                                    theta_viewPairs_all = theta_viewPairs[selector],
                                    patches_embedding = patches_embedding[selector],
                                    viewPairs = viewPairs, # (N_cubes, N_viewPairs, 2)
                                    viewPairs_index = viewPairs_index, # (N_cubes, N_viewPairs)
                                    D_featureVec = params.__D_viewPairFeature,
                                    batchSize_embeddingPair2simil = params.__batchSize_similNet_embeddingPair2simil,
                                    embeddingPair2simil_fn = embeddingPair2simil_fn
                                    )
                            val_acc, val_predict = val_fn(_CVCs2_sub, viewPairs_featureVec, _gt_sub)
                                                # if params.__N_viewPairs4train == 1 \
                                                # else nViewPair_SurfaceNet_fn(_CVCs2_sub, w_viewPairs4Reconstr[_batch[validCubes]])
                        else:  # without relative importance
                            val_acc, val_predict = val_fn(_CVCs2_sub, _gt_sub)

                        val_acc_batches.append(val_acc)   


                        if params.__visualizeValModel:
                            val_rgb_sub = np.mean(_nRGB_CVCs_sub, axis=1)  # ((N_cubes, N_CVCs, 3) + (D_CVC,) * 3) 6D --> ((N_cubes, 3) + (D_CVC,) * 3) 5D
                            updated_sparse_list_np = sparseCubes.append_dense_2sparseList( \
                                    prediction_sub = val_predict, rgb_sub = val_rgb_sub, param_sub = cube_param_val[selector],
                                    min_prob = 0.5, cube_ijk_np = 'N/A', # don't process cube_ijk information
                                    enable_centerCrop = False, enable_rayPooling = False,
                                    cube_was_cropped = True, # will shift the cubes' 'min_xyz' value
                                    prediction_list = prediction_list, rgb_list = rgb_list, vxl_ijk_list = vxl_ijk_list,
                                    param_np = param_np)
                            prediction_list, rgb_list, vxl_ijk_list, _,  _, param_np, _ = updated_sparse_list_np

                    if params.__visualizeValModel:
                        ply_filename = os.path.join(params.__output_PCDFile_rootFld, 'stage{}-epoch{}_{}-visualization_{}.ply'.format(trainingStage, epoch, _iter_val, modelList_2load))
                        vxl_mask_list = sparseCubes.filter_voxels(vxl_mask_list=[],prediction_list=prediction_list, prob_thresh= 0.5)
                        if len(vxl_mask_list) != 0:
                            sparseCubes.save_sparseCubes_2ply(vxl_mask_list, vxl_ijk_list, rgb_list, \
                                    param_np, ply_filePath=ply_filename, normal_list=None)

                val_acc = np.asarray(val_acc_batches).mean()
                # move these print log from other threads to the main thread
                print("Validation iter {}: N_on_off_surfacePts_train: {}; ModelList_2load: {}; Record_lastLightCondition4models: {}".format( \
                        _iter_val, N_on_off_surfacePts_train, modelList_2load, record_lastLightCondition4models))      
                print("val_acc %g" %(val_acc))


            if _iter == N_iters-1:
                modelFileName = 'stage{}-epoch{}_{}-{:0.3}_{:0.3}.model'.format(trainingStage, \
                        epoch, _iter, train_acc, val_acc)
                modelFilePath = utils_nets.save_entire_model(net[layer_2_save_model], 
                        save_folder = params.__output_modelFile_rootFld,
                        filename = modelFileName) 
    print('************\n')
    return modelFilePath


