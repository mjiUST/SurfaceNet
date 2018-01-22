import os
import sys
import random
import numpy as np
import time

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
    load camera params and images, that don't change during training
    """

    cameraPOs_np = camera.readCameraPOs_as_np(datasetFolder = params.__datasetFolder, datasetName = params.__datasetName, poseNamePattern = params.__poseNamePattern, viewList = params.__viewList)  # (N_views, 3, 4) np
    cameraTs_np = camera.cameraPs2Ts(cameraPOs = cameraPOs_np)  # (N_views, 3) np

    # load images for training, as list of numpy array (uint8)
    images_list_train = image.readImages_models_views_lights(datasetFolder = params.__datasetFolder, 
            modelList = params.__modelList_train,  # test: can put comment
            viewList = params.__viewList, 
            lightConditions = params.__lightConditions, 
            imgNamePattern_fn = params.imgNamePattern_fn)
    images_list_val = image.readImages_models_views_lights(datasetFolder = params.__datasetFolder, 
            modelList = params.__modelList_val,
            viewList = params.__viewList,
            lightConditions = params.__lightConditions,
            imgNamePattern_fn = params.imgNamePattern_fn)
    return {'cameraPOs_np': cameraPOs_np, 'cameraTs_np': cameraTs_np, 'images_list_train': images_list_train, 'images_list_val': images_list_val}


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


def train(cameraPOs_np, cameraTs_np, images_list_train, images_list_val, 
        cube_param_train, vxl_ijk_list_train, density_list_train, 
        cube_param_val, vxl_ijk_list_val, density_list_val, cube_D_loaded, lr_tensor = None, 
        train_fn = None, val_fn = None):
    """

    inputs
    ----------
    lr_tensor: used for weight decay
    """


    #######################
    # SurfaceNet training #
    #######################

    N_viewPairs = params.__N_viewPairs4train
    N_cubes_train = cube_param_train.shape[0]
    dense_gt_train = sparseCubes.sparse2dense(vxl_ijk_list_train, density_list_train, coords_shape = cube_D_loaded, dt = np.float32)
    N_cubes_val = cube_param_val.shape[0]
    dense_gt_val = sparseCubes.sparse2dense(vxl_ijk_list_val, density_list_val, coords_shape = cube_D_loaded, dt = np.float32)

    for epoch in range(1, params.__N_epochs):
        # TODO: load partial models; partial views/lightings in another thread / process?
        if epoch%params.__lr_decay_per_N_epoch == 0:
            lr_tensor.set_value(lr_tensor.get_value() * params.__lr_decay)        
            print 'current updated lr_tensor = {}'.format(lr_tensor.get_value())

        acc_train_batches, acc_guess_all0 = [], []
        for _batch in range(N_cubes_train / params.__chunk_len_train):
            start_time_batch = time.time()
            selector = random.sample(range(N_cubes_train), params.__chunk_len_train) # randomly select = shuffle the samples
            # generate CVC
            rand_viewPairs = np.random.randint(0, len(params.__viewList), (params.__chunk_len_train, N_viewPairs, 2)) # (params.__chunk_len_train, N_viewPair, 2) randomly select viewPairs for each cube
            # dtype = uint8
            _CVCs1_sub = CVC.gen_models_coloredCubes( \
                    viewPairs = rand_viewPairs,  \
                    cube_params = cube_param_train[selector], \
                    cameraPOs = cameraPOs_np, \
                    models_img_list = images_list_train, \
                    cube_D = cube_param_train['cube_D'][0] \
                    ) # ((N_cubeSub * __N_viewPairs4train, 3 * 2) + (D_CVC,) * 3) 5D
            _gt_sub = dense_gt_train[selector][:, None]  # (N, D,D,D) --> (N, 1, D,D,D)
            _gt_sub, _CVCs2_sub = CVC.preprocess_augmentation(_gt_sub, _CVCs1_sub, mean_rgb = params.__MEAN_CVC_RGBRGB[None,:,None,None,None], augment_ON=True, crop_ON = True, cube_D = params.__cube_D)
            # TODO: eliminate the 'if' condition
            # TODO: train_fn have different setting for different training procedures.
            _loss, acc, surfacePrediction = train_fn(_CVCs2_sub, _gt_sub)
                                    # if params.__N_viewPairs4train == 1 \
                                    # else nViewPair_SurfaceNet_fn(_CVCs2_sub, w_viewPairs4Reconstr[_batch[validCubes]])
            print("perform_similNet takes {}".format(time.time() - start_time_batch))


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



        
        # if params.__val_ON:
        #     if (epoch % 1) == 0:    # every N epoch
        #         print "starting validation..."    
        #         acc_val_batches = []

        #         for batch_val in range(0, N_cubes_val/params.__chunk_len_val):
        #             selected = range(batch_val*params.__chunk_len_val,(batch_val+1)*params.__chunk_len_val)
        #             val_gt_sub = val_gt[selected][:,None,...] ## convert to 5D

        #             if params.__train_SurfaceNet_with_SimilarityNet:
        #                 selected_viewPairs, similNet_features = perform_similNet(similNet_fn=similNet_fn, \
        #                         occupiedCubes_param = val_gt_param[selected], N_select_viewPairs = params.__N_select_viewPairs2val, models_img=images_list, \
        #                         view_set = params.__view_set, cameraPOs=cameraPOs, cameraTs=cameraTs, patch_r=32, batch_size=100, similNet_features_dim = params.__similNet_features_dim)
        #                 
        #                 val_X_sub = gen_coloredCubes(selected_viewPairs = selected_viewPairs, occupiedCubes_param = val_gt_param[selected], \
        #                         cameraPOs=cameraPOs, models_img=images_list, visualization_ON = False, occupiedCubes_01 = val_gt_sub)                    
        #                 val_gt_sub, val_X_sub, val_X_rgb_sub = preprocess_augmentation(val_gt_sub, val_X_sub, augment_ON=False, color2grey = input_is_grey)
        #                 acc_val, predict_val = val_fn(val_X_sub, similNet_features.reshape(-1,similNet_features.shape[-1]), val_gt_sub)
        #                 # acc_val, predict_val = fuseNet_val_fn(val_X_sub, val_gt_sub)
        #             else:
        #                 selected_viewPairs = select_M_viewPairs_from_N_randViews_for_N_samples(params.__view_set, params.__N_randViews4val, \
        #                         params.__N_select_viewPairs2val, cubes_visib = val_gt_visib[selected] if params.__only_nonOcclud_cubes else None)
        #                 # selected_viewPairs = np.random.choice(params.__view_set, (params.__chunk_len_val,params.__N_select_viewPairs2val,2))
        #                 val_X_sub = gen_coloredCubes(selected_viewPairs = selected_viewPairs, occupiedCubes_param = val_gt_param[selected], \
        #                         cameraPOs=cameraPOs, models_img=images_list, visualization_ON = False, occupiedCubes_01 = val_gt_sub)                    
        #                 val_gt_sub, val_X_sub, val_X_rgb_sub = preprocess_augmentation(val_gt_sub, val_X_sub, augment_ON=False, color2grey = input_is_grey)
        #                 acc_val, predict_val = val_fn(val_X_sub, val_gt_sub)
        #                 
        #                 
        #             acc_val_batches.append(acc_val)   
        #             if params.__val_visualize_ON :
        #                 X_1, X_2 = [val_X_rgb_sub[0:params.__N_select_viewPairs2val], val_gt_sub[0:1]]
        #                 result = predict_val[0]
        #                 X_1 += 0 #params.__CHANNEL_MEAN # [None,:,None,None,None]   #(X_1+.5)*255. 
        #                 tmp_5D = np.copy(X_1) # used for visualize the surface part of the colored cubes
        #                 # if want to visualize the result, just stop at the following code, and in the debug probe run the visualize_N_densities_pcl func.
        #                 # rimember to 'enter' before continue the program~
        #                 tmp_5D[:,:,X_2[0].squeeze()==0]=0 
        #                 if not params.__train_ON:
        #                     visualize_N_densities_pcl([X_2[0]*params.__surfPredict_scale4visual, result*params.__surfPredict_scale4visual, tmp_5D[0,3:], tmp_5D[0,:3], X_1[0,3:], X_1[0,:3]])
        #         acc_val = np.asarray(acc_val_batches).mean()
        #         print("val_acc %g" %(acc_val))

        #     if (epoch % params.__every_N_epoch_2saveModel) == 0:
        #         save_entire_model(net[params.__layer_2_save_model], '2D_2_3D-{}-{:0.3}_{:0.3}.model'.format(epoch, np.asarray(acc_train_batches).mean(), acc_val))             



