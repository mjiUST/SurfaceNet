import random
import numpy as np

import params
sys.path.append("./utils")
import image
import camera
import prepareData
sys.path.append("./nets")
import SurfaceNet
import similarityNet


def train():

    ##############
    # prepare data
    ##############

    cameraPOs_np = camera.readCameraPOs_as_np(datasetFolder = datasetFolder, datasetName = params.__datasetName, poseNamePattern = poseNamePattern, model = model, viewList = viewList)  # (N_views, 3, 4) np
    cameraTs_np = camera.cameraPs2Ts(cameraPOs = cameraPOs_np)  # (N_views, 3) np

    # load images and surface points for training
    images_list_train = image.readImages_models_views_lights(datasetFolder = datasetFolder, modelList = params.__modelList_train, viewList = viewList, lightConditions = params.__lightConditions)     
    images_list_val = image.readImages_models_views_lights(datasetFolder = datasetFolder, modelList = params.__modelList_val, viewList = viewList, lightConditions = params.__lightConditions)     

    # generate / load sparse surface points, (save if not exit)
    # cube_param: min_xyz / resolution / cube_D / modelIndex
    cube_param_train, vxl_ijk_list_train, density_list_train = prepareData.load_sparse_surfacePts_asnp( \
            modelIndexList = params.__modelList_train, \
            modelFile_pattern = os.path.join(params.__datasetFolder, params.__modelFile_pattern), \
            npzFile_pattern = os.path.join(params.__input_data_rootFld, params.__npzFile_pattern), \
            N_pts_onOffSurface = [1000, 1000], cube_D = 50, inputDataType = 'pcd', \
            cube_resolutionList = [0.8, 0.4, 0.2], density_dtype = 'float' if params_volume.__soft_label else 'bool')
    N_cubes_train = cube_param_train.shape[0]

    cube_param_val, vxl_ijk_list_val, density_list_val = prepareData.load_sparse_surfacePts_asnp( \
            modelIndexList = params.__modelList_val, \
            modelFile_pattern = os.path.join(params.__datasetFolder, params.__modelFile_pattern), \
            npzFile_pattern = npzFile_pattern, \
            N_pts_onOffSurface = [1000, 0], cube_D = 50, inputDataType = 'pcd', \
            cube_resolutionList = [0.8, 0.4, 0.2], density_dtype = 'float' if params_volume.__soft_label else 'bool')
    N_cubes_val = cube_param_val.shape[0]


    ################
    # define network
    ################

    train_fn, val_fn = SurfaceNet.SurfaceNet_trainVal()

    # TODO: define and load similarityNet


    #####################
    # SurfaceNet training
    #####################

    for _batch in range(N_train / params_volume.__chunk_len_train):
        selector = random.sample(range(N_cubes_train), params_volume.__chunk_len_train) # shuffle the samples
        # generate CVC
        _CVCs1_sub = CVC.gen_coloredCubes( \
                selected_viewPairs = viewPairs4Reconstr[_batch[validCubes]],  \
                min_xyz = cubes_param_np['min_xyz'][selector],  \
                resol = cubes_param_np['resolution'][selector],  \
                colorize_cube_D = cubes_param_np['cube_D'][0],\
                cameraPOs=cameraPOs_np, \
                models_img=images_list, \
                visualization_ON = False)   # ((N_cubeSub * N_viewPairs4inference, 3 * 2) + (D_CVC,) * 3) 5D

        _, _CVCs2_sub = CVC.preprocess_augmentation(None, _CVCs1_sub, mean_rgb = params.__MEAN_CVC_RGBRGB[None,:,None,None,None], augment_ON=False, crop_ON = False)
        # TODO: eliminate the 'if' condition
        surfacePrediction, unfused_predictions = nViewPair_SurfaceNet_fn(_CVCs2_sub) if N_viewPairs4inference == 1 \
                                else nViewPair_SurfaceNet_fn(_CVCs2_sub, w_viewPairs4Reconstr[_batch[validCubes]])

        if epoch%params_volume.__lr_decay_per_N_epoch == 0:
            lr_tensor.set_value(lr_tensor.get_value() * params_volume.__lr_decay)        
            print 'current updated lr_tensor = {}'.format(lr_tensor.get_value())


        if params_volume.__train_fusionNet:
            selected_viewPairs, similNet_features = perform_similNet(similNet_fn=similNet_fn, \
                    occupiedCubes_param = train_gt_param[selected], N_select_viewPairs = params_volume.__N_viewPairs2train, models_img=models_img_train, \
                    view_set = params_volume.__view_set, cameraPOs=cameraPOs, cameraTs=cameraTs, patch_r=32, batch_size=100, similNet_features_dim = params_volume.__similNet_features_dim)
            
            train_X_sub = gen_coloredCubes(selected_viewPairs = selected_viewPairs, occupiedCubes_param = train_gt_param[selected], \
                    cameraPOs=cameraPOs, models_img=models_img_train, visualization_ON = False, occupiedCubes_01 = train_gt_sub)
            train_gt_sub, train_X_sub, train_X_rgb_sub = preprocess_augmentation(train_gt_sub, train_X_sub, augment_ON=True, color2grey = input_is_grey)
            _loss, acc, predict_train, similFeature_softmax_output = train_fn(train_X_sub, similNet_features.reshape(-1,similNet_features.shape[-1]), train_gt_sub)

        else:
            selected_viewPairs = select_M_viewPairs_from_N_randViews_for_N_samples(params_volume.__view_set, params_volume.__N_randViews4train, \
                    params_volume.__N_viewPairs2train, N_samples = params_volume.__chunk_len_train, \
                    cubes_visib = train_gt_visib[selected] if params_volume.__only_nonOcclud_cubes else None)
            # selected_viewPairs = np.random.choice(params_volume.__view_set, (params_volume.__chunk_len_train,params_volume.__N_viewPairs2train,2))
            train_X_sub = gen_coloredCubes(selected_viewPairs = selected_viewPairs, occupiedCubes_param = train_gt_param[selected], \
                    cameraPOs=cameraPOs, models_img=models_img_train, visualization_ON = False, occupiedCubes_01 = train_gt_sub)
            train_gt_sub, train_X_sub, train_X_rgb_sub = preprocess_augmentation(train_gt_sub, train_X_sub, augment_ON=True, color2grey = input_is_grey)
            _loss, acc, predict_train = train_fn(train_X_sub, train_gt_sub)
            

        acc_train_batches.append(list(acc))
        acc_guess_all0.append(1-float(train_gt_sub.sum())/train_gt_sub.size)
        print("Epoch %d, _batch %d: Loss %g, acc %g, acc_guess_all0 %g" % \
                                      (epoch, _batch, np.sum(_loss), np.asarray(acc_train_batches).mean(), np.asarray(acc_guess_all0).mean()))



        
        if params_volume.__val_ON:
            if (epoch % 1) == 0:    # every N epoch
                print "starting validation..."    
                acc_val_batches = []

                for batch_val in range(0, N_cubes/params_volume.__chunk_len_val):
                    selected = range(batch_val*params_volume.__chunk_len_val,(batch_val+1)*params_volume.__chunk_len_val)
                    val_gt_sub = val_gt[selected][:,None,...] ## convert to 5D

                    if params_volume.__train_fusionNet:
                        selected_viewPairs, similNet_features = perform_similNet(similNet_fn=similNet_fn, \
                                occupiedCubes_param = val_gt_param[selected], N_select_viewPairs = params_volume.__N_select_viewPairs2val, models_img=images_list, \
                                view_set = params_volume.__view_set, cameraPOs=cameraPOs, cameraTs=cameraTs, patch_r=32, batch_size=100, similNet_features_dim = params_volume.__similNet_features_dim)
                        
                        val_X_sub = gen_coloredCubes(selected_viewPairs = selected_viewPairs, occupiedCubes_param = val_gt_param[selected], \
                                cameraPOs=cameraPOs, models_img=images_list, visualization_ON = False, occupiedCubes_01 = val_gt_sub)                    
                        val_gt_sub, val_X_sub, val_X_rgb_sub = preprocess_augmentation(val_gt_sub, val_X_sub, augment_ON=False, color2grey = input_is_grey)
                        acc_val, predict_val = val_fn(val_X_sub, similNet_features.reshape(-1,similNet_features.shape[-1]), val_gt_sub)
                        # acc_val, predict_val = fuseNet_val_fn(val_X_sub, val_gt_sub)
                    else:
                        selected_viewPairs = select_M_viewPairs_from_N_randViews_for_N_samples(params_volume.__view_set, params_volume.__N_randViews4val, \
                                params_volume.__N_select_viewPairs2val, cubes_visib = val_gt_visib[selected] if params_volume.__only_nonOcclud_cubes else None)
                        # selected_viewPairs = np.random.choice(params_volume.__view_set, (params_volume.__chunk_len_val,params_volume.__N_select_viewPairs2val,2))
                        val_X_sub = gen_coloredCubes(selected_viewPairs = selected_viewPairs, occupiedCubes_param = val_gt_param[selected], \
                                cameraPOs=cameraPOs, models_img=images_list, visualization_ON = False, occupiedCubes_01 = val_gt_sub)                    
                        val_gt_sub, val_X_sub, val_X_rgb_sub = preprocess_augmentation(val_gt_sub, val_X_sub, augment_ON=False, color2grey = input_is_grey)
                        acc_val, predict_val = val_fn(val_X_sub, val_gt_sub)
                        
                        
                    acc_val_batches.append(acc_val)   
                    if params_volume.__val_visualize_ON :
                        X_1, X_2 = [val_X_rgb_sub[0:params_volume.__N_select_viewPairs2val], val_gt_sub[0:1]]
                        result = predict_val[0]
                        X_1 += 0 #params_volume.__CHANNEL_MEAN # [None,:,None,None,None]   #(X_1+.5)*255. 
                        tmp_5D = np.copy(X_1) # used for visualize the surface part of the colored cubes
                        # if want to visualize the result, just stop at the following code, and in the debug probe run the visualize_N_densities_pcl func.
                        # rimember to 'enter' before continue the program~
                        tmp_5D[:,:,X_2[0].squeeze()==0]=0 
                        if not params_volume.__train_ON:
                            visualize_N_densities_pcl([X_2[0]*params_volume.__surfPredict_scale4visual, result*params_volume.__surfPredict_scale4visual, tmp_5D[0,3:], tmp_5D[0,:3], X_1[0,3:], X_1[0,:3]])
                acc_val = np.asarray(acc_val_batches).mean()
                print("val_acc %g" %(acc_val))

            if (epoch % params_volume.__every_N_epoch_2saveModel) == 0:
                save_entire_model(net[params_volume.__layer_2_save_model], '2D_2_3D-{}-{:0.3}_{:0.3}.model'.format(epoch, np.asarray(acc_train_batches).mean(), acc_val))             



