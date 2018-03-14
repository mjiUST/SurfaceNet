import numpy as np #import params
import sys
import os
import math
import time
import progressbar

import params
sys.path.append("./utils")
import binarization
import image
import camera
import utils
import sparseCubes
sys.path.append("./nets")
import scene
import similarityNet
import SurfaceNet
import earlyRejection
import viewPairSelection
import CVC
import denoising





def reconstruction(datasetFolder, model, imgNamePattern, poseNamePattern, initialPtsNamePattern, outputFolder, N_viewPairs4inference, resol, BB, viewList):
    """
    pipeline for reconstruction
    inputs: 
        datasetFolder:
        model: 2 / 5 / "dinoSparseRing" ...
        imgNamePattern:
        poseNamePattern:
        initialPtsNamePattern: if None: initialize from BB; otherwise initialize from this initialPts
        outputFolder:
        N_viewPairs4inference:
        resol:
        BB:
        viewList:
    outputs:
    """

    print "start reconstruction ..."  

    cube_D = params.__cube_D
    # from now on, view indexes will be like [0,1,...]
    images_list = image.readImages(datasetFolder = datasetFolder, imgNamePattern = imgNamePattern, viewList = viewList, return_list = True)     
    cameraPOs_np = camera.readCameraPOs_as_np(datasetFolder = datasetFolder, datasetName = params.__datasetName, poseNamePattern = poseNamePattern, model = model, viewList = viewList)  # (N_views, 3, 4) np
    cameraTs_np = camera.cameraPs2Ts(cameraPOs = cameraPOs_np)  # (N_views, 3) np
    if initialPtsNamePattern is None:
        cubes_param_np, cube_D_mm = scene.initializeCubes(resol = resol, cube_D = cube_D, 
                cube_Dcenter = params.__cube_Dcenter, 
                cube_overlapping_ratio = params.__cube_overlapping_ratio, BB = BB)  # (N_cubes,N_params), scalar. the scene is divided into multiple overlapping cubes, each of which has several attributes, such as param_np["xyz"/"ijk"/"resol"]
    else:
        initial_pts_xyz = scene.readPointCloud_xyz(pointCloudFile = os.path.join(datasetFolder, initialPtsNamePattern))
        cubes_param_np, cube_D_mm = scene.quantizePts2Cubes(pts_xyz = initial_pts_xyz, resol = resol, cube_D = cube_D, \
              cube_Dcenter = params.__cube_Dcenter,
              cube_overlapping_ratio = params.__cube_overlapping_ratio, BB = BB)
    sparseCubes.save2ply(os.path.join(outputFolder, 'initialCubes.ply'), xyz_np = cubes_param_np['xyz'] + cube_D_mm/2)  # save the cube positions to ply file
    img_h_cubesCorner, img_w_cubesCorner = camera.perspectiveProj_cubesCorner(projection_M = cameraPOs_np, cube_xyz_min = cubes_param_np['xyz'], cube_D_mm = cube_D_mm, return_int_hw = False, return_depth = False)       # img_w/h_cubesCorner (N_views, N_cubes, 8)
    img_h_cubesCenter, img_w_cubesCenter = camera.perspectiveProj(projection_M = cameraPOs_np, \
            xyz_3D = cubes_param_np['xyz'] + cube_D_mm/2, \
            return_int_hw = False, return_depth = False)    # img_w/h: (N_Ms, N_pts) 
    N_views, N_cubes = img_h_cubesCorner.shape[:2]
    D_embedding = params.__D_imgPatchEmbedding 

    # define and load similarityNet
    patch2embedding_fn, embeddingPair2simil_fn = similarityNet.similarityNet_inference(model_file = params.__pretrained_similNet_model_file, \
            imgPatch_hw_size = (params.__imgPatch_hw_size, )*2 )
    viewPair_relativeImpt_fn, nViewPair_SurfaceNet_fn = SurfaceNet.SurfaceNet_inference(N_viewPairs4inference = N_viewPairs4inference, model_file = params.__pretrained_SurfaceNet_model_file, layerNameList_2_load = params.__layerNameList_2_load)


    #################
    # early rejection
    #################

    # patches generation --> patch embedding
    viewPairs = utils.k_combination_np(range(N_views), k = 2)     # (N_viewPairs, 2)
    N_viewPairs = viewPairs.shape[0]
    patches_mean_bgr = params.__MEAN_PATCHES_BGR
    patches_embedding, inScope_cubes_vs_views = earlyRejection.patch2embedding( \
            images_list, img_h_cubesCorner, img_w_cubesCorner, patch2embedding_fn, patches_mean_bgr, \
            N_cubes, N_views, D_embedding, patchSize = params.__imgPatch_hw_size, \
            batchSize = params.__batchSize_similNet_patch2embedding, \
            cubeCenter_hw = np.stack([img_h_cubesCenter, img_w_cubesCenter], axis=0))    # (N_cubes, N_views, D_embedding), (N_cubes, N_views)

    dissimilarity = earlyRejection.embeddingPairs2simil(embeddings = patches_embedding, 
            embeddingPair2simil_fn = embeddingPair2simil_fn, 
            inScope_cubes_vs_views = inScope_cubes_vs_views, 
            viewPairs = viewPairs, 
            N_views = N_views,
            batchSize = params.__batchSize_similNet_embeddingPair2simil)   # (N_cubes, N_viewPairs), TODO: need to set the dissimil value of the viewPairs with at least one invalid_view to 0.
    validCubes = earlyRejection.selectFromSimilarity(dissimilarityProb = dissimilarity, N_viewPairs4inference = N_viewPairs4inference)    # (N_cubes,) np.bool
    N_validCubes = validCubes.sum()
    print("\nEarly rejection step reduced the # of cubes from {} to {}.".format(N_cubes, N_validCubes))

    ####################
    # viewPair selection
    ####################

    viewPairs4Reconstr, w_viewPairs4Reconstr = viewPairSelection.viewPairSelection( \
            cameraTs_np = cameraTs_np,  \
            e_viewPairs = patches_embedding,  \
            d_viewPairs = dissimilarity,  \
            validCubes = validCubes,  \
            cubeCenters_xyz = cubes_param_np['xyz'] + cube_D_mm / 2., \
            viewPair_relativeImpt_fn = viewPair_relativeImpt_fn,  \
            batchSize = params.__batchSize_viewPair_w,  \
            N_viewPairs4inference = N_viewPairs4inference, \
            viewPairs = viewPairs)     # (N_validCubes, N_viewPairs4inference, 2), (N_validCubes, N_viewPairs4inference)

    if params.__weighted_fusion is False:
        w_viewPairs4Reconstr[:] = 1.0 / N_viewPairs4inference   # (N_validCubes, N_viewPairs4inference)

    ######################
    # SurfaceNet inference
    ######################

    # TODO: to polish the code
    print("SurfaceNet inference process ...")
    prediction_list, rgb_list, vxl_ijk_list, rayPooling_votes_list = [], [], [], []
    cube_ijk_np, param_np, viewPair_np = None, None, None

    batchSelectors_list = utils.gen_non0Batch_npBool(boolIndicators = validCubes, batch_size = params.__batchSize_nViewPair_SurfaceNet)
    N_batches = len(batchSelectors_list)
    if N_batches == 0:
        return "Empty!"
    bar = progressbar.ProgressBar(maxval=N_batches, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    for _i, _batch in enumerate(batchSelectors_list):      # note that bool selector: _batch.shape == (N_cubes,). 
        # TODO: in the test process, the generated coloredCubes could be the exact size we want. Don't need to crop in the preprocess method. 
        _CVCs1_sub = CVC.gen_coloredCubes( \
                selected_viewPairs = viewPairs4Reconstr[_batch[validCubes]],  \
                xyz = cubes_param_np['xyz'][_batch],  \
                resol = cubes_param_np['resol'][_batch],  \
                colorize_cube_D = cube_D,\
                cameraPOs=cameraPOs_np, \
                models_img=images_list, \
                visualization_ON = False)   # ((N_cubeSub * N_viewPairs4inference, 3 * 2) + (D_CVC,) * 3) 5D

        _, _CVCs2_sub = CVC.preprocess_augmentation(None, _CVCs1_sub, mean_rgb = params.__MEAN_CVC_RGBRGB[None,:,None,None,None], augment_ON=False, crop_ON = False)
        # TODO: eliminate the 'if' condition
        surfacePrediction, unfused_predictions = nViewPair_SurfaceNet_fn(_CVCs2_sub) if N_viewPairs4inference == 1 \
                                else nViewPair_SurfaceNet_fn(_CVCs2_sub, w_viewPairs4Reconstr[_batch[validCubes]])


        # save the intermedian results
        _CVCs2_sub += params.__MEAN_CVC_RGBRGB[None,:,None,None,None]
        _CVCs_sub_weighted = utils.generate_voxelLevelWeighted_coloredCubes(viewPair_coloredCubes = _CVCs2_sub, \
                viewPair_surf_predictions = unfused_predictions, weight4viewPair = w_viewPairs4Reconstr[_batch[validCubes]])

        updated_sparse_list_np = sparseCubes.append_dense_2sparseList( \
                prediction_sub = surfacePrediction, rgb_sub = _CVCs_sub_weighted, param_sub = cubes_param_np[_batch],\
                viewPair_sub = viewPairs4Reconstr[_batch[validCubes]], min_prob = params.__min_prob, rayPool_thresh = 0,\
                enable_centerCrop = True, cube_Dcenter = params.__cube_Dcenter,\
                enable_rayPooling = True, cameraPOs = cameraPOs_np, cameraTs = cameraTs_np, \
                prediction_list = prediction_list, rgb_list = rgb_list, vxl_ijk_list = vxl_ijk_list, \
                rayPooling_votes_list = rayPooling_votes_list, \
                cube_ijk_np = cube_ijk_np, param_np = param_np, viewPair_np = viewPair_np)
        prediction_list, rgb_list, vxl_ijk_list, rayPooling_votes_list, cube_ijk_np, param_np, viewPair_np = updated_sparse_list_np
        if sys.stdout.isatty():     # running in terminal
            bar.update(_i+1)
        else:   # if the results are redirected
            print("batch {} / {}".format(_i, N_batches))

    bar.finish()

    time_ply = time.time()
    ply_filename = os.path.join(outputFolder, 'fixThresh_tau{:.3}_gamma{:.3}.ply'.format(params.__tau, params.__gamma))
    vxl_mask_list = sparseCubes.filter_voxels(vxl_mask_list=[],prediction_list=prediction_list, prob_thresh= params.__tau,\
            rayPooling_votes_list=rayPooling_votes_list, rayPool_thresh = params.__gamma * N_viewPairs4inference * 2)    # thinning (ray pooling)
    # TODO: fix thresh thinning (prob_thresh)
    vxl_maskDenoised_list = denoising.denoise_crossCubes(cube_ijk_np, vxl_ijk_list, vxl_mask_list = vxl_mask_list, D_cube = cube_D)
    sparseCubes.save_sparseCubes_2ply(vxl_maskDenoised_list, vxl_ijk_list, rgb_list, \
            param_np, ply_filePath=ply_filename, normal_list=None)
    print("Saved ply file '{}'. It takes {:.3f}s".format(ply_filename, time.time() - time_ply))

    time_npz = time.time()
    save_npz_file_path = os.path.join(outputFolder, 'model{}-{}views.npz'.format(model, N_views))
    sparseCubes.save_sparseCubes(save_npz_file_path, *updated_sparse_list_np)
    print("Saved npz takes {:.3f}s".format(time.time() - time_npz))
    return save_npz_file_path







