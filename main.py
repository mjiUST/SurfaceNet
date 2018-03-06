import numpy as np
import sys
import os

import params
import main_train
import main_reconstruct
sys.path.append("./utils")
import adapthresh

# import main_train

if __name__ == "__main__":

    print "\ncurrent mode *** {} ***\n".format(params.whatUWant)

    if params.whatUWant is 'reconstruct_model':
        for _model in params.__modelList:
            datasetFolder, imgNamePattern, poseNamePattern, N_viewPairs4inference_list, resol, BB, viewList = params.load_modelSpecific_params(datasetName=params.__datasetName, model=_model)
            for N_viewPairs4inference in N_viewPairs4inference_list: # easy to loop through parameters


                #################
                # fix threshold #
                #################

                print('\n ********* \n model {}, N_viewPairs4inference = {} \n ********* \n'.format(_model, N_viewPairs4inference))
                outputFolder = os.path.join(params.__output_data_rootFld, '{}_s{}/{}_{}views_Nv{}_resol{:0.3}/'.format(params.__datasetName, params.__cube_D, _model, len(viewList), N_viewPairs4inference, resol))
                save_npz_file_path = main_reconstruct.reconstruction(datasetFolder, _model, imgNamePattern, poseNamePattern, outputFolder, N_viewPairs4inference, resol, BB, viewList)
                # save_npz_file_path  = '/home/mengqi//fileserver/results/MVS/SurfaceNet/DTU/9_[5]_0.4/model9-49views.npz'
                

                ######################
                # adaptive threshold #
                ######################

                kwargs = {'init_probThresh': 0.5, 'min_probThresh': 0.5, 'max_probThresh': 0.9, 'D_cube': params.__cube_Dcenter, \
                        'N_refine_iter': params.__N_refine_iter, 'save_result_fld': os.path.dirname(save_npz_file_path), \
                        'rayPool_thresh': int(round(params.__gamma * N_viewPairs4inference*2)), 'beta': params.__beta, 'gamma': params.__gamma, \
                        'RGB_visual_ply': False or params.__DEBUG_output_data_rootFld_exists, 'npz_file': save_npz_file_path }
                ply_filePath = adapthresh.adapthresh(**kwargs)


    elif params.whatUWant is 'train_model':

        kwargs = {}
        # load camera params and images, which don't change in the training process
        kwargs.update(main_train.loadFixedVar_4training())


        ####################
        # train SurfaceNet #
        ####################
        N_modelCubes_train = 100 # Sample such NO. of cubes for each cube. 1000 is too large to loop through all the training models in reasonable time.
        N_modelCubes_val = 100

        # by default, load the saved model (or None)
        SurfaceNet_model_path = params.__pretrained_SurfaceNet_model_path
        if params.__train_SurfaceNet_wo_offSurfacePts: # If False, need to specify the pretrained model, otherwise will trainfrom scratch

            if params.__define_fns:  # can turn off to debug other part
                kwargs.update(main_train.load_dnn_fns(
                        with_relativeImpt = False, 
                        SurfaceNet_model_path = SurfaceNet_model_path))

            SurfaceNet_model_path = main_train.train(N_on_off_surfacePts_train = [N_modelCubes_val, N_modelCubes_val] if params.__sameTrainValSamples4visual else [N_modelCubes_train, 0], trainingStage = 0,
                    layer_2_save_model = params.__layer_2_save_SurfaceNet, N_epoch = params.__N_epoches[0],
                    N_on_off_surfacePts_val = [N_modelCubes_val, N_modelCubes_val], **kwargs)  # use the save validation data for different stages for comparison


        if params.__train_SurfaceNet_with_offSurfacePts:

            if params.__define_fns:  # can turn off to debug other part
                kwargs.update(main_train.load_dnn_fns(
                        with_relativeImpt = False, 
                        SurfaceNet_model_path = SurfaceNet_model_path))

            SurfaceNet_model_path = main_train.train(N_on_off_surfacePts_train = [N_modelCubes_train, N_modelCubes_train], trainingStage = 1,
                    layer_2_save_model = params.__layer_2_save_SurfaceNet, N_epoch = params.__N_epoches[1],
                    N_on_off_surfacePts_val = [N_modelCubes_val, N_modelCubes_val], **kwargs)


        #######################################
        # finetune SurfaceNet + SimilarityNet #
        #######################################
        
        if params.__train_SurfaceNet_with_SimilarityNet:

            if params.__define_fns:  # can turn off for debug
                kwargs.update(main_train.load_dnn_fns(
                        with_relativeImpt = True, 
                        SurfaceNet_model_path = SurfaceNet_model_path, 
                        SimilarityNet_model_path = params.__pretrained_similNet_model_path))

            main_train.train(N_on_off_surfacePts_train = [N_modelCubes_train, N_modelCubes_train], trainingStage = 2,
                    layer_2_save_model = params.__layer_2_save_fusionNet, N_epoch = params.__N_epoches[2],
                    N_on_off_surfacePts_val = [N_modelCubes_val, N_modelCubes_val], **kwargs)







