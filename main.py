import numpy as np
import sys
import os

import params
import main_reconstruct
sys.path.append("./utils")
import adapthresh

# import main_train

if __name__ == "__main__":

    print "\ncurrent mode *** {} ***\n".format(params.whatUWant)

    if params.whatUWant is 'reconstruct_model':
        for _model in params.__modelList:
            datasetFolder, imgNamePattern, poseNamePattern, initialPtsNamePattern, N_viewPairs4inference_list, resol, BB, viewList = params.load_modelSpecific_params(datasetName=params.__datasetName, model=_model)
            for N_viewPairs4inference in N_viewPairs4inference_list: # easy to loop through parameters

                ###############
                # fix threshold
                ###############

                print('\n ********* \n model {}, N_viewPairs4inference = {} \n ********* \n'.format(_model, N_viewPairs4inference))
                outputFolder = os.path.join(params.__output_data_rootFld, '{}_s{}/{}_{}views_Nv{}_resol{:0.3}/'.format(params.__datasetName, params.__cube_D, _model, len(viewList), N_viewPairs4inference, resol))
                save_npz_file_path = main_reconstruct.reconstruction(datasetFolder, _model, 
                        imgNamePattern, poseNamePattern, initialPtsNamePattern, 
                        outputFolder, N_viewPairs4inference, resol, BB, viewList)
                # save_npz_file_path  = '/home/mengqi//fileserver/results/MVS/SurfaceNet/DTU/9_[5]_0.4/model9-49views.npz'
                

                ####################
                # adaptive threshold
                ####################

                if not os.path.exists(save_npz_file_path):
                    continue  # Don't exist
                kwargs = {'init_probThresh': 0.5, 'min_probThresh': 0.5, 'max_probThresh': 0.9, 'D_cube': params.__cube_Dcenter, \
                        'N_refine_iter': params.__N_refine_iter, 'save_result_fld': os.path.dirname(save_npz_file_path), \
                        'rayPool_thresh': int(round(params.__gamma * N_viewPairs4inference*2)), 'beta': params.__beta, 'gamma': params.__gamma, \
                        'RGB_visual_ply': False or params.__DEBUG_output_data_rootFld_exists, 'npz_file': save_npz_file_path }
                ply_filePath = adapthresh.adapthresh(**kwargs)

    elif params.whatUWant is 'train_model':
        # main_train.train()
        pass
