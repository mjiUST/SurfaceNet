# CVC: Colored Voxel Cube
import copy
import numpy as np
np.random.seed(201801)

import transforms

def __colorize_cube__(view_set, cameraPOs_np, model_imgs_np, min_xyz, resol, densityCube, colorize_cube_D):
    """ 
    generate colored cubes of a perticular densityCube  
    inputs: 
    output: [views_N, 3, colorize_cube_D, colorize_cube_D, colorize_cube_D]. 3 is for RGB
    """
    min_x,min_y,min_z = min_xyz
    indx_xyz = range(0,colorize_cube_D)
    ##meshgrid: indexing : {'xy', 'ij'}, optional     ##Cartesian ('xy', default) or matrix ('ij') indexing of output.    
    indx_x,indx_y,indx_z = np.meshgrid(indx_xyz,indx_xyz,indx_xyz,indexing='ij')  
    indx_x = indx_x * resol + min_x
    indx_y = indx_y * resol + min_y
    indx_z = indx_z * resol + min_z
    homogen_1s = np.ones(colorize_cube_D**3, dtype=np.float64)
    pts_4D = np.vstack([indx_x.flatten(),indx_y.flatten(),indx_z.flatten(),homogen_1s])

    N_views = len(view_set)
    colored_cubes = np.zeros((N_views,3,colorize_cube_D,colorize_cube_D,colorize_cube_D))
    # only chooce from inScope views
    # center_pt_xyz1 = np.asarray([grid_D*resol/2 + min_x, grid_D*resol/2 + min_y, grid_D*resol/2 + min_z, 1])
    # center_pt_3D = np.dot(cameraPOs_np,center_pt_xyz1)
    # center_pt_wh = center_pt_3D[:,:-1] / center_pt_3D[:,-1:]# the result is vector: [w,h,1], w is the first dim!!!
    # valid_views = (center_pt_wh[:,0]<max_w) & (center_pt_wh[:,1]<max_h) & (center_pt_wh[:,0]>0) & (center_pt_wh[:,1]>0)      
    # while valid_views.sum() < N_randViews: ## if only n views can see this pt, where n is smaller than N_randViews, randomly choose some more
    #     valid_views[random.randint(1,cameraPOs_np.shape[0]-1)] = True
    # valid_view_list = list(valid_views.nonzero()[0]) ## because the cameraPOs_np[0] is zero, don't need +1 here
    # view_list = random.sample(valid_view_list,N_randViews)    

    for _n, _view in enumerate(view_set):
        # perspective projection
        projection_M = cameraPOs_np[_view]  ## use viewIndx
        pts_3D = np.dot(projection_M, pts_4D)
        pts_3D[:-1] /= pts_3D[-1] # the result is vector: [w,h,1], w is the first dim!!!
        pts_2D = pts_3D[:-1].round().astype(np.int32)
        pts_w, pts_h = pts_2D[0], pts_2D[1]
        # access rgb of corresponding model_img using pts_2D coordinates
        pts_RGB = np.zeros((colorize_cube_D**3, 3))
        img = model_imgs_np[_view]  ## use viewIndx
        max_h, max_w, _ = img.shape
        # only assign the in scope voxels. Otherwise, simply leave to black.
        inScope_pts_indx = (pts_w<max_w) & (pts_h<max_h) & (pts_w>=0) & (pts_h>=0)
        pts_RGB[inScope_pts_indx] = img[pts_h[inScope_pts_indx],pts_w[inScope_pts_indx]]
        colored_cubes[_n] = pts_RGB.T.reshape((3,colorize_cube_D,colorize_cube_D,colorize_cube_D))
        
    return colored_cubes


def gen_coloredCubes(selected_viewPairs, min_xyz, resol, cameraPOs, model_imgs, colorize_cube_D):     
    """
    inputs: 
    selected_viewPairs: (N_cubes, N_select_viewPairs, 2)
    min_xyz, resol: parameters for each occupiedCubes (N,params)
    return:
    coloredCubes = (N*N_select_viewPairs,3*2)+(colorize_cube_D,)*3 
    """
    N_cubes, N_select_viewPairs = selected_viewPairs.shape[:2]
    coloredCubes = np.zeros((N_cubes,N_select_viewPairs*2,3)+(colorize_cube_D,)*3, dtype=np.float32) # reshape at the end

         

    for _n_cube in range(0, N_cubes): ## each cube
        occupiedCube_01 = None
        ##randViewIndx = random.sample(range(1,cameraPOs.shape[0]),N_randViews)

        #(N_cubes, N_select_viewPairs, 2) ==> (N_select_viewPairs*2,). 
        selected_views = selected_viewPairs[_n_cube].flatten() 
        # because selected_views could include duplicated views, this case is not the best way. But if the N_select_viewPairs is small, it doesn't matter too much
        coloredCube = __colorize_cube__(view_set = selected_views, \
                cameraPOs_np = cameraPOs, model_imgs_np = model_imgs, min_xyz = min_xyz[_n_cube], resol = resol[_n_cube], \
                colorize_cube_D=colorize_cube_D, densityCube = occupiedCube_01)


        # [a,b,c] ==> [a,b,a,c,b,c]
        ##all_pairIndx = ()
        ##for _pairIndx in itertools.combinations(range(0,N_randViews),2):
            ##all_pairIndx += _pairIndx
        ##all_pairIndx = list(all_pairIndx)

        # # [a,b,c,d,e,f,g,h,i,j] ==> [a,b,g,c,f,e]
        # all_pairIndx = []
        # for _pairIndx in itertools.combinations(range(0,N_randViews),2):
        #     all_pairIndx.append(_pairIndx)
        # all_pairIndx = random.sample(all_pairIndx, N_select_viewPairs)
        # all_pairIndx = [x for pair_tuple in all_pairIndx for x in pair_tuple] ## [(a,),(a,b),(a,b,c)] ==> [a,a,b,a,b,c]
        
        coloredCubes[_n_cube] = coloredCube
            
    return coloredCubes.reshape((N_cubes*N_select_viewPairs,3*2)+(colorize_cube_D,)*3)


def gen_models_coloredCubes(viewPairs, cube_params, cameraPOs, models_img_list, cube_D, random_colorCondition = False):
    """
    given different cubes' params (xyz_min, model_index, resolution) & images with cameraPOs
    generate CVCs (N_cubes * N_viewPairs, 3+3, s, s, s), where s = cube_D
    random_colorCondition: If True: randomly select images with different (random) light conditions
            If False: the set of model_imgs is under the same light condition.
    
    outputs:
        images_slice: [(N_views, N_views), ] * N_cubes, used to select from the images[images_slice[i]]
    """

    N_cubes, N_viewPairs = viewPairs.shape[:2]
    cube_D = cube_params[0]['cube_D']  # all the cubes should have the same size (cube_D, )*3
    output_CVC = np.zeros((N_cubes, N_viewPairs, 3+3) + (cube_D, ) * 3)
    images_slice = []
    for _cube in range(N_cubes):
        # cube_param: min_xyz / resolution / cube_D / modelIndex
        _modelIndex = cube_params[_cube]['modelIndex']
        # input (1, N_viewPair, 2) view index, return (N_viewPair * 1, 3+3, s, s, s) 
        _1cube_slice = slice(_cube, _cube+1, 1)
        _model_lights_imgs = models_img_list[_modelIndex] # (N_views, N_lights, H, W, 3)
        _N_views, _N_lights = _model_lights_imgs.shape[:2] # (N_views, N_lights, H, W, 3)

        if random_colorCondition: # each view may have different light condition 
            _light_selector = np.random.randint(0, _N_lights, (_N_views, ))
        else: # the set of model_imgs is under the same light condition.
            _light_selector = np.random.randint(0, _N_lights, (1, )).repeat(_N_views)

        _images_slice = (np.arange(_N_views), _light_selector)
        images_slice.append(_images_slice)
        _model_imgs = _model_lights_imgs[_images_slice] # (N_views, N_lights, H, W, ...) --> (N_views, H, W, ...)
        output_CVC[_cube] = gen_coloredCubes(selected_viewPairs = viewPairs[_1cube_slice],
                min_xyz = cube_params[_1cube_slice]['min_xyz'], 
                resol = cube_params[_1cube_slice]['resolution'], 
                cameraPOs = cameraPOs, 
                model_imgs = _model_imgs, 
                colorize_cube_D = cube_D)
    return output_CVC.reshape((N_cubes * N_viewPairs, 3+3) + (cube_D, ) * 3), images_slice


def data_augment_rand_crop(Xlist, crop_size):
    # random crop on ending 3 dimensions of any tensor with grid_D>=3
    randx,randy,randz = np.random.randint(0,grid_D-crop_size+1,size=(3,))
    #[...,xxx], ... means 
    return [X[...,randx:randx+crop_size,randy:randy+crop_size,randz:randz+crop_size] for X in Xlist]
 


def preprocess_augmentation(gt_sub, X_sub_rgb, mean_rgb, augment_ON = True, 
        crop_ON = True, cube_D = 32, color2grey = False, return_cropped_rgb_np = False):
    """
    gt_sub: (N, 1, D,D,D)
    X_sub_rgb: (N, 6, D,D,D), [0-255] uint8
    if gt_sub == None, only output processed X_np
    """

    X_sub_rgb = X_sub_rgb.astype(np.float32)
    if color2grey:
        X_sub = np.tensordot(X_sub_rgb.reshape(_shape[:1]+(2,3)+_shape[2:]),
                             np.array([0.299,0.587,0.114]).astype(np.float32),
                             axes=([2],[0])) # convert RGB to grey (N,6,D,D,D)==> (N,2,3,D,D,D)==> (N,2,D,D,D)
    else:
        X_sub = np.copy(X_sub_rgb) if return_cropped_rgb_np else X_sub_rgb  # if return rgb, use deep copy


    X_sub -= mean_rgb

    if gt_sub is None:
        input_np_tuple = (X_sub,) 
    else:
        input_np_tuple = (gt_sub, X_sub)  #  ((N,1,D,D,D), (6*N, 3+3,D,D,D))
    output_np_tuple = input_np_tuple # if don't augment and don't crop, direct return

    if augment_ON:
        X_sub += np.random.randint(-30,30,1) # illumination argmentation
        X_sub += np.random.randint(-5,5,mean_rgb.shape) # color channel argmentation
        output_np_tuple = transforms.coTransform_flip( 
                input_np_tuple = input_np_tuple,   # transform together 
                axes = (-3, -2, -1),    # axes need to be flipped
                randomFlip = True)      # randomly select axes to flip
    if crop_ON:
        if return_cropped_rgb_np:
            input_np_tuple += (X_sub_rgb, )
        output_np_tuple = transforms.coTransform_crop(
                input_np_tuple,     # multiple numpy arrays
                output_shape = (-1, -1) + (cube_D, ) * 3,  # (N, 6, D, D, D), -1 will keep the shape of the axis
                randomCrop = augment_ON)

    return list(output_np_tuple)




import doctest
doctest.testmod()

