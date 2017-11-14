# CVC: Colored Voxel Cube
import copy
import numpy as np


def __colorize_cube__(view_set, cameraPOs_np, model_imgs_np, xyz, resol, densityCube, colorize_cube_D, visualization_ON=False):
    """ 
    generate colored cubes of a perticular densityCube  
    inputs: 
    output: [views_N, 3, colorize_cube_D, colorize_cube_D, colorize_cube_D]. 3 is for RGB
    """
    min_x,min_y,min_z = xyz
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
        inScope_pts_indx = (pts_w<max_w) & (pts_h<max_h) & (pts_w>=0) & (pts_h>=0)
        pts_RGB[inScope_pts_indx] = img[pts_h[inScope_pts_indx],pts_w[inScope_pts_indx]]
        colored_cubes[_n] = pts_RGB.T.reshape((3,colorize_cube_D,colorize_cube_D,colorize_cube_D))
        
    if visualization_ON:    
        visualize_N_densities_pcl([densityCube]+[colored_cubes[n] for n in range(0,len(5))])
        

    return colored_cubes


def gen_coloredCubes(selected_viewPairs, xyz, resol, cameraPOs, models_img, colorize_cube_D, visualization_ON = False, \
            occupiedCubes_01=None):     
    """
    inputs: 
    selected_viewPairs: (N_cubes, N_select_viewPairs, 2)
    xyz, resol: parameters for each occupiedCubes (N,params)
    occupiedCubes_01: multiple occupiedCubes (N,)+(colorize_cube_D,)*3
    return:
    coloredCubes = (N*N_select_viewPairs,3*2)+(colorize_cube_D,)*3 
    """
    N_cubes, N_select_viewPairs = selected_viewPairs.shape[:2]
    coloredCubes = np.zeros((N_cubes,N_select_viewPairs*2,3)+(colorize_cube_D,)*3, dtype=np.float32) # reshape at the end

         

    for _n_cube in range(0, N_cubes): ## each cube
        if visualization_ON:
            if occupiedCubes_01 is None:
                print 'error: [func]gen_coloredCubes, occupiedCubes_01 should not be None when visualization_ON==True'
            occupiedCube_01 = occupiedCubes_01[_n_cube]
        else:
            occupiedCube_01 = None
        ##randViewIndx = random.sample(range(1,cameraPOs.shape[0]),N_randViews)
            

        #(N_cubes, N_select_viewPairs, 2) ==> (N_select_viewPairs*2,). 
        selected_views = selected_viewPairs[_n_cube].flatten() 
        # because selected_views could include duplicated views, this case is not the best way. But if the N_select_viewPairs is small, it doesn't matter too much
        coloredCube = __colorize_cube__(view_set = selected_views, \
                cameraPOs_np = cameraPOs, model_imgs_np = models_img, xyz = xyz[_n_cube], resol = resol[_n_cube], \
                visualization_ON=visualization_ON, colorize_cube_D=colorize_cube_D, densityCube = occupiedCube_01)


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



def preprocess_augmentation(gt_sub, X_sub, mean_rgb, augment_ON = True, crop_ON = True):
    # X_sub /= 255.
    X_sub = X_sub.astype(np.float32)
    X_sub -= mean_rgb  ##.5

    if augment_ON:
        X_sub += np.random.randint(-30,30,1) # illumination argmentation
        X_sub += np.random.randint(-5,5,mean_rgb.shape) # color channel argmentation
        gt_sub, X_sub = data_augment_rand_rotate(gt_sub, X_sub) # randly rotate multiple times
        gt_sub, X_sub = data_augment_rand_rotate(gt_sub, X_sub)
        gt_sub, X_sub = data_augment_rand_rotate(gt_sub, X_sub)
        ##gt_sub, X_sub = data_augment_scipy_rand_rotate(gt_sub, X_sub) ## take a lot of time
    if crop_ON:
        gt_sub, X_sub = data_augment_crop([gt_sub, X_sub], random_crop=augment_ON) # smaller size cube       
    return gt_sub, X_sub




import doctest
doctest.testmod()

