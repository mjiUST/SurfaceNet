import numpy as np
import utils
import image


def patch2embedding(images_list, img_h_cubesCorner, img_w_cubesCorner, patch2embedding_fn, patches_mean_bgr, N_cubes, N_views, D_embedding, patchSize, batchSize, cubeCenter_hw):
    """
    given the imgs and the cubeCorners' projection, return the patches' embeddings.

    --------------
    inputs:
        images_list: [(img_h,img_w,3/1), (img_h',img_w',3/1), ...]. list of view images. 
        img_h/w_cubesCorner: (N_views, N_cubes, 8). projectioin of the cubes' corners
        patch2embedding_fn: CNN embedding function
        N_cubes: # of cubes
        N_views: # of views
        D_embedding: dim of the embedding

    --------------
    outputs:
        patches_embedding: (N_cubes, N_views, D_embedding), np.float32
        inScope_cubes_vs_views: (N_cubes, N_views), np.bool
    """

    # since the images' size may be different, some numpy array operations upon multiple images cannot be used. Just loop through the view image.
    inScope_cubes_vs_views = np.zeros((N_cubes, N_views), dtype=np.bool)    # bool indicator matrix (N_cubes, N_views)

    patch_allBlack = image.preprocess_patches(np.zeros((1, patchSize, patchSize, 3), dtype = np.float32), mean_BGR = patches_mean_bgr)
    patches_embedding = np.zeros((N_cubes, N_views, D_embedding), dtype=np.float32)     # (N_cubes, N_views, D_embedding)
    # (1, 3, patchSize, patchSize) --> (N_cubes, N_cubes, D_embedding). Use the all black patch's embedding to initialize.
    patches_embedding[:,:] = patch2embedding_fn(patch_allBlack)[0] # don't use np.repeat (out of memory)

    projection_h_range = np.stack([img_h_cubesCorner.min(axis=-1), img_h_cubesCorner.max(axis=-1)], axis=-1)  # (N_views, N_cubes, 8) --> (N_views, N_cubes) --> (N_views, N_cubes, 2)
    projection_w_range = np.stack([img_w_cubesCorner.min(axis=-1), img_w_cubesCorner.max(axis=-1)], axis=-1)

    for _view, _image in enumerate(images_list):      # patch size determined by the similarityNet
        _img_h, _img_w, _img_c = _image.shape
        _inScope = image.img_hw_cubesCorner_inScopeCheck(hw_shape = (_img_h, _img_w), img_h_cubesCorner = img_h_cubesCorner[_view], img_w_cubesCorner = img_w_cubesCorner[_view])   # (N_cubes,) inScope check and select perticular _logRatio_int. 
        inScope_cubes_vs_views[:,_view] = _inScope
        N_cubes_inScope = _inScope.sum()
        if not N_cubes_inScope:   # if there is no inScope patch, just return None.
            continue
        else:
            _patches_embedding_inScope = np.zeros((N_cubes_inScope, D_embedding), dtype=np.float32)     # (N_cubes_inScope, N_views, D_embedding)
            _patches = image.cropImgPatches(img = _image, range_h = projection_h_range[_view][_inScope], range_w = projection_w_range[_view][_inScope], patchSize = patchSize, pyramidRate = 1, interp_order = 2, cubeCenter_hw = cubeCenter_hw[:, _view, _inScope])  # (N_cubes_inScope, patchSize, patchSize, 3/1)
            _patches_preprocessed = image.preprocess_patches(_patches.astype(np.float32), mean_BGR = patches_mean_bgr)
            for _batch in utils.yield_batch_npBool(N_all = N_cubes_inScope, batch_size = batchSize):      # note that bool selector: _batch.shape == (N_cubes,)
                _patches_embedding_inScope[_batch] = patch2embedding_fn(_patches_preprocessed[_batch])     # (N_batch, 3/1, patchSize, patchSize) --> (N_batch, D_embedding). similarityNet: patch --> embedding
            patches_embedding[_inScope, _view] = _patches_embedding_inScope    # try to avoid index chain for assignment: a[xx][xx]=xx
    return patches_embedding, inScope_cubes_vs_views 


def embeddingPairs2simil(embeddings, N_views, inScope_cubes_vs_views, embeddingPair2simil_fn, batchSize, viewPairs):
    """
    given patches' embeddings, return the disimilarity probability map that set the outScope view pairs

    ---------
    inputs:
        embeddings: (N_cubes, N_views, D_embedding)
        inScope_cubes_vs_views:
        embeddingPair2simil_fn:
        batchSize
        viewPairs: (N_viewPairs, 2), 2-combination over range(N_views), such as np.array([[0,1], [0,2], [1,2]])
    """

    D_embedding = embeddings.shape[-1]
    viewPairs = utils.k_combination_np(range(N_views), k = 2)     # (N_viewPairs, 2)
    N_viewPairs = viewPairs.shape[0]
    N_cubes = embeddings.shape[0]
    dissimilarity_1D_list = []
    for _i_cubes, _j_viewPairs in utils.yield_batch_ij_npBool(ij_lists = (range(N_cubes), viewPairs.flatten()), batch_size = int(batchSize*2)):      # (N_batch, ) for each.  Make sure N_batch%2=0, due to the input paris.
        _embeddingPairs_sub = embeddings[_i_cubes, _j_viewPairs] # (N_cubes, N_views, D_embedding) --> (N_batch, D_embedding), only generate when needed, in order to save memory
        dissimilarity_1D_list.append(embeddingPair2simil_fn(_embeddingPairs_sub))   # (N_batch, D_embedding) --> (N_batch/2, )
    dissimilarity = np.vstack(dissimilarity_1D_list).reshape((N_cubes, N_viewPairs)) # (N_cubes, N_viewPairs)

    # Don't mask the invalid viewPairs (with at least 1 outScope view). Let the network to decide how to choose. And robust to the case with N < N_viewPairs4inference possible view pairs.
    # inScope_cubes_vs_viewPairs = inScope_cubes_vs_views[:, viewPairs[:,0]] & inScope_cubes_vs_views[:, viewPairs[:,1]]  # (N_cubes, N_viewPairs)
    # dissimilarity *= inScope_cubes_vs_viewPairs    # (N_cubes, N_viewPairs)
    return dissimilarity

def selectFromSimilarity(dissimilarityProb, N_viewPairs4inference):
    """
    given the dissimilarity probability map, return the bool selection np.
    The rejection stretagy: for each row, there are at least 3

    """

    # TODO: need to change the dissimilarityProb
    similarityBool = (dissimilarityProb < 0.5) & (dissimilarityProb > 0.1)  # remove the case with 2 identical input patches and the case with 1+ outScope view(s).
    selectionBool = (similarityBool.sum(axis=1) >= N_viewPairs4inference).astype(np.bool)  
    return selectionBool



