import numpy as np

import CVC 
import sparseCubes



def __cluster_inCube__(vxl_ijk_list, vxl_mask_list=[]):
    """
    eliminate the floating noise in each voxel cube.
    Clustering in each cube, only keep the largest cluster.

    -------
    inputs:
        vxl_ijk_list: [(N_pts, 3), ...] ijk of point set
        vxl_mask_list: used to filter voxels

    -------
    outputs:
        indexClusters_list: list of pts index, the ones in the same cluster are gethoded together.
                The index are the left pts after masking.
        clusters_1stIndex_list: lenght equals to N_clusters, the nth index is corresponding to the start pt of the nth cluster in indexClusters_list.

    -------
    examples:
    >>> vxl_ijk_list = [np.array([[1,0,0], [2,2,2], [3,3,3], [1,0,1], [2,3,3], [0,3,3]]), \
                        np.array([[0,2,3], [0,1,0], [0,0,0], [0,3,3]]), \
                        np.array([[0,2,3], [0,1,0], [0,2,3]]), \
                        np.array([[0,2,3], [0,1,3], [0,0,0], [0,3,3], [3,3,3]], dtype=np.uint8) ]
    >>> vxl_mask_list = [np.array([1,0,1,1,1,1], dtype=np.bool), \
                         np.array([1,1,0,1], dtype=np.bool), \
                         np.array([0,0,0], dtype=np.bool), \
                         np.array([1,1,1,1,1], dtype=np.bool)]
    >>> __cluster_inCube__(vxl_ijk_list, vxl_mask_list)     # the masked index should not appear
    ([[0, 2, 1, 3, 4], [0, 2, 1], [], [0, 1, 3, 2, 4]], [[0, 2, 4], [0, 2], [], [0, 3, 4]])
    """

    indexClusters_list, clusters_1stIndex_list = [], []
    for _cube, _select in enumerate(vxl_mask_list):
        if _select.sum() == 0:
            indexClusters_list.append([])
            clusters_1stIndex_list.append([])
            continue
        vxl_ijk = vxl_ijk_list[_cube][_select].astype(np.int64)  # (N_pts, 3)
        adjacencyMatrix_ijk = np.abs((vxl_ijk[None, :] - vxl_ijk[:, None]))  # (1, N_pts, 3) - (N_pts, 1, 3) --> (N_pts, N_pts, 3)
        adjacencyMatrix = np.sum(adjacencyMatrix_ijk <= 1, axis=-1) == 3    # large distance --> False --> sum along ijk cannot reach 3.
        indexClusters, clusters_1stIndex = CVC.clusterFromAdjacency(adjacencyMatrix)   # (N_pts, N_pts) --> ([N_pts indexes], [cluster 1st index])
        # make sure indexClusters don't have duplicates.
        if len(indexClusters) != len(set(indexClusters)):
            raise Warning("There should not exist duplicates. Current is {}".format(indexClusters))
        indexClusters_list.append(indexClusters)
        clusters_1stIndex_list.append(clusters_1stIndex)

    return indexClusters_list, clusters_1stIndex_list




def __denoise_inCube__(vxl_overlappingMask_list, indexClusters_list, clusters_1stIndex_list, vxl_mask_list):
    """
    remove the non-overlapping, make sure the N_cluster will not vanish.
    The overlappingMask is checked with the neighboring cubes.

    -------
    inputs:
        vxl_overlappingMask_list: mark the overlapping status of ALL the voxels.
        vxl_mask_list: all the following index are based on the remaining voxels.
        indexClusters_list: the voxels of a same cluster forms a index sequence.
        clusters_1stIndex_list: indicate the start point of each cluster index sequence.

    -------
    outputs:
        vxl_denoiseMask_list: updated vxl_mask_list, must be the subset of original vxl_mask_list

    -------
    examples:
    >>> vxl_overlappingMask_list = [np.array([1,0,1,0,1], dtype=np.bool), \
                                    np.array([1,0,0,0,1,1], dtype=np.bool), \
                                    np.array([0,0,0,0], dtype=np.bool)] 
    >>> indexClusters_list = [[0, 2, 1, 3], [0, 2, 4, 1, 3], [0, 1, 2, 3]]
    >>> clusters_1stIndex_list = [[0, 1], [0, 3, 4], [0, 1, 2]]
    >>> vxl_mask_list = [np.array([0,1,1,1,1], dtype=np.bool), \
                         np.array([1,1,1,0,1,1], dtype=np.bool), \
                         np.array([1,1,1,1], dtype=np.bool)] 
    >>> __denoise_inCube__(vxl_overlappingMask_list, indexClusters_list, clusters_1stIndex_list, vxl_mask_list)
    [array([False, False,  True,  True,  True], dtype=bool), array([ True, False,  True, False,  True,  True], dtype=bool), array([False, False,  True,  True], dtype=bool)]
    """

    vxl_denoiseMask_list = []
    for _cube, _select in enumerate(vxl_mask_list):
        if _select.sum() == 0:
            vxl_denoiseMask_list.append(_select)
            continue
        vxl_overlappingMask = vxl_overlappingMask_list[_cube][_select]
        indexClusters = indexClusters_list[_cube] 
        clusters_1stIndex = clusters_1stIndex_list[_cube]
        if (_select.sum() != vxl_overlappingMask.size) or (_select.sum() != len(indexClusters)):
            raise Warning("vxl_mask_list[i].sum() should == vxl_overlappingMask_list[i].size == len(indexClusters_list[i]).")
        vxlIndexes, = np.where(_select)  # (N_vxls,) used to update _select

        # for each cluster, check whether there is an overlapping voxel.
        noOverlappingCluster = True
        clusterVxls_withOverlapping = np.zeros(_select.shape, dtype=np.bool)
        clustersSize = []
        for _cluster, _cluster_1stIndex in enumerate(clusters_1stIndex):
            final_cluster = _cluster_1stIndex == clusters_1stIndex[-1]
            _cluster_nextIndex = len(indexClusters) if final_cluster else clusters_1stIndex[_cluster + 1]
            clustersSize.append(_cluster_nextIndex - _cluster_1stIndex)
            # if there is any overlapping voxel
            if vxl_overlappingMask[indexClusters[_cluster_1stIndex : _cluster_nextIndex]].max() == 1:
                noOverlappingCluster = False
                clusterVxls_withOverlapping[vxlIndexes[indexClusters[_cluster_1stIndex : _cluster_nextIndex]]] = 1

        # If all the clusters don't have overlapping, remain the largest cluster.
        if noOverlappingCluster:
            clusterIndex_argmax = np.argmax(clustersSize)
            _cluster_1stIndex = clusters_1stIndex[clusterIndex_argmax]
            _cluster_nextIndex = len(indexClusters) if clusterIndex_argmax == clusters_1stIndex[-1] else clusters_1stIndex[clusterIndex_argmax + 1]
            clusterVxls_withOverlapping[vxlIndexes[indexClusters[_cluster_1stIndex : _cluster_nextIndex]]] = 1
        if (clusterVxls_withOverlapping.astype(np.int8) - _select.astype(np.int8)).max() != 0:
            raise Warning("clusterVxls_withOverlapping should not select the voxels that is not contained int he _select")
        vxl_denoiseMask_list.append(clusterVxls_withOverlapping)
    return vxl_denoiseMask_list



def __mark_overlappingVxls__(cube_ijk_np, vxl_ijk_list, vxl_mask_list, D_cube):
    """
    mark the overlapping voxels in each cube.
    Note the calculation efficiency: for a cube pair, only need to calculate once.

    -------
    inputs:
        cube_ijk_np: ijk of cubes
        vxl_ijk_list: [(N_pts, 3), ...] ijk of point set
        probThresh_list: 
        vxl_mask_list: used to filter voxels
        D_cube: size of cube = (D_cube, )*3

    -------
    outputs:
        vxl_overlappingMask_list: mark the overlapping status of ALL the voxels. Should have the same shape with vxl_mask_list.

    -------
    examples:
    >>> cube_ijk_np = np.array([[1,6,8], [2,6,8], [2,7,8], [2,5,8]], dtype=np.uint8)
    >>> vxl_ijk_list = [np.array([[1,0,0], [2,2,2], [3,3,3], [1,0,1], [2,3,3]], dtype=np.uint8), \
                        np.array([[0,2,3], [0,1,3], [0,0,0], [0,3,3], [3,3,0]], dtype=np.uint8), \
                        np.array([[0,2,3], [0,1,3], [0,0,0], [0,3,3]], dtype=np.uint8), \
                        np.array([[0,2,3], [0,1,3], [0,0,0], [0,3,3], [3,3,3]], dtype=np.uint8) ]
    >>> vxl_mask_list = [np.array([1,0,1,1,1], dtype=np.bool), \
                         np.array([1,1,0,1,1], dtype=np.bool), \
                         np.array([0,0,0,0], dtype=np.bool), \
                         np.array([1,1,1,1,1], dtype=np.bool)]
    >>> __mark_overlappingVxls__(cube_ijk_np, vxl_ijk_list, vxl_mask_list, D_cube = 4)
    [array([False, False, False, False,  True], dtype=bool), array([False,  True, False,  True, False], dtype=bool), array([False, False, False, False], dtype=bool), array([False, False, False,  True, False], dtype=bool)]
    """

    vxl_overlappingMask_list = []
    cube_ijk2index = {}
    neigh_shifts = np.asarray([[1,0,0],[0,1,0],[0,0,1],[-1,0,0],[0,-1,0],[0,0,-1]]).astype(np.int8)

    # cube_ijk2index = {tuple(_ijk): _n for _n, _ijk in enumerate(cube_ijk_np)}
    for _n, _ijk in enumerate(cube_ijk_np): 
        if vxl_mask_list[_n].sum() > 0:     # ignore the obviously empty cubes
            cube_ijk2index.update({tuple(_ijk): _n})

    for _n, _ijk in enumerate(cube_ijk_np):
        if not cube_ijk2index.has_key(tuple(_ijk)):
            vxl_overlappingMask_list.append(vxl_mask_list[_n])
            continue # this cube is filtered in the very beginning
        i_current = cube_ijk2index[tuple(_ijk)]
        vxl_mask_current = vxl_mask_list[i_current]
        vxl_overlappingMask = np.zeros(vxl_mask_current.shape, dtype=np.bool)
        for _ijk_shift in neigh_shifts:
            ijk_neigh = _ijk + _ijk_shift
            if not cube_ijk2index.has_key(tuple(ijk_neigh)):     
                continue
            else:   # exist the overlapping cube
                i_neigh = cube_ijk2index[tuple(ijk_neigh)]
                vxl_ijk_neigh = vxl_ijk_list[i_neigh].astype(np.int64)   # even consider the masked vxls to keep the order. 
                vxl_ijk_newCoords_neigh = vxl_ijk_neigh + (D_cube / 2) * _ijk_shift   # (N_vxls, 3)

                vxl_ijk_current = vxl_ijk_list[i_current].astype(np.int64)
                # (N_vxls_current, 1, 3), (1, N_vxls_neigh, 3) --> (N_vxls_current, N_vxls_neigh) --> (N_vxls_current, )
                overlappingMatrix = np.sum(np.abs((vxl_ijk_current[:,None] - vxl_ijk_newCoords_neigh[None,])), axis=-1)

                # set to any positive value indicating that overlapping vxls don't exist. Since these vxls are masked.
                overlappingMatrix[~vxl_mask_current] = 11     
                overlappingMatrix[:, ~vxl_mask_list[i_neigh]] = 11

                overlapping_vxl_index = np.where(np.min(overlappingMatrix, axis=-1) == 0)  
                vxl_overlappingMask[overlapping_vxl_index] = 1
        vxl_overlappingMask_list.append(vxl_overlappingMask)

    return vxl_overlappingMask_list



def denoise_crossCubes(cube_ijk_np, vxl_ijk_list, vxl_mask_list, D_cube):
    """

    -------
    inputs:
        cube_ijk_np: ijk of cubes
        vxl_ijk_list: [(N_pts, 3), ...] ijk of point set
        probThresh_list: 
        vxl_mask_list: used to filter voxels
        D_cube: size of cube = (D_cube, )*3

    -------
    outputs:
        vxl_overlappingMask_list: mark the overlapping status of ALL the voxels. Should have the same shape with vxl_mask_list.

    -------
    examples:
    >>> cube_ijk_np = np.array([[1,6,8], [2,6,8], [2,7,8], [2,5,8]], dtype=np.uint8)
    >>> vxl_ijk_list = [np.array([[1,0,0], [2,2,2], [3,3,3], [1,0,1], [2,3,3]], dtype=np.uint8), \
                        np.array([[0,2,3], [0,1,3], [0,0,0], [0,3,3], [3,3,0]], dtype=np.uint8), \
                        np.array([[0,2,3], [0,1,3], [0,0,0], [0,3,3]], dtype=np.uint8), \
                        np.array([[0,2,3], [0,1,3], [0,0,0], [0,3,3], [3,3,3]], dtype=np.uint8) ]
    >>> vxl_mask_list = [np.array([1,0,1,1,1], dtype=np.bool), \
                         np.array([1,1,0,1,1], dtype=np.bool), \
                         np.array([0,0,0,0], dtype=np.bool), \
                         np.array([1,1,1,1,1], dtype=np.bool)]
    >>> denoise_crossCubes(cube_ijk_np, vxl_ijk_list, vxl_mask_list, D_cube = 4)
    [array([False, False,  True, False,  True], dtype=bool), array([ True,  True, False,  True, False], dtype=bool), array([False, False, False, False], dtype=bool), array([ True,  True, False,  True, False], dtype=bool)]
    """

    indexClusters_list, clusters_1stIndex_list = __cluster_inCube__(vxl_ijk_list, vxl_mask_list=vxl_mask_list)

    vxl_overlappingMask_list = __mark_overlappingVxls__(cube_ijk_np, vxl_ijk_list, vxl_mask_list, D_cube)
    vxl_maskDenoise_list = __denoise_inCube__(vxl_overlappingMask_list, indexClusters_list, clusters_1stIndex_list, vxl_mask_list)
    return vxl_maskDenoise_list







import doctest
doctest.testmod()
