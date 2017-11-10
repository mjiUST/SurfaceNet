import numpy as np

import CVC 



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
        clusters_1stIndex_list: lenght equals to N_clusters, the nth index is corresponding to the start pt of the nth cluster in indexClusters_list.

    -------
    examples:
    >>> vxl_ijk_list = [np.array([[1,0,0], [2,2,2], [3,3,3], [1,0,1], [2,3,3]]), \
             np.array([[0,2,3], [0,1,0], [0,0,0], [0,3,3]])]
    >>> vxl_mask_list = [np.array([1,0,1,1,1], dtype=np.bool), \
             np.array([1,1,0,1], dtype=np.bool)]
    >>> __cluster_inCube__(vxl_ijk_list, vxl_mask_list)
    ([[0, 2, 1, 3], [0, 2, 1]], [[0, 1], [0, 1]])
    """

    indexClusters_list, clusters_1stIndex_list = [], []
    for _cube, _select in enumerate(vxl_mask_list):
        vxl_ijk = vxl_ijk_list[_cube][_select]  # (N_pts, 3)
        adjacencyMatrix = np.sum(np.abs((vxl_ijk[None, :] - vxl_ijk[:, None])), axis=-1) <= 3  # (1, N_pts, 3) - (N_pts, 1, 3) --> (N_pts, N_pts)
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
        vxl_mask_list

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




def denoise_crossCubes(cube_ijk_np, vxl_ijk_list, vxl_mask_list):
    for _ijk in cube_ijk_np:
        if not cube_ijk2indx.has_key(tuple(_ijk)):
            continue # this cube is filtered in the very beginning
        i_current = cube_ijk2indx[tuple(_ijk)]
        element_cost = np.array([0,0,0]).astype(np.float16)
        for _ijk_shift in neigh_shifts:
            ijk_ovlp = _ijk + _ijk_shift
            # ijk_adjc = _ijk + 2 * _ijk_shift
            exist_ovlp = cube_ijk2indx.has_key(tuple(ijk_ovlp))
            # exist_adjc = cube_ijk2indx.has_key(tuple(ijk_adjc))
            if exist_ovlp:
                i_ovlp = cube_ijk2indx[tuple(ijk_ovlp)]
                tmp_occupancy_ovlp = vxl_ijk_list[i_ovlp][occupied_vxl(i_ovlp, 0)] # this will be changed in the next func.
                partial_occ_ovlp = access_partial_Occupancy_ijk(Occ_ijk=tmp_occupancy_ovlp, \
                        shift=_ijk_shift*-1, D_cube = D_cube)
            else:
                partial_occ_ovlp = np.empty((0,3), dtype=np.uint8)
            for _n_thresh, _thresh_perturb in enumerate(thresh_perturb_list):
                tmp_occupancy_current = vxl_ijk_list[i_current][occupied_vxl(i_current, _thresh_perturb)]# this will be changed in the next func.
                partial_occ_current = access_partial_Occupancy_ijk(Occ_ijk=tmp_occupancy_current, \
                        shift=_ijk_shift, D_cube = D_cube)
                ovlp_AND, ovlp_XOR = sparseOccupancy_AND_XOR(partial_occ_current, partial_occ_ovlp)
                element_cost[_n_thresh] += ovlp_XOR
                if partial_occ_current.shape[0] >= 6:
                    if partial_occ_ovlp.shape[0] >= 6:
                        element_cost[_n_thresh] -= beta * ovlp_AND



import doctest
doctest.testmod()
