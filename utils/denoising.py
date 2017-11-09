import numpy as np

import CVC 



def __clustering_inCube__(vxl_ijk_list, vxl_mask_list=[]):
    """
    eliminate the floating noise in each voxel cube.
    Clustering in each cube, only keep the largest cluster.

    -------
    inputs:
        vxl_ijk_list: [(N_pts, 3), ...] ijk of point set
        vxl_mask_list: used to filter voxels

    -------
    outputs:
        clustersIndexes_list: list of pts index, the ones in the same cluster are gethoded together.
        clusters_1stIndex_list: lenght equals to N_clusters, the nth index is corresponding to the start pt of the nth cluster in clustersIndexes_list.

    -------
    examples:
    >>> vxl_ijk_list = [np.array([[1,0,0], [2,2,2], [3,3,3], [1,0,1], [2,3,3]]), \
             np.array([[0,2,3], [0,1,0], [0,0,0], [0,3,3]])]
    >>> vxl_mask_list = [np.array([1,0,1,1,1], dtype=np.bool), \
             np.array([1,1,0,1], dtype=np.bool)]
    >>> __clustering_inCube__(vxl_ijk_list, vxl_mask_list)
    ([[0, 2, 1, 3], [0, 2, 1]], [[0, 1], [0, 1]])
    """

    clustersIndexes_list, clusters_1stIndex_list = [], []
    for _cube, _select in enumerate(vxl_mask_list):
        vxl_ijk = vxl_ijk_list[_cube][_select]  # (N_pts, 3)
        adjacencyMatrix = np.sum(np.abs((vxl_ijk[None, :] - vxl_ijk[:, None])), axis=-1) <= 3  # (1, N_pts, 3) - (N_pts, 1, 3) --> (N_pts, N_pts)
        clustersIndexes, clusters_1stIndex = CVC.clusteringFromAdjacency(adjacencyMatrix)   # (N_pts, N_pts) --> ([N_pts indexes], [cluster 1st index])
        # make sure clustersIndexes don't have duplicates.
        if len(clustersIndexes) is not len(set(clustersIndexes)):
            raise Warning("There should not exist duplicates. Current is {}".format(clustersIndexes))
        clustersIndexes_list.append(clustersIndexes)
        clusters_1stIndex_list.append(clusters_1stIndex)

    return clustersIndexes_list, clusters_1stIndex_list



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
