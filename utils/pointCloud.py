import numpy as np
from scipy.spatial import cKDTree
from plyfile import PlyData, PlyElement



def convert_mesh_2_TSDF():
    """
    """
    pass

def convert_pointCloud_2_probMap():
    pass

def generate_on_off_surface_pts():
    """ 
    Generate 3D pts candidates for similarityNet
    """
    pass


# read ply
# reconstruct KDTree
    # this octree can also be used to detect occlusion (coarse2fine, view occlusion detection in the upper level).
    # generate similarityNet training 3D pts
    # OctNet is used to find the distance to the nearest point of the model.
# generate 3D surface gt (prob,normal_xyz), with shift, rotation augmentation.
# generate off surface f
# if mesh data, generate TSDF. TODO: how to generate the TSDF
