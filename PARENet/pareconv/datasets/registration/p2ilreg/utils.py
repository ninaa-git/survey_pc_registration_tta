import numpy as np
import open3d as o3d
import torch
from scipy.spatial import cKDTree

def _enough_overlap(src, tgt, R, t, radius=0.0065, min_corr=1000):
    """Count src–tgt pairs within `radius` in the PHYSICAL, post-aug frame.
    count_neighbors returns the same total #pairs KDTree_corr would, but in C."""
    src_t = src @ R.T + t.reshape(1, 3)          # physical src -> tgt frame
    n_pairs = cKDTree(tgt).count_neighbors(cKDTree(src_t), radius)
    return n_pairs >= min_corr

def to_array(tensor):
    """
    Conver tensor to array
    """
    if(not isinstance(tensor,np.ndarray)):
        if(tensor.device == torch.device('cpu')):
            return tensor.numpy()
        else:
            return tensor.cpu().numpy()
    else:
        return tensor

def to_o3d_pcd(xyz):
    """
    Convert tensor/array to open3d PointCloud
    xyz:       [N, 3]
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(to_array(xyz))
    return pcd

def to_tsfm(rot,trans):
    tsfm = np.eye(4)
    tsfm[:3,:3]=rot
    tsfm[:3,3]=trans.flatten()
    return tsfm

def KDTree_corr ( src_pcd_transformed, tgt_pcd, search_voxel_size, K=None):

    pcd_tree = o3d.geometry.KDTreeFlann(tgt_pcd)
    correspondences = []
    for i, point in enumerate(src_pcd_transformed.points):
        [count, idx, _] = pcd_tree.search_radius_vector_3d(point, search_voxel_size)
        if K is not None:
            idx = idx[:K]
        for j in idx:
            correspondences.append([i, j])
    correspondences = np.array(correspondences)
    return correspondences

def get_correspondences(src_pcd, tgt_pcd, trans, search_voxel_size, K=None):

    src_pcd.transform(trans)
    # distance_src = src_pcd.compute_nearest_neighbor_distance()
    # distance_tgt = tgt_pcd.compute_nearest_neighbor_distance()
    # print("avg_dist of src and tgt: ", np.mean(distance_src), np.mean(distance_tgt))
    correspondences =  KDTree_corr ( src_pcd, tgt_pcd, search_voxel_size, K=None)
    # correspondences = torch.from_numpy(correspondences)
    return correspondences

