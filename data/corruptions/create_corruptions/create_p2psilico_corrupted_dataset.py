import argparse
import os
import os.path as osp
import pickle
import random
from typing import Dict

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation
#from pygem import FFD
#from pygem import RBF
import torch
import torch.utils.data

from multiprocessing import Pool
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="silico")
    parser.add_argument('--dataset_root', type=str, default="../../P2P/Dataset/")
    parser.add_argument('--corrupted_dataset_path', type=str, default='../datasets')

    return parser.parse_args()


def read_dict(out_name):
    import json
    f = open(out_name)
    data = json.load(f)
    f.close()
    return data

def load_from_npz(file):
    with np.load(file, allow_pickle=True) as entry:
        vs = entry['vs_vox']
    return vs

def random_dropout_exact(xyz: np.ndarray, n_keep: int, seed: int) -> np.ndarray:
        N = xyz.shape[0]
        n_keep = max(1, min(int(n_keep), N))
        rng = np.random.default_rng(seed)
        idx = rng.choice(N, size=n_keep, replace=False)
        return xyz[idx]

def get_transform_from_rotation_translation(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    r"""Get rigid transform matrix from rotation matrix and translation vector.

    Args:
        rotation (array): (3, 3)
        translation (array): (3,)

    Returns:
        transform: (4, 4)
    """
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


class ThreeDMatchPairDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_root,
        vox_size=None,
        cor=None,
        sev=None,
        map=None
    ):
        super(ThreeDMatchPairDataset, self).__init__()

        self.dataset_root = dataset_root
        self.vox_size = vox_size
        self.cor = cor 
        self.sev = sev

        self.test_root = osp.join(self.dataset_root, "Deform_mesh_npz_test", "Test/")
        test_list_root = osp.join(self.dataset_root, "Deform_mesh_npz_test", "list.npz")
        self.test_list = np.load(test_list_root)['test']

        
        
        """ self.MAP = {
            'shear': self.shear,
            'distortion': self.ffd_distortion,
            'distortion_rbf': self.rbf_distortion,
            'distortion_rbf_inv': self.rbf_distortion_inv,
            
        } """

        
        """ self.MAP = {
            'uniform': self.uniform_noise,
            'gaussian': self.gaussian_noise,
            'background_noise': self.background_noise, 
            'impulse_noise': self.impulse_noise,
            'global_density_dec': self.global_density_dec,
            'local_density_dec': self.local_density_dec,
            'cutout' : self.cutout,
            'occlusion': self.occlusion,
            'scale': self.aug_scale,
            }
        """
        self.MAP = {
            'local_density_dec': self.local_density_dec,
            'cutout' : self.cutout,
            'occlusion': self.occlusion,
            }
            
        

        self.matching_radius = 0.04
        self.return_corr_indices = False

    def __len__(self):
        return len(self.test_list)
        
    def load_test_from_npz(self, file, sigma=None, rot=False):
        with np.load(file, allow_pickle=True) as entry:
            cor = self.cor
            sev = self.sev
            MAP = self.MAP

            
                
            src_vs = entry['src_pcd']
            tgt_vs = entry['tgt_pcd']
            # flow = entry['flow']

            src_markers = entry['src_vol']
            tgt_markers = entry['tgt_vol']

            R_gt = entry['R_gt']
            t_gt = entry['t_gt']

            # faces = entry['src_f']
            # edges = entry['src_edges']
            # tgt_f = entry['tgt_f']

            
            if sigma in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"]:
                noise = entry[str(sigma)]
                tgt_vs = tgt_vs + noise

            if rot:
               # rot_src = entry['rot_src']
                rot_tgt = entry['rot_tgt']
                #src_vs = (np.matmul(rot_src, src_vs.T)).T
                tgt_vs = (np.matmul(rot_tgt, tgt_vs.T)).T

                #src_markers = (np.matmul(rot_src, src_markers.T)).T
                tgt_markers = (np.matmul(rot_tgt, tgt_markers.T)).T

                R_gt = np.matmul(R_gt, rot_tgt)
                t_gt = np.matmul(rot_tgt, t_gt)

        return src_vs, tgt_vs, src_markers, tgt_markers, R_gt, t_gt

    
    def pc_normalize(self, pc, centroid=None, m=None):
        if centroid is None:
            centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        if m is None:
            m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        return pc, centroid, m

    def ply2np_vox(self, xyz, voxel_size=2, scale=1.0):
        if type(xyz) is str:
            pcd = o3d.io.read_point_cloud(xyz)
        else:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)
        downpcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        pcd_pts = np.asarray(downpcd.points)
        pcd_pts = pcd_pts / scale
        return pcd_pts

    def norm_vox(self, xyz):
        pc, centroid, m = self.pc_normalize(xyz)
        pc_vox = self.ply2np_vox(pc, voxel_size=self.vox_size, scale=1.0) * m + centroid
        return pc_vox, m

    def norm_vox_scaled(self, xyz, centroid, m):
        pc, _, _ = self.pc_normalize(xyz, centroid=centroid, m=m)
        pc_vox = self.ply2np_vox(pc, voxel_size=self.vox_size, scale=1.0) * m + centroid
        return pc_vox, m

    def norm_tgt_vox_scaled(self, xyz, v, centroid, m):
        pc, _, _ = self.pc_normalize(xyz, centroid=centroid, m=m)
        pc_vox = self.ply2np_vox(pc, voxel_size=v, scale=1.0) * m + centroid
        return pc_vox, m

    def crop(self, points, p_keep, rand_xyz=None):
        if rand_xyz is None:
            rand_xyz = self.uniform_2_sphere()
        centroid = np.mean(points[:, :3], axis=0)
        points_centered = points[:, :3] - centroid

        dist_from_plane = np.dot(points_centered, rand_xyz)
        if p_keep == 0.5:
            mask = dist_from_plane > 0
        else:
            mask = dist_from_plane > np.percentile(dist_from_plane, (1.0 - p_keep) * 100)

        return points[mask, :], mask, rand_xyz

    def uniform_2_sphere(self, num: int = None):
        """Uniform sampling on a 2-sphere
        Source: https://gist.github.com/andrewbolster/10274979
        Args:
            num: Number of vectors to sample (or None if single)
        Returns:
            Random Vector (np.ndarray) of size (num, 3) with norm 1.
            If num is None returned value will have size (3,)

        """
        if num is not None:
            phi = np.random.uniform(0.0, 2 * np.pi, num)
            cos_theta = np.random.uniform(-1.0, 1.0, num)
        else:
            phi = np.random.uniform(0.0, 2 * np.pi)
            cos_theta = np.random.uniform(-1.0, 1.0)

        theta = np.arccos(cos_theta)
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)

        return np.stack((x, y, z), axis=-1)

    def rand_rot(self, pcd, euler_ab=None):
        if euler_ab is None:
            euler_ab = np.random.rand(3) * np.pi * 2
        rot = Rotation.from_euler('zyx', euler_ab).as_matrix()
        pcd = (np.matmul(rot, pcd.T)).T
        return pcd, euler_ab

    def center(self, points, centroid=None):
        if centroid is None:
            centroid = np.mean(points[:, :3], axis=0)
        points_centered = points[:, :3] - centroid
        return points_centered, centroid
    
    def downsample_grid(self, tgt_raw, sev):
        #quelle 
        vox_size = self.vox_size
        tgt_vox_size = vox_size + sev*0.01
        return self.norm_tgt_vox(tgt_raw, tgt_vox_size)
    
    def core_distortion(self,points, n_control_points=[2, 2, 2], displacement=None):
        """
            Ref: http://mathlab.github.io/PyGeM/tutorial-1-ffd.html
        """
        # the size of displacement matrix: 3 * control_points.shape
        if displacement is None:
            displacement = np.zeros((3, *n_control_points))

        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        
        # Add 20% margin to ensure all points are inside the FFD box
        margin = (max_coords - min_coords) * 0.2
        box_origin = min_coords - margin
        box_length = max_coords - min_coords + 2 * margin
        
        # Initialize displacement if not provided
        if displacement is None:
            displacement = np.zeros((3, *n_control_points))

        ffd = FFD(n_control_points=n_control_points)
        m = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
        #ffd.box_length = [m, m, m]
        #ffd.box_origin = [-m/2., -m/2., -m/2]
        ffd.box_length = box_length.tolist()  # CHANGED: use actual data extent
        ffd.box_origin = box_origin.tolist() 
        ffd.array_mu_x = displacement[0, :, :, :]
        ffd.array_mu_y = displacement[1, :, :, :]
        ffd.array_mu_z = displacement[2, :, :, :]
        new_points = ffd(points)

        return new_points


    def distortion(self, points, direction_mask=np.array([1, 1, 1]), point_mask=np.ones((5, 5, 5)), severity=0.5):
        n_control_points = [5, 5, 5]
        # random
        displacement = np.random.rand(3, *n_control_points) * 2 * severity - np.ones((3, *n_control_points)) * severity
        displacement *= np.transpose(np.tile(direction_mask, (5, 5, 5, 1)), (3, 0, 1, 2))
        displacement *= np.tile(point_mask, (3, 1, 1, 1))

        points = self.core_distortion(points, n_control_points=n_control_points, displacement=displacement)

        return points


    def distortion_2(self, points, severity=(0.4, 3), func='gaussian_spline'):
        rbf = RBF(func=func)
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        xv = np.linspace(min_coords[0], max_coords[0], severity[1])
        yv = np.linspace(min_coords[1], max_coords[1], severity[1])
        zv = np.linspace(min_coords[2], max_coords[2], severity[1])
        z, y, x = np.meshgrid(zv, yv, xv)
        mesh = np.array([x.ravel(), y.ravel(), z.ravel()]).T
        rbf.original_control_points = mesh
        alpha = np.random.uniform(-np.pi, np.pi, mesh.shape[0])
        gamma = np.random.uniform(-np.pi, np.pi, mesh.shape[0])
        distance = np.ones(mesh.shape[0]) * severity[0]
        displacement_x = distance * np.cos(alpha) * np.sin(gamma)
        displacement_y = distance * np.sin(alpha) * np.sin(gamma)
        displacement_z = distance * np.cos(gamma)
        displacement = np.array([displacement_x, displacement_y, displacement_z]).T
        rbf.deformed_control_points = mesh + displacement
        new_points = rbf(points)
        return new_points
    
    def core_occlusion(mesh, type, camera_extrinsic=None, camera_intrinsic=None, window_width=1080, window_height=720,
                   n_points=None, downsample_ratio=None):
        if camera_extrinsic is None:
            camera_extrinsic = get_default_camera_extrinsic()

        if camera_intrinsic is None:
            camera_intrinsic = get_default_camera_intrinsic()

        camera_parameters = o3d.camera.PinholeCameraParameters()
        camera_parameters.extrinsic = camera_extrinsic
        camera_parameters.intrinsic.set_intrinsics(**camera_intrinsic)

        viewer = o3d.visualization.Visualizer()
        viewer.create_window(width=window_width, height=window_height)
        viewer.add_geometry(mesh)

        control = viewer.get_view_control()
        control.convert_from_pinhole_camera_parameters(camera_parameters)
        # viewer.run()

        depth = viewer.capture_depth_float_buffer(do_render=True)

        viewer.destroy_window()
        pcd = o3d.geometry.PointCloud.create_from_depth_image(depth, camera_parameters.intrinsic,
                                                            extrinsic=camera_parameters.extrinsic)

        if downsample_ratio is not None:
            ratio = int((1 - downsample_ratio) / downsample_ratio)
            pcd = pcd.uniform_down_sample(ratio)
        elif n_points is not None:
            # print(np.asarray(pcd.points).shape[0])
            ratio = int(np.asarray(pcd.points).shape[0] / n_points)
            if ratio > 0:
                # if type == 'occlusion':
                set_points(pcd, shuffle_data(np.asarray(pcd.points)))
                pcd = pcd.uniform_down_sample(ratio)

        return pcd

    
    def uniform_noise(self,src_raw, tgt_raw, clean_c, clean_m, sev) : 
        sigma = [2, 4, 6, 8, 10][sev - 1]
        N, C = tgt_raw.shape
        jitter = np.random.uniform(-sigma/2, sigma/2, (N, C)) 
        new_pc = tgt_raw + jitter
    
        new_pc, _ = self.norm_vox_scaled(new_pc, clean_c, clean_m)

        return new_pc, None

    def gaussian_noise(self,src_raw, tgt_raw, clean_c, clean_m, sev) : 
        
        
        # quelle corruption en fonction de la sévérité
        sigma = [2, 3, 4, 5, 6][sev - 1]
        N, C = tgt_raw.shape
        jitter = np.random.normal(size=(N, C)) * (sigma/2)
        jitter = np.clip(jitter, -sigma, sigma)
        new_pc = tgt_raw + jitter

        new_pc, _ = self.norm_vox_scaled(new_pc, clean_c, clean_m)

        return new_pc, None

    def background_noise(self,src_raw, tgt_raw, clean_c, clean_m, sev) : 
        N, C = tgt_raw.shape
        c = [N // 100, N // 50, N // 30, N // 15, N // 5][sev - 1]

        jitter = np.random.uniform(-clean_m + clean_c, clean_m + clean_c, (c, C))
        new_pc = np.concatenate((tgt_raw, jitter), axis=0).astype('float32')
        new_pc, _ = self.norm_vox_scaled(new_pc, clean_c, clean_m)

        return new_pc, None
    
    def impulse_noise(self,src_raw, tgt_raw, clean_c, clean_m, sev) : 
        N, C = tgt_raw.shape
        c = [N // 45,  N // 30, N // 20, N // 10, N // 4][sev - 1]
        index = np.random.choice(N, c, replace=False)
        sigma = 10
        tgt_raw[index] += np.random.choice([-sigma, sigma], size=(c, C)) 
        new_pc, _ = self.norm_vox_scaled(tgt_raw, clean_c, clean_m)
        return new_pc, None

    """ def upsampling(self,src_raw, tgt_raw, clean_c, clean_m, sev):
        N, C = tgt_raw.shape
        c = [N // 5, N // 3, N, 2 * N, 4 * N][sev - 1]
        if c > N:
            index = np.random.choice(N, c, replace=True)
        else:
            index = np.random.choice(N, c, replace=False)

        sigma = 8
        add = tgt_raw[index] + np.random.uniform(-sigma, sigma, (c, C))
        new_pc = np.concatenate((tgt_raw, add), axis=0).astype('float32')
        new_pc, _ = self.norm_vox(tgt_raw)
        return new_pc """

    def global_density_dec(self,src_raw, tgt_raw, clean_c, clean_m, sev):
        v = self.vox_size
        n_v = [1.5*v, 2*v, 2.5*v, 3*v, 3.5*v][sev - 1]
        new_pc, _ = self.norm_tgt_vox_scaled(tgt_raw, n_v, clean_c, clean_m)
        return new_pc, None

    def local_density_dec(self,src_raw, tgt_raw, clean_c, clean_m, sev):
        norm_vox_pc, _ = self.norm_vox_scaled(tgt_raw, clean_c, clean_m)
        N, C = norm_vox_pc.shape
        n = N //10
        c = [(1, n),  (2, n), (4, n), (6, n), (8, n)][sev - 1]
        pc = norm_vox_pc.copy()
        for s in range(c[0]):
            np.random.seed(s)
            i = np.random.choice(pc.shape[0], 1)
            picked = pc[i]
            dist = np.sum((pc - picked) ** 2, axis=1, keepdims=True)
            idx = np.argpartition(dist, c[1], axis=0)[:c[1]]
            idx_2 = np.random.choice(c[1], int((3 / 4) * c[1]), replace=False)
            idx = idx[idx_2]
            pc = np.delete(pc, idx.squeeze(), axis=0)
        
        return pc, None

    def cutout(self,src_raw, tgt_raw, clean_c, clean_m, sev):
        norm_vox_pc, _ = self.norm_vox_scaled(tgt_raw, clean_c, clean_m)
        N, C = norm_vox_pc.shape
        n = N // 30 #sur 30
        c = [(2, n), (3, n), (5, n), (7, n), (10, n)][sev - 1]

        pc = norm_vox_pc.copy()
        for s in range(c[0]):
            np.random.seed(s)
            i = np.random.choice(pc.shape[0], 1)
            picked = pc[i]
            dist = np.sum((pc - picked) ** 2, axis=1, keepdims=True)
            idx = np.argpartition(dist, c[1], axis=0)[:c[1]]
            pc = np.delete(pc, idx.squeeze(), axis=0)
            
        return pc, None

    def occlusion(self, src_raw, tgt_raw, clean_c, clean_m, sev):
        norm_vox_pc, _ = self.norm_vox_scaled(tgt_raw, clean_c, clean_m)
        
        instrument_diameters_mm = [6, 12, 18, 24, 30]
        radius = instrument_diameters_mm[sev - 1] / 2.0

        pc = norm_vox_pc.copy() 

        centroid   = pc.mean(axis=0)
        pc_centered = pc - centroid                          
        _, _, Vt   = np.linalg.svd(pc_centered, full_matrices=False)
        u = Vt[0]
        v = Vt[1]

        u_coords = pc_centered @ u   # (N,)
        v_coords = pc_centered @ v   # (N,)

        np.random.seed(0)
        angle        = np.random.uniform(0, np.pi/2)
        shaft_normal = np.array([-np.sin(angle), np.cos(angle)])
        
        cloud_extent = max(u_coords.ptp(), v_coords.ptp())
        offset       = np.random.uniform(-0.1, 0.1) * cloud_extent
        center_2d    = offset * shaft_normal

        coords_2d     = np.stack([u_coords, v_coords], axis=1) - center_2d
        dist_to_shaft = np.abs(coords_2d @ shaft_normal)

        in_shadow = dist_to_shaft < radius
        pc_occluded = pc[~in_shadow]   

        if pc_occluded.shape[0] < 10:
            pc_occluded = pc

        
        return pc_occluded, None

    ## SCALE
    def compute_pca_scale(self, points):
        """Compute scale along principal axis using PCA."""
        # Center points
        centroid = np.mean(points, axis=0)  # (1, 3)
        centered = points - centroid  # (N, 3)
        
        # Covariance matrix: C = (1/N) * X^T * X
        covariance = (centered.T @ centered) / points.shape[0]  # (3, 3)
        
        # Get principal axis (eigenvector with max eigenvalue)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        principal_axis = eigenvectors[:, -1]  # Last column = max eigenvalue
        
        # Project onto principal axis and get range
        projections = centered @ principal_axis
        scale = projections.max() - projections.min()
        
        return scale, centroid
    
    def compute_scale_factor(self, src_raw, tgt_raw):
        """
        Compute PCA-based scale factor: σ = σ_src / σ_tgt
        
        Args:
            src_raw: Source point cloud (N_src, 3)
            tgt_raw: Target point cloud (N_tgt, 3)
        
        Returns:
            scale_factor: σ to scale tgt to match src
        """
        src_scale, _ = self.compute_pca_scale(src_raw)
        tgt_scale, _ = self.compute_pca_scale(tgt_raw)
        
        return src_scale / tgt_scale

    def scale_target_to_source(self, src_raw, tgt_raw, scale_factor):
        """
        Scale tgt to match src's spatial extent using PCA.
        
        Args:
            src_raw: Source point cloud (N_src, 3)
            tgt_raw: Target point cloud (N_tgt, 3)
        
        Returns:
            tgt_scaled: Scaled target point cloud (N_tgt, 3)
            scale_factor: The scale factor σ applied
        """
        tgt_scale, tgt_centroid = self.compute_pca_scale(tgt_raw)
        
        # Center, scale, recenter
        tgt_scaled = (tgt_raw - tgt_centroid) * scale_factor + tgt_centroid
        
        return tgt_scaled, scale_factor

    def aug_scale(self,src_raw, tgt_raw, clean_c, clean_m, sev)  : 

        scale = self.compute_scale_factor(src_raw, tgt_raw)
        scale_factor = [scale, scale*1.2, scale*1.6, scale*1.8, scale*2.0][sev -  1]
        tgt_scaled, sigma = self.scale_target_to_source(src_raw, tgt_raw, scale_factor)
        new_pc, _ = self.norm_vox_scaled(tgt_scaled, clean_c, clean_m)
        list_scale = scale_factor
        
        return new_pc, list_scale
    
    def shear(self,src_raw, tgt_raw, clean_c, clean_m, sev) :
        N, C = tgt_raw.shape
        c = [0.01, 0.015, 0.03, 0.06, 0.12][sev - 1]
        a = np.random.uniform(c - 0.05, c + 0.05) * np.random.choice([-1, 1])
        b = np.random.uniform(c - 0.05, c + 0.05) * np.random.choice([-1, 1])
        d = np.random.uniform(c - 0.05, c + 0.05) * np.random.choice([-1, 1])
        e = np.random.uniform(c - 0.05, c + 0.05) * np.random.choice([-1, 1])
        f = np.random.uniform(c - 0.05, c + 0.05) * np.random.choice([-1, 1])
        g = np.random.uniform(c - 0.05, c + 0.05) * np.random.choice([-1, 1])

        matrix = np.array([[1, 0, b], [d, 1, e], [f, 0, 1]])
        new_pc = np.matmul(tgt_raw, matrix).astype('float32')
        new_pc, _ = self.norm_vox_scaled(new_pc, clean_c, clean_m)
        return new_pc, matrix
    
    def ffd_distortion(self,src_raw, tgt_raw, clean_c, clean_m, sev) :
        N, C = tgt_raw.shape
        c = [0.1, 0.2, 0.3, 0.4, 0.5][sev - 1]
        new_pc = self.distortion(tgt_raw, severity=c)
        new_pc, _ = self.norm_vox_scaled(new_pc, clean_c, clean_m)
        return new_pc


    def rbf_distortion(self,src_raw, tgt_raw, clean_c, clean_m, sev):
        N, C = tgt_raw.shape
        c = [(0.25, 5), (0.5, 5), (1, 5), (3, 5), (5, 5)][sev - 1]
        new_pc = self.distortion_2(tgt_raw, severity=c, func='multi_quadratic_biharmonic_spline')
        new_pc, _ = self.norm_vox_scaled(new_pc.astype('float32'), clean_c, clean_m)
        return new_pc

    def rbf_distortion_inv(self,src_raw, tgt_raw, clean_c, clean_m, sev) :
        N, C = tgt_raw.shape
        c = [(0.25, 5), (0.5, 5), (1, 5), (3, 5), (15, 5)][sev - 1]
        new_pc = self.distortion_2(tgt_raw, severity=c, func='inv_multi_quadratic_biharmonic_spline')
        new_pc, _ = self.norm_vox_scaled(new_pc.astype('float32'), clean_c, clean_m)
        return new_pc


    def get_scale(self, points):

        center_p = points[:, :3] - np.mean(points[:, :3], axis=0)

        return np.max(np.sqrt(np.sum(center_p ** 2, axis=1)))
    
    def get_test_sample(self, index, scale=1.0, sigma=None, rot=True):
           sample_name = self.test_list[index]
           src_raw, tgt_raw, src_markers, tgt_markers, R_gt, t_gt = self.load_test_from_npz(self.test_root + self.test_list[index], sigma, rot)
           return src_raw, tgt_raw, src_markers, tgt_markers, R_gt, t_gt
    
    def get_input_test(self, index, vis=False, sigma=None, rot=True):
        data_dict = {}

        cor = self.cor
        sev = self.sev
        MAP = self.MAP
        
        src_raw, tgt_raw, src_markers, tgt_markers, R_gt, t_gt = self.get_test_sample(index, rot=rot)

        src_xyz, m = self.norm_vox(src_raw)

        if cor is not None :
            tgt_raw_cp = tgt_raw.copy()
            clean_c = np.mean(tgt_raw, axis=0)
            clean_centered = tgt_raw - clean_c
            clean_m = np.max(np.sqrt(np.sum(clean_centered ** 2, axis=1)))
            tgt_xyz, cor_transform = self.MAP[cor](src_raw,tgt_raw_cp, clean_c, clean_m, sev)
        if cor is None : 
            tgt_xyz, _ = self.norm_vox(tgt_raw)

        clean_xyz, _ = self.norm_vox(tgt_raw)

        out_dir = osp.join("./viz", 'raw')
        if not osp.exists(out_dir) : 
            os.makedirs(out_dir)
        out_path = osp.join(out_dir, 'raw.png')
        gt_points = (np.matmul(R_gt, src_xyz.T) + t_gt).T
        visualize(clean_xyz, out_path, gt_points) 

        s_c = np.mean(src_xyz[:, :3], axis=0)
        t_c = np.mean(clean_xyz[:, :3], axis=0)

        src_xyz = (src_xyz - s_c) / m
        tgt_xyz = (tgt_xyz - t_c) / m

        src_markers = (src_markers - s_c) / m
        tgt_markers = (tgt_markers - t_c) / m

        ref_points = tgt_xyz
        src_points = src_xyz

        sample_name = self.test_list[index]
        sample_name_no_extention = sample_name.split('.')[0]
        sample_name_underscore = sample_name_no_extention.replace("/", "_")
        data_dict['scene_name'] = "Test"
        data_dict['ref_frame'] = str(sample_name_underscore)
        data_dict['src_frame'] = str(sample_name_underscore)
        data_dict['overlap'] = str('Not known')

        if cor == 'scale':
            s = cor_transform
            data_dict['scale'] = s

        t_norm = (R_gt @ s_c + t_gt.reshape(-1) - t_c) / m
        transform = get_transform_from_rotation_translation(R_gt, t_norm)
        
        if self.return_corr_indices:
            corr_indices = get_correspondences(ref_points, src_points, transform, self.matching_radius)
            data_dict['corr_indices'] = corr_indices

        data_dict['ref_points'] = ref_points.astype(np.float32)
        data_dict['src_points'] = src_points.astype(np.float32)
        data_dict['ref_feats'] = np.ones((ref_points.shape[0], 1), dtype=np.float32)
        data_dict['src_feats'] = np.ones((src_points.shape[0], 1), dtype=np.float32)
        data_dict['transform'] = transform.astype(np.float32)

        data_dict['src_markers'] = src_markers
        data_dict['tgt_markers'] = tgt_markers
        data_dict['s_c'] = s_c
        data_dict['t_c'] = t_c
        data_dict['m'] = m

        return data_dict

    def __getitem__(self, index, vis=False, sigma=None, rot=True):
        return self.get_input_test(index, vis=vis, sigma=sigma, rot=rot)
    

def visualize(tgt_pcd,
              out_path,
              src_pcd=None,
              figsize=(8, 8),
                dpi=200,) : 
    import matplotlib.pyplot as plt
    def to_np(x):
        if x is None:
            return None
        try:
            import torch
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
        except Exception:
            pass
        return np.asarray(x)

    t_pc   = to_np(tgt_pcd)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax  = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")

    z = tgt_pcd[:, 2]
    z_min = tgt_pcd[:, 2].min()
    shadow_z = z_min - 1 * (tgt_pcd[:, 2].max() - z_min)
    z_norm = (z - z.min()) / (z.max() - z.min())

    if tgt_pcd is not None and len(tgt_pcd) > 0:
        ax.scatter(tgt_pcd[:, 0], tgt_pcd[:, 1], tgt_pcd[:, 2],
                    c='red', #[[1.0, 92/255, 92/255]],  
                    s=10,
                    alpha=1,
                    edgecolors="none") 
    if src_pcd is not None and len(src_pcd) > 0:
        ax.scatter(src_pcd[:, 0], src_pcd[:, 1], src_pcd[:, 2],
                   c=[[0, 150/255, 1.0]], s=10, alpha=0.4, edgecolors="none")

    ax.set_axis_off()

    all_pts = tgt_pcd if src_pcd is None else np.vstack([tgt_pcd, src_pcd])
    ax.set_xlim(all_pts[:,0].min(), all_pts[:,0].max())
    ax.set_ylim(all_pts[:,1].min(), all_pts[:,1].max())
    ax.set_zlim(all_pts[:,2].min(), all_pts[:,2].max())
        
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def process_corruption(args):
    cor, sev, index, dataset_root, voxel_size = args
    t = ThreeDMatchPairDataset(dataset_root, vox_size=voxel_size, cor=cor, sev=sev)
    return t.__getitem__(index)['ref_points']    
        
def main() :
    args = get_args()
    voxel_size = 0.04

    t = ThreeDMatchPairDataset(
                args.dataset_root, 
                vox_size=voxel_size,
            )
    
    clean_path = osp.join(args.corrupted_dataset_path, 'clean.npy')
    if not osp.isfile(clean_path): 

        print(f"\nProcessing clean data")

        data_dicts_clean = []
        for index in tqdm(range(len(t))) : 
            data_dict = t.__getitem__(index, vis=True, sigma=None, rot=True)
            data_dicts_clean.append(data_dict)
        
        np.save(clean_path, data_dicts_clean)

    for cor in t.MAP.keys():
        print(f"\nProcessing corruption: {cor}")
        for sev in [1,2,3,4,5]:
            corruption_path = osp.join(args.corrupted_dataset_path, f'data_{cor}_{str(sev)}.npy')
            #if not osp.isfile(corruption_path): #not

            print(f"  Severity {sev}...")
            data_dicts = []
            t = ThreeDMatchPairDataset(
                args.dataset_root, 
                vox_size=voxel_size,
                cor=cor,
                sev=sev
            )
            
            for index in tqdm(range(len(t))): #for index in [10]: #10 for slides
                data_dict = t.__getitem__(index, vis=True, sigma=None, rot=True)
                data_dicts.append(data_dict)

            viz = False
            if viz : 
                out_dir = osp.join("./viz", cor)
                if not osp.exists(out_dir) : 
                    os.makedirs(out_dir)
                #visualize(data_dict['ref_points'], out_path, data_dict['src_points']) #, data_dict['src_points']

                transform = data_dict['transform']
                rotation = transform[:3, :3]  # (B, 3, 3)
                translation = transform[:3, 3]  # (B, 1, 3)
                gt_points = np.matmul(data_dict['src_points'], rotation.transpose(-1, -2)) + translation

                out_path1 = osp.join(out_dir, f'{sev}.png')
                visualize(data_dict['ref_points'], out_path1) #gt_points , data_dict['src_points']
                out_path2 = osp.join(out_dir, f'{sev}_gt.png')
                visualize(data_dict['ref_points'], out_path2, gt_points)

                if cor == 'scale':
                    scale_factor = data_dict['scale']

                    # forward
                    gt_points_c = np.mean(data_dict['src_points'], axis = 0)
                    gt_points_scaled = (gt_points - gt_points_c)*scale_factor + gt_points_c 
                    gt_points = np.matmul(gt_points_scaled, rotation.transpose(-1, -2)) + translation * scale_factor
                    
                    out_path3  = osp.join(out_dir, f'{sev}_scale_fwd.png')
                    visualize(data_dict['ref_points'], out_path3, gt_points_scaled)

                    # inverse
                    ref_inv = data_dict['ref_points']
                    ref_inv_c = np.mean(ref_inv, axis=0)
                    ref_inv_scaled = (ref_inv - ref_inv_c)*(1/scale_factor) + ref_inv_c
                    ref_inv = (rotation.T @ (ref_inv_scaled - translation).T).T
                    
                    out_path4 = osp.join(out_dir, f'{sev}_scale_inv.png')
                    visualize(ref_inv, out_path4, data_dict['src_points'])

            np.save(corruption_path, data_dicts)
    
    
if __name__ == "__main__":
    main()

    
        
