import os.path as osp
import pickle
import random
from typing import Dict

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation
import torch
import torch.utils.data


from pareconv.utils.pointcloud import (
    random_sample_rotation,
    random_sample_rotation_v2,
    get_transform_from_rotation_translation,
    uniform_2_sphere
)

from pareconv.utils.registration import get_correspondences

def select_random_val_indices(n_samples, n_total, seed=42):
    n_samples = min(n_samples, n_total)
    
    rng = np.random.default_rng(seed)
    selected_indices = rng.choice(n_total, size=n_samples, replace=False)
    
    return selected_indices

def read_dict(out_name, max_lines=None):
    import json
    with open(out_name, 'r') as f:
        data = json.load(f)
    if max_lines != None :
        data = {k: v for k, v in data.items() if int(k) < max_lines}
    return data

def load_from_npz(file):
    with np.load(file, allow_pickle=True) as entry:
        vs = entry['vs_vox']
    return vs

def load_test_from_npz(file, sigma=None, rot=False):
    with np.load(file, allow_pickle=True) as entry:
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


        #if sigma in range(1, 13):
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



class P2PSilicoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_root,
        subset,
        point_limit=None,
        use_augmentation=False,
        augmentation_noise=0.005,
        augmentation_rotation=1,
        overlap_threshold=None,
        return_corr_indices=False,
        matching_radius=None,
        rotated=False,
        vox_size=None,
        min_vis=None,
        max_vis=None,
        overfit=False,
        n_val_ratio=0.2,
        seed=42,
    ):
        super(P2PSilicoDataset, self).__init__()

        self.dataset_root = dataset_root

        self.subset = subset
        self.point_limit = point_limit
        self.overlap_threshold = overlap_threshold
        self.rotated = rotated

        self.return_corr_indices = return_corr_indices
        self.matching_radius = matching_radius
        if self.return_corr_indices and self.matching_radius is None:
            raise ValueError('"matching_radius" is None but "return_corr_indices" is set.')

        self.use_augmentation = use_augmentation
        self.aug_noise = augmentation_noise
        self.aug_rotation = augmentation_rotation

        self.vox_size = vox_size
        self.min_vis = min_vis
        self.max_vis = max_vis
        self.overfit = overfit

        if self.subset != 'test' :
            self.train_val_root = osp.join(self.dataset_root, 'Deform_mesh_npz/')
            self.full_file_list = read_dict(osp.join(self.dataset_root, "Deform_mesh_npz", "dict.json"), max_lines=None)

            file_list_keys = list(self.full_file_list.keys()) 
            # split into train and val
            n_total = len(self.full_file_list)
            self.n_val_samples = int(n_total * n_val_ratio)
            self.seed = seed
            stat_dir =  osp.join(self.dataset_root, "Deform_mesh_npz_test", "stat_svd" )

            ss_val_idx, ss_bins, ss_samples = np.random.SeedSequence(self.seed).spawn(3)

            selected_val_indices = select_random_val_indices(
                n_samples=self.n_val_samples,
                n_total=n_total,
                seed=ss_val_idx
            )
            if self.subset == 'train':
                all_indices = np.arange(len(file_list_keys))
                train_indices = all_indices[~np.isin(all_indices, selected_val_indices)]
                self.file_list = {str(i): self.full_file_list[file_list_keys[idx]] for i, idx in enumerate(train_indices)}
            else:
                self.file_list = {str(i): self.full_file_list[file_list_keys[idx]] for i, idx in enumerate(selected_val_indices)}
                self.visibility_ratios = [(round(0.2 + i*0.1, 1), round(0.3 + i*0.1, 1)) for i in range(8)]
                n = len(self.file_list)
                per_sample_ss = ss_samples.spawn(n)
                self.val_seeds = {str(i): per_sample_ss[i] for i in range(n)}

                rng_bins = np.random.default_rng(ss_bins)
                base_bins = np.tile(np.arange(len(self.visibility_ratios)), int(np.ceil(n / len(self.visibility_ratios))))[:n]
                shuffled_bins = rng_bins.permutation(base_bins)
                self.val_bin_assignment = {str(i): int(shuffled_bins[i]) for i in range(n)}
        else :
            self.test_root = osp.join(self.dataset_root, "Deform_mesh_npz_test", "Test/")
            test_list_root = osp.join(self.dataset_root, "Deform_mesh_npz_test", "list.npz")
            self.test_list = np.load(test_list_root)['test']


    def __len__(self):
        if self.subset == 'test':
            return len(self.test_list)
        else:
            return len(self.file_list)
        
    
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

    def get_scale(self, points):

        center_p = points[:, :3] - np.mean(points[:, :3], axis=0)

        return np.max(np.sqrt(np.sum(center_p ** 2, axis=1)))


    def get_input_train(self, index, vis=False, m =100.0, p=None, rng=None):
        data_dict = {}

        group = self.file_list[str(index)]
        if self.overfit:
            pair_file = [group[0], random.sample(group, 1)[0]]
        elif rng is not None:  # val — deterministic
            pair_indices = rng.choice(len(group), size=2, replace=False)
            pair_file = [group[i] for i in pair_indices]
        else:  # train — keep random
            pair_file = random.sample(group, 2)

        src_file = self.train_val_root + pair_file[0]
        tgt_file = self.train_val_root + pair_file[1]

        src_vs = load_from_npz(src_file)
        tgt_vs_full = load_from_npz(tgt_file)

        src_pcd, m = self.norm_vox(src_vs)
        tgt_pcd_full,_ = self.norm_vox(tgt_vs_full)

        if p is None:
            p = self.min_vis + (self.max_vis-self.min_vis)*np.random.rand(1)[0]
            if p > 1:
                p = 1.0

        tgt_pcd, mask, rand_xyz = self.crop(tgt_pcd_full, p)

        # random noise # Removed for TTA
        #sigma = np.random.rand(1)[0] * self.aug_noise
        #tgt_pcd += (np.random.rand(tgt_pcd.shape[0], 3) - 0.5) * sigma

        m = self.get_scale(src_pcd)

        
        
        if self.return_corr_indices:
            print("return corr indices is true")
            corr_indices = get_correspondences_n(to_o3d_pcd(src_pcd), to_o3d_pcd(tgt_pcd),
                                                self.matching_radius)
            data_dict['corr_indices'] = corr_indices
        

        centroid = np.mean(src_pcd[:, :3], axis=0)
        src_pcd = (src_pcd - centroid) / m
        tgt_pcd = (tgt_pcd - centroid) / m

        trans = -np.mean(tgt_pcd[:, :3], axis=0)
        tgt_pcd += trans

        # get transformation and point cloud
        rot = np.zeros([3, 3])
        rot[0, 0] = 1
        rot[1, 1] = 1
        rot[2, 2] = 1
        trans = trans.T  # np.zeros([3, 1])

        # rotate the point cloud
        euler_ab = np.random.rand(3) * np.pi * 2  # anglez, angley, anglex
        rot_ab = Rotation.from_euler('zyx', euler_ab).as_matrix()
        if (np.random.rand(1)[0] > 0.5):
            src_pcd = np.matmul(rot_ab, src_pcd.T).T
            rot = np.matmul(rot, rot_ab.T)
        else:
            tgt_pcd = np.matmul(rot_ab, tgt_pcd.T).T
            rot = np.matmul(rot_ab, rot)
            trans = np.matmul(rot_ab, trans)

            # src_pcd += (np.random.rand(src_pcd.shape[0], 3) - 0.5) * self.augment_noise # Removed for TTA

        if trans.ndim == 1:
            trans = trans[:, None]

        trans = trans.reshape(-1)
        transform = get_transform_from_rotation_translation(rot, trans)

        ref_points = tgt_pcd
        src_points = src_pcd       


        #metadata
        frag_id1 = pair_file[1]
        frag_id1_no_extention = frag_id1.split('.')[0]
        frag_id1_underscore = frag_id1_no_extention.replace("/", "_")
        frag_id0 = pair_file[0]
        frag_id0_no_extention = frag_id0.split('.')[0]
        frag_id0_underscore = frag_id0_no_extention.replace("/", "_")
        

        data_dict['scene_name'] = "Train"
        data_dict['ref_frame'] = str(frag_id1_underscore)
        data_dict['src_frame'] = str(frag_id0_underscore)
        data_dict['overlap'] = str('Not known')

        data_dict['ref_points'] = ref_points.astype(np.float32)
        data_dict['src_points'] = src_points.astype(np.float32)
        data_dict['ref_feats'] = np.ones((ref_points.shape[0], 1), dtype=np.float32)
        data_dict['src_feats'] = np.ones((src_points.shape[0], 1), dtype=np.float32)
        data_dict['transform'] = transform.astype(np.float32)

        return data_dict
    
    def get_test_sample(self, index, scale=1.0, sigma=None, rot=True):
        src_raw, tgt_raw, src_markers, tgt_markers, R_gt, t_gt = load_test_from_npz(self.test_root + self.test_list[index], sigma, rot)
        return src_raw, tgt_raw, src_markers, tgt_markers, R_gt, t_gt

    
    def get_input_test(self, index, vis=False, sigma=None, rot=False):
        data_dict = {}
        src_raw, tgt_raw, src_markers, tgt_markers, R_gt, t_gt = self.get_test_sample(index, sigma=sigma, rot=rot)

        src_xyz, m = self.norm_vox(src_raw)
        tgt_xyz, _ = self.norm_vox(tgt_raw)

        s_c = np.mean(src_xyz[:, :3], axis=0)
        t_c = np.mean(tgt_xyz[:, :3], axis=0)

        src_xyz = (src_xyz - s_c) / m
        tgt_xyz = (tgt_xyz - t_c) / m

        src_markers = (src_markers - s_c) / m
        tgt_markers = (tgt_markers - t_c) / m

        t_norm = (R_gt @ s_c + t_gt.reshape(-1) - t_c) / m
        transform = get_transform_from_rotation_translation(R_gt, t_norm)
    
        ref_points = tgt_xyz
        src_points = src_xyz

        sample_name = self.test_list[index]
        sample_name_no_extention = sample_name.split('.')[0]
        sample_name_underscore = sample_name_no_extention.replace("/", "_")
        data_dict['scene_name'] = "Test"
        data_dict['ref_frame'] = str(sample_name_underscore)
        data_dict['src_frame'] = str(sample_name_underscore)
        data_dict['overlap'] = str('Not known')

        # get correspondences
        if self.return_corr_indices:
            corr_indices = get_correspondences(ref_points, src_points, transform, self.matching_radius)
            correspondences = get_correspondences_n(to_o3d_pcd(src_deform), to_o3d_pcd(tgt_raw),
                                                self.matching_radius)
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
        if self.subset =="train":
            return self.get_input_train(index, vis=vis)
        elif self.subset =='val':
            bin_idx = self.val_bin_assignment[str(index)]
            p_range = self.visibility_ratios[bin_idx]
            rng = np.random.default_rng(seed=self.val_seeds[str(index)])
            p = rng.uniform(p_range[0], p_range[1])
            return self.get_input_train(index, vis=vis, p=p, rng=rng)
        else: 
            return self.get_input_test(index, vis=vis, sigma=sigma, rot=rot)


class CorruptedP2PSilicoDataset(torch.utils.data.Dataset):
    def __init__(self, args, cor_dataset_root):
        super(CorruptedP2PSilicoDataset, self).__init__()
        if args.corruption == 'clean':
            self.data_path = osp.join(cor_dataset_root, args.corruption + '.npy')
        else : 
            self.data_path = osp.join(cor_dataset_root, 'data_' + args.corruption + str(f'_{int(args.severity)}') + '.npy')
        self.data_dicts = np.load(self.data_path, allow_pickle=True)

    def __getitem__(self,item):
        return self.data_dicts[item]
    
    def __len__(self):
        return len(self.data_dicts)