import torch.utils.data as data
from PIL import Image
import os
#import cv2
#os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import os.path
import torch
import numpy as np
#import torchvision.transforms as transforms
import argparse
import time
import random
import _pickle as cPickle
import numpy.ma as ma
import copy
import scipy.misc
import scipy.io as scio
import open3d as o3d
from plyfile import PlyData
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import yaml
import re
import logging
from pareconv.datasets.registration.p2ilreg.utils import get_correspondences, to_o3d_pcd, to_tsfm, _enough_overlap
logging.getLogger('PIL').setLevel(logging.WARNING)

# import imgaug.augmenters as iaa
current_dir = os.path.dirname(os.path.abspath(__file__))

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

def _ensure_npy(ply_path):
    npy_path = ply_path + ".npy"
    if not os.path.exists(npy_path):
        v = PlyData.read(ply_path)['vertex']
        pts = np.stack([v['x'], v['y'], v['z']], axis=1).astype(np.float32)
        tmp = f"{npy_path}.{os.getpid()}.tmp"
        try:
            with open(tmp, "wb") as fh:
                np.save(fh, pts)
            os.replace(tmp, npy_path)    # atomic on same filesystem
        except Exception:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise
    return npy_path

def get_bbox(bbox):
    """ Compute square image crop window. """
    y1, x1, y2, x2 = bbox

    img_width = 1024
    img_length = 1280

    window_size = (max(y2 - y1, x2 - x1) // 40 + 1) * 40
    window_size = min(window_size, 640)
    center = [(y1 + y2) // 2, (x1 + x2) // 2]
    rmin = center[0] - int(window_size / 2)
    rmax = center[0] + int(window_size / 2)
    cmin = center[1] - int(window_size / 2)
    cmax = center[1] + int(window_size / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > img_width:
        delt = rmax - img_width
        rmax = img_width
        rmin -= delt
    if cmax > img_length:
        delt = cmax - img_length
        cmax = img_length
        cmin -= delt
    return rmin, rmax, cmin, cmax


def voxel_downsample_np(points, voxel_size):
    if voxel_size is None or voxel_size <= 0 or points.shape[0] == 0:
        return points
    coords = np.floor(points / voxel_size).astype(np.int64)
    coords -= coords.min(axis=0)
    mx = coords.max(axis=0) + 1
    key = (coords[:, 0] * mx[1] + coords[:, 1]) * mx[2] + coords[:, 2]
    _, inv = np.unique(key, return_inverse=True)
    n_vox = int(inv.max()) + 1
    counts = np.bincount(inv, minlength=n_vox)
    out = np.empty((n_vox, points.shape[1]), dtype=points.dtype)
    for d in range(points.shape[1]):
        out[:, d] = np.bincount(inv, weights=points[:, d], minlength=n_vox) / counts
    return out

def unique_rows(a):
    a = np.ascontiguousarray(a)
    v = a.view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
    _, idx = np.unique(v, return_index=True)
    return a[idx]


class PoseDataset(data.Dataset):
    def __init__(
        self,
        mode,
        config,
        data_augmentation=False,
    ):
        self.patients = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
        #self.patients = [1]

        if mode not in ('train', 'val', 'test'):
            raise ValueError(
                f"Unknown mode '{mode}'. Expected 'train', 'val' or 'test'."
            )
        self.mode = mode
        self.root = config.data.dataset_root

        self.real_syn = config.data.dataset

        self.data_augmentation = data_augmentation

        self.list_label = []
        self.list_img = []
        self.list_liverPcd = []
        self.list_patient = []
        self.list_rank = []
        self.list_scale = []
        self.list_posemeta = {}
        self.pre_model = {}
        self.pre_model_normal = {}
        self.list_camK = {}

        self.rot_factor = 1.
        self.angle_sigma = 0.06
        self.angle_clip = 0.18
        self.scale_low = 0.8
        self.scale_high = 1.25
        self.augment_noise = 0.0#0.002 # no noise augmentation

        self.max_points = config.train.point_limit

        self.vox_size = config.data.voxel_size

        self.overlap_radius = config.model.ground_truth_matching_radius
        self.pre_model_scale = {}
        self.pre_model_reconstructed_faces = {}
        self.pre_model_reconstructed_verts = {}
        self.rng = np.random

        item_count = 0
        for patient in self.patients:
            if self.mode == 'train':
                fname = 'train_real.txt' if self.real_syn == 'real' else 'train_syn.txt'
            elif self.mode == 'val':
                fname = 'val_real.txt'   if self.real_syn == 'real' else 'val_syn.txt'
            else:
                fname = 'test_real.txt'  if self.real_syn == 'real' else 'test_syn.txt'
            input_file = open('{0}/{1}/{2}'.format(self.root, '%02d' % patient, fname))

            while 1:
                item_count = item_count + 1
                input_line = input_file.readline()
                if not input_line:
                    break
                if input_line[-1:] == '\n':
                    input_line = input_line[:-1]
                if self.real_syn == 'syn':
                    self.list_liverPcd.append('{0}/{1}/syn/liverPcds/{2}.ply'.format(self.root, '%02d' % patient, input_line))
                    self.list_patient.append(patient)
                    self.list_rank.append(input_line[-5:])
                else:
                    self.list_label.append('{0}/{1}/real/labels/{2}/label.png'.format(self.root, '%02d' % patient, input_line))
                    self.list_img.append('{0}/{1}/real/labels/{2}/img.png'.format(self.root, '%02d' % patient, input_line))
                    self.list_liverPcd.append('{0}/{1}/real/liverPcds/{2}.ply'.format(self.root, '%02d' % patient, input_line))
                    self.list_patient.append(patient)
                    match = re.search(r"frame_(\d+)_json", input_line)
                    if match:
                        cur_rank = int(match.group(1))
                    self.list_rank.append(cur_rank)
                    self.list_scale.append('{0}/{1}/real/scale/{2}.txt'.format(self.root, '%02d' % patient, input_line))

            if self.real_syn == 'syn':
                pose_file = open('{0}/{1}/syn/camPose.yml'.format(self.root, '%02d' % patient), 'r')
                self.list_posemeta[patient] = yaml.safe_load(pose_file)
            else:
                camK_file = open('{0}/{1}/real/CAM_K.yml'.format(self.root, '%02d' % patient), 'r')
                intrinsic = np.array(yaml.safe_load(camK_file)['00000'][0]['cam_K']).reshape(3, 3)
                self.list_camK[patient] = intrinsic

            # load pre-operative model
            self.preope_mesh_path = '{0}/{1}/model/reconstructed_mesh_world_m.obj'.format(self.root, '%02d' % patient)

            points, normals, scale = self._build_or_load_pre_model(self.preope_mesh_path)
            self.pre_model[patient] = points
            self.pre_model_normal[patient] = normals
            self.pre_model_scale[patient] = scale

            if self.real_syn == 'real':
                reconstructed_faces, reconstructed_verts = self._build_or_load_faces(
                    self.preope_mesh_path, points, normals)
                self.pre_model_reconstructed_faces[patient] = reconstructed_faces
                self.pre_model_reconstructed_verts[patient] = reconstructed_verts
 

        with ProcessPoolExecutor(max_workers=8) as ex:
            self.list_liverPcd_npy = list(tqdm(
                ex.map(_ensure_npy, self.list_liverPcd, chunksize=16),
                total=len(self.list_liverPcd),
                desc=f"{self.mode}: ply→npy cache",
            ))
        self.length = len(self.list_liverPcd)
        print("{0} data is {1}".format(self.mode, self.length))

    def random_idx(self):
        n = self.length
        idx = self.rng.randint(0, n)
        return idx

    PRE_MODEL_VERSION = 1          # bump if the build logic below changes
    PRE_MODEL_N_SAMPLE = 200000
    PRE_MODEL_VOXEL = 0.0006

    def _build_or_load_pre_model(self, mesh_path):
        """Return (points[N,3] f32, normals[N,3] f32, scale float) for the
        pre-operative model, caching the expensive build keyed by version +
        sampling params, so it is computed once and reloaded thereafter."""
        cache_path = "{0}.premodel.v{1}.s{2}.vox{3:g}.npz".format(
            mesh_path, self.PRE_MODEL_VERSION, self.PRE_MODEL_N_SAMPLE, self.PRE_MODEL_VOXEL
        )
 
        # ---- fast path: reload from cache ----
        if os.path.exists(cache_path):
            try:
                d = np.load(cache_path)
                return (d['points'].astype(np.float32),
                        d['normals'].astype(np.float32),
                        float(d['scale']))
            except Exception:
                pass 
 
        # ---- slow path: build from the mesh ----
        if not os.path.exists(mesh_path):
            raise FileNotFoundError(
                f"Pre-operative mesh not found: {mesh_path}\n"
                f"  dataset_root = {self.root!r} (cwd = {os.getcwd()!r}).\n"
                f"  Use an ABSOLUTE dataset_root so the path does not depend on "
                f"the launch directory."
            )
        preope_mesh = o3d.io.read_triangle_mesh(mesh_path)
        if len(preope_mesh.triangles) == 0:
            raise RuntimeError(
                f"Loaded mesh has no triangles: {mesh_path}\n"
                f"  The file exists but Open3D/ASSIMP could not parse it "
                f"(empty, corrupt, or not a valid .obj)."
            )
 
        preope_pcd = preope_mesh.sample_points_uniformly(self.PRE_MODEL_N_SAMPLE)
        preope_pcd = preope_pcd.voxel_down_sample(voxel_size=self.PRE_MODEL_VOXEL)
        points = np.asarray(preope_pcd.points, dtype=np.float32)  # m
 
        s_c = points.mean(axis=0)
        scale = float(np.max(np.sqrt(np.sum((points - s_c) ** 2, axis=1))))
 
        preope_mesh.compute_vertex_normals()
        original_normals = np.asarray(preope_mesh.vertex_normals)
        tree = cKDTree(np.asarray(preope_mesh.vertices))
        _, indices = tree.query(points, k=1)
        normals = original_normals[indices].squeeze().astype(np.float32)
 
        tmp = "{0}.{1}.tmp.npz".format(cache_path, os.getpid())
        try:
            np.savez(tmp, points=points, normals=normals, scale=np.float32(scale))
            os.replace(tmp, cache_path)
        except Exception:
            if os.path.exists(tmp):
                os.remove(tmp)
        return points, normals, scale

    def _build_or_load_faces(self, mesh_path, points, normals):
        """Return (faces[F,3] int32, verts[V,3] float32) from the SAME mesh load,
        so faces index exactly into verts (required for the overlay)."""
        cache_path  = mesh_path + ".faces.v2.npz"   # v2: now stores verts too
        if os.path.exists(cache_path):
            try:
                d = np.load(cache_path)
                return d['faces'].astype(np.int32), d['verts'].astype(np.float32)
            except Exception:
                pass  

        mesh  = o3d.io.read_triangle_mesh(mesh_path)
        faces = np.asarray(mesh.triangles).astype(np.int32)
        verts = np.asarray(mesh.vertices).astype(np.float32)   
        if faces.shape[0] == 0:
            return np.zeros((0, 3), np.int32), np.zeros((0, 3), np.float32)

        assert int(faces.max()) < verts.shape[0], \
            f"faces idx {int(faces.max())} >= #verts {verts.shape[0]} in {mesh_path}"

        tmp = "{0}.{1}.tmp.npz".format(cache_path, os.getpid())
        try:
            np.savez(tmp, faces=faces, verts=verts)
            os.replace(tmp, cache_path)
        except Exception:
            if os.path.exists(tmp):
                os.remove(tmp)
        return faces, verts

    # ------------------------------------------------------------------ #
    #  Normalization
    # ------------------------------------------------------------------ #

    def normalize_pair(self, src, tgt, R, t, M):
        if (not np.isfinite(M)) or M < 1e-8:
            return None
        s_c = src.mean(axis=0)
        t_c = tgt.mean(axis=0)
        src_u = (src - s_c) / M
        tgt_u = (tgt - t_c) / M

        t_norm = (R @ s_c + t.reshape(-1) - t_c) / M
        return src_u, tgt_u, R, t_norm.astype(np.float32), \
            np.float32(M), s_c.astype(np.float32), t_c.astype(np.float32)

    CLEAN_VERSION = 1  # bump if you change the cleaning logic below

    def _load_clean_intra(self, index):
        base = self.list_liverPcd_npy[index]
        clean_path = f"{base}.clean.v{self.CLEAN_VERSION}.npy"
        bbx_path = f"{base}.clean.v{self.CLEAN_VERSION}.bbx.npy"

        if os.path.exists(clean_path):
            intra = np.load(clean_path)
            if self.real_syn == 'syn':
                return intra
            return intra, np.load(bbx_path)

        pts = np.load(base, mmap_mode='r')

        bounding_box_center = None
        if self.real_syn == 'real':
            pts_np = np.asarray(pts, dtype=np.float32)
            aabb_center = (pts_np.max(axis=0) + pts_np.min(axis=0)) / 2.0
            bounding_box_center = aabb_center.astype(np.float32) / 1000.0  # m

        intra = unique_rows(pts[::8].astype(np.float32, copy=True))
        if intra.shape[0] > 21:
            d, _ = cKDTree(intra).query(intra, k=21)
            md = d[:, 1:].mean(axis=1)
            intra = intra[md < md.mean() + 2.0 * md.std()]
        intra = intra / 1000.0
        intra = intra[~np.all(intra == 0., axis=-1)]
        intra = intra.astype(np.float32)

        tmp = f"{clean_path}.{os.getpid()}.tmp"
        try:
            with open(tmp, "wb") as f:   
                np.save(f, intra)
            os.replace(tmp, clean_path)  
        except Exception:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise

        if self.real_syn == 'syn':
            return intra
        np.save(bbx_path, bounding_box_center)
        return intra, bounding_box_center

    def get_item(self, index):

        which_patient = self.list_patient[index]
        if self.real_syn == 'syn':
            gt_pose = self.list_posemeta[which_patient]
        rank = self.list_rank[index]

        if self.real_syn == 'real':
            cam_K = self.list_camK[which_patient]
            if which_patient == 5:
                img_size = np.array([2160, 3840])
            else:
                img_size = np.array([1080, 1920])
        
        if self.real_syn == 'syn':
            pose_R = np.array(gt_pose[rank][0]["cam_R_c2w"]).reshape(3, 3).astype(np.float32)
            pose_t = np.array(gt_pose[rank][0]["cam_t_c2w"]).reshape(3, 1).astype(np.float32) / 1000.0  # m

            tsfm = np.eye(4)
            tsfm[:3,:3]=pose_R
            tsfm[:3,3]=pose_t.flatten()
            tsfm = np.linalg.inv(tsfm)  ##[4,4]
            pose_R = tsfm[:3, :3].astype(np.float32)
            pose_t = tsfm[:3, 3].reshape(3, 1).astype(np.float32)  # W2C (S2T)
        else:
            pose_R = np.eye(3).astype(np.float32)
            pose_t = np.zeros([3, 1]).astype(np.float32)
            tsfm = np.eye(4).astype(np.float32)

        if self.real_syn == 'real':
            labels = np.array(Image.open(self.list_label[index]))
            imgs = np.array(Image.open(self.list_img[index]))  ## original images
            liver_labels = labels.copy()
            
        if self.real_syn == 'real':
            scale = np.loadtxt(self.list_scale[index])

        preope_pcd_src = self.pre_model[which_patient]
        if self.real_syn == 'real':
            preope_pcd_src_normal = self.pre_model_normal[which_patient]

        if self.real_syn == 'syn':
            intra_liver_pcd_tgt = self._load_clean_intra(index)
        else : 
            intra_liver_pcd_tgt, bounding_box_center = self._load_clean_intra(index)

        if self.real_syn == 'real':
            ocv2blender = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
            intra_liver_pcd_tgt = np.dot(ocv2blender, intra_liver_pcd_tgt.T).T

        else:
            ocv2blender = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])


        if intra_liver_pcd_tgt.shape[0] == 0:
            return None  # degenerate sample

        if self.real_syn == 'real':
            reconstructed_faces = self.pre_model_reconstructed_faces[which_patient]
            reconstructed_verts = self.pre_model_reconstructed_verts[which_patient]

        if self.data_augmentation and self.real_syn == 'syn':
            euler_ab = np.clip(self.angle_sigma * self.rng.randn(3),
                               -self.angle_clip, self.angle_clip)
            rot_ab = Rotation.from_euler('zyx', euler_ab).as_matrix()

            scales = self.rng.uniform(self.scale_low, self.scale_high)
            scale_matrix = np.diag([scales, scales, scales])
            scale_matrix_inv = np.linalg.inv(scale_matrix)

            if (self.rng.rand(1)[0] > 0.5):
                preope_pcd_src = np.matmul(rot_ab, preope_pcd_src.T).T
                pose_R = np.matmul(pose_R, rot_ab.T)

                preope_pcd_src = np.matmul(scale_matrix, preope_pcd_src.T).T
                pose_R = np.matmul(pose_R, scale_matrix_inv)
            else:
                intra_liver_pcd_tgt = np.matmul(rot_ab, intra_liver_pcd_tgt.T).T
                pose_R = np.matmul(rot_ab, pose_R)
                pose_t = np.matmul(rot_ab, pose_t)

                intra_liver_pcd_tgt = np.matmul(scale_matrix, intra_liver_pcd_tgt.T).T
                pose_R = np.matmul(scale_matrix, pose_R)
                pose_t = np.matmul(scale_matrix, pose_t)

            # Removed noise augmentation for corruption training
            #preope_pcd_src = preope_pcd_src + (self.rng.rand(*preope_pcd_src.shape) - 0.5) * self.augment_noise
            #intra_liver_pcd_tgt = intra_liver_pcd_tgt + (self.rng.rand(*intra_liver_pcd_tgt.shape) - 0.5) * self.augment_noise
        #elif self.data_augmentation:
            # real mode: identity GT, only metric noise
            #preope_pcd_src = preope_pcd_src + (self.rng.rand(*preope_pcd_src.shape) - 0.5) * self.augment_noise
            #intra_liver_pcd_tgt = intra_liver_pcd_tgt + (self.rng.rand(*intra_liver_pcd_tgt.shape) - 0.5) * self.augment_noise
        
        pose_R = pose_R.astype(np.float32)
        pose_t = pose_t.astype(np.float32)

        M = self.pre_model_scale[which_patient]
        norm = self.normalize_pair(preope_pcd_src, intra_liver_pcd_tgt, pose_R, pose_t, M)
        if norm is None:
            return None
        src_u, tgt_u, R_n, t_n, m, s_c, t_c = norm
                    
        src_u = voxel_downsample_np(src_u, self.vox_size)

        if src_u.shape[0] == 0 or tgt_u.shape[0] == 0:
            return None

        if self.max_points is not None and tgt_u.shape[0] > self.max_points:
            idx = self.rng.choice(tgt_u.shape[0], size=self.max_points, replace=False)
            tgt_u = tgt_u[idx]

        src_u = src_u.astype(np.float32)
        tgt_u = tgt_u.astype(np.float32)

        src_feats = np.ones_like(src_u[:, :1]).astype(np.float32)
        tgt_feats = np.ones_like(tgt_u[:, :1]).astype(np.float32)

        transform = np.eye(4, dtype=np.float32)
        transform[:3, :3] = R_n
        transform[:3, 3]  = t_n.reshape(-1)

        R_gt = tsfm[:3, :3].astype(np.float32)
        t_gt = tsfm[:3, 3].reshape(3, 1).astype(np.float32)
        # normalize using s_c / t_c / M:
        t_gt_norm = (R_gt @ s_c + t_gt.reshape(-1) - t_c) / m
        transform_for_gt = np.eye(4, dtype=np.float32)
        transform_for_gt[:3, :3] = R_gt
        transform_for_gt[:3, 3]  = t_gt_norm

        # metadata
        frame_name = "{0}_{1}".format('%02d' % which_patient, rank)

        item_dict = dict(
            src_points=src_u,
            ref_points=tgt_u,
            src_feats=src_feats,
            ref_feats=tgt_feats,
            transform=transform.astype(np.float32),
            transform_for_gt=transform_for_gt.astype(np.float32),
            scene_name="{0}_{1}".format(self.mode, '%02d' % which_patient),
            ref_frame=str(frame_name),
            src_frame=str(frame_name),
            overlap=str('Not known'),
            m=m,
        )

        if self.real_syn == 'real' :
            rgbs = np.zeros_like(intra_liver_pcd_tgt).astype(np.float32)

        if self.mode != 'train':
            # Normalization scalars are needed to denormalize predictions at test time
            item_dict['s_c'] = s_c
            item_dict['t_c'] = t_c

        if self.mode != 'train' and self.real_syn == 'real':
            item_dict['liver_label'] = liver_labels.astype(np.float32)
            item_dict['rgbs'] = rgbs.astype(np.float32)          # was 'rbgs' / no.float32
            item_dict['img_size'] = img_size.astype(np.int32)
            item_dict['ocv2blender'] = ocv2blender.astype(np.float32)
            item_dict['bbx_center'] = bounding_box_center.astype(np.float32)
            item_dict['scale'] = scale.astype(np.float32)
            item_dict['cam_k'] = cam_K.astype(np.float32)
            item_dict['imgs'] = imgs.astype(np.uint8)
            item_dict['preope_reconstructed_faces'] = reconstructed_faces.astype(np.int32)
            item_dict['preope_reconstructed_verts'] = reconstructed_verts.astype(np.float32)

        return item_dict

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.mode == 'train':
            data = self.get_item(idx)
            while data is None:
                item_name = self.random_idx()
                data = self.get_item(item_name)
            return data
        else:
            data = self.get_item(idx)
            while data is None:
                item_name = self.random_idx()
                data = self.get_item(item_name)
            return data

class CorruptedP2ILRegDataset(torch.utils.data.Dataset):
    def __init__(self, args, cor_dataset_root):
        super(CorruptedP2ILRegDataset, self).__init__()
        if args.corruption == 'clean':
            self.data_path = os.path.join(cor_dataset_root, args.corruption + '.npy')
        else : 
            self.data_path = os.path.join(cor_dataset_root, 'data_' + args.corruption + str(f'_{int(args.severity)}') + '.npy')
        self.data_dicts = np.load(self.data_path, allow_pickle=True)

    def __getitem__(self,item):
        return self.data_dicts[item]
    
    def __len__(self):
        return len(self.data_dicts)


if __name__ == '__main__':
    import yaml
    from easydict import EasyDict as edict
    from datasets.dataloader import get_dataloader, get_datasets
    from configs.models import architectures
    import open3d as o3d
    import numpy as np

    config_pth = '/configs/train/main_config.yaml'
    with open(config_pth, 'r') as f:
        config = yaml.safe_load(f)
    config = edict(config)
    dataset = PoseDataset(mode='train', config=config, data_augmentation=True)

    for idx in range(1):
        print("processiong: {0}".format(idx), end='r')
        data = dataset.__getitem__(idx)
        print("src_points shape: ", data['src_points'].shape)
        print("ref_points shape: ", data['ref_points'].shape)
        print("src_feats shape: ", data['src_feats'].shape)
        print("ref_feats shape: ", data['ref_feats'].shape)
        print("transform shape: ", data['transform'].shape)