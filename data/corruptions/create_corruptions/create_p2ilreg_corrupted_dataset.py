import torch.utils.data as data
from PIL import Image
import os
#import cv2
#os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import os.path as osp
import torch
import numpy as np
#import torchvision.transforms as transforms
import argparse
import time
import random
import _pickle as cPickle
import numpy.ma as ma
import copy
#import scipy.misc
import scipy.io as scio
import open3d as o3d
from plyfile import PlyData
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import yaml
import re
import logging
#from pareconv.datasets.p2ilreg.utils import get_correspondences, to_o3d_pcd
logging.getLogger('PIL').setLevel(logging.WARNING)

# import imgaug.augmenters as iaa
current_dir = os.path.dirname(os.path.abspath(__file__))

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="syn")
    parser.add_argument('--dataset_root', type=str, default="../../Liver_regis")
    parser.add_argument('--corrupted_dataset_path', type=str, default='../datasets_corruptions_p2ilreg')
    parser.add_argument('--point_limit', type=int, default=8192)
    # Voxel size used to resample src/tgt in the NORMALIZED frame (unit ~= pre-op
    # model radius M). Must match the backbone's init_voxel_size so the saved
    # clouds live at exactly the resolution PARE-Conv operates at. Set <=0 to disable.
    parser.add_argument('--voxel_size', type=float, default=0.02)
    return parser.parse_args()

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
        args,
        data_augmentation=False,
        cor=None,
        sev=None,
    ):
        self.patients = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
        #self.patients = [3]

        if mode not in ('train', 'val', 'test'):
            raise ValueError(
                f"Unknown mode '{mode}'. Expected 'train', 'val' or 'test'."
            )
        self.mode = mode
        self.root = args.dataset_root

        self.real_syn = args.dataset

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
        self.augment_noise = 0.002

        self.max_points = args.point_limit

        # resolution for the normalized-frame voxelization in get_item (see there).
        # Falls back to 0.02 (= backbone init_voxel_size) if the arg is absent.
        self.vox_size = getattr(args, "voxel_size", 0.02)

        self.pre_model_scale = {}
        self.pre_model_reconstructed_faces = {}
        self.pre_model_reconstructed_verts = {}
        self.rng = np.random

        self.cor = cor
        self.sev = sev

        self.MAP = {
            'uniform': self.uniform_noise,
            'gaussian': self.gaussian_noise,
            'background_noise': self.background_noise, 
            'impulse_noise': self.impulse_noise,
            'global_density_dec': self.global_density_dec,
            'local_density_dec': self.local_density_dec,
            'cutout' : self.cutout,
            'occlusion': self.occlusion,
            }
        """ 
            'uniform': self.uniform_noise,
            'gaussian': self.gaussian_noise,
            'background_noise': self.background_noise, 
            'impulse_noise': self.impulse_noise,
            'global_density_dec': self.global_density_dec,
            'local_density_dec': self.local_density_dec,
            'cutout' : self.cutout,
            'occlusion': self.occlusion,
            'scale': self.aug_scale, """

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
 

        #self.list_liverPcd_npy = [_ensure_npy(p) for p in self.list_liverPcd]
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
        # item = self.list[idx]
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
                pass  # corrupt/old cache -> fall through and rebuild
 
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
 
        # fixed normalization scale = radius of the clean pre-op model (m)
        s_c = points.mean(axis=0)
        scale = float(np.max(np.sqrt(np.sum((points - s_c) ** 2, axis=1))))
 
        # propagate mesh-vertex normals to the sampled points (nearest vertex)
        preope_mesh.compute_vertex_normals()
        original_normals = np.asarray(preope_mesh.vertex_normals)
        tree = cKDTree(np.asarray(preope_mesh.vertices))
        _, indices = tree.query(points, k=1)
        normals = original_normals[indices].squeeze().astype(np.float32)
 
        # atomic write (safe under DDP / multiple processes)
        tmp = "{0}.{1}.tmp.npz".format(cache_path, os.getpid())
        try:
            np.savez(tmp, points=points, normals=normals, scale=np.float32(scale))
            os.replace(tmp, cache_path)
        except Exception:
            if os.path.exists(tmp):
                os.remove(tmp)
            # caching is best-effort: never fail the run because it couldn't write
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
                pass  # old/corrupt cache -> rebuild

        mesh  = o3d.io.read_triangle_mesh(mesh_path)
        faces = np.asarray(mesh.triangles).astype(np.int32)
        verts = np.asarray(mesh.vertices).astype(np.float32)   # metres, same frame as pre_model
        if faces.shape[0] == 0:
            return np.zeros((0, 3), np.int32), np.zeros((0, 3), np.float32)

        # invariant: every face index must point at a real vertex
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
    #  P2PSilico-style normalization
    # ------------------------------------------------------------------ #
    def normalize_pair(self, src, tgt, R, t, M):
        if (not np.isfinite(M)) or M < 1e-8:
            return None
        s_c = src.mean(axis=0)
        t_c = tgt.mean(axis=0)
        src_u = (src - s_c) / M
        tgt_u = (tgt - t_c) / M

        # tgt = R @ src + t  =>  tgt_u = R @ src_u + t_norm
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
            # real: the bbox center is needed for rendering; restore it from the sidecar
            return intra, np.load(bbx_path)

        pts = np.load(base, mmap_mode='r')

        bounding_box_center = None
        if self.real_syn == 'real':
            # `pts` is a numpy array (NOT an open3d cloud), so it has no
            # `get_axis_aligned_bounding_box()`. Compute the AABB center directly.
            # Raw points are stored in millimetres -> convert to metres.
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
            with open(tmp, "wb") as f:   # file handle => np.save does NOT append .npy
                np.save(f, intra)
            os.replace(tmp, clean_path)  # atomic on same filesystem
        except Exception:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise

        if self.real_syn == 'syn':
            return intra
        # cache the bbox center so the cache-hit branch above can also return it
        np.save(bbx_path, bounding_box_center)
        return intra, bounding_box_center

    def uniform_noise(self,src_raw, tgt_raw, clean_c, clean_m, sev) : 
        sigma = [2, 4, 6, 8, 10][sev - 1]
        N, C = tgt_raw.shape
        jitter = np.random.uniform(-sigma/2, sigma/2, (N, C)) 
        new_pc = tgt_raw + jitter

        # NOTE: per-point jitter is left un-smoothed HERE on purpose. Smoothing is
        # applied centrally in get_item (voxel_downsample_np at self.vox_size, in the
        # normalized frame) so that EVERY corruption + the clean path are resampled
        # identically and at the backbone's resolution. This matches the role that
        # norm_vox_scaled plays in the P2PSilico generator.
        return new_pc, None

    def gaussian_noise(self,src_raw, tgt_raw, clean_c, clean_m, sev) : 
        
        
        # quelle corruption en fonction de la sévérité
        sigma = [2, 3, 4, 5, 6][sev - 1]
        N, C = tgt_raw.shape
        jitter = np.random.normal(size=(N, C)) * (sigma/2)
        jitter = np.clip(jitter, -sigma, sigma)
        new_pc = tgt_raw + jitter

        return new_pc, None

    def background_noise(self,src_raw, tgt_raw, clean_c, clean_m, sev) : 
        N, C = tgt_raw.shape
        c = [N // 100, N // 50, N // 30, N // 15, N // 5][sev - 1]

        jitter = np.random.uniform(-clean_m + clean_c, clean_m + clean_c, (c, C))
        new_pc = np.concatenate((tgt_raw, jitter), axis=0).astype('float32')

        return new_pc, None
    
    def impulse_noise(self,src_raw, tgt_raw, clean_c, clean_m, sev) : 
        N, C = tgt_raw.shape
        c = [N // 45,  N // 30, N // 20, N // 10, N // 4][sev - 1]
        index = np.random.choice(N, c, replace=False)
        sigma = 10
        tgt_raw[index] += np.random.choice([-sigma, sigma], size=(c, C)) 

        return tgt_raw, None

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

    """ def global_density_dec(self,src_raw, tgt_raw, clean_c, clean_m, sev):
        v = self.vox_size
        n_v = [1.5*v, 2*v, 2.5*v, 3*v, 3.5*v][sev - 1]
        return new_pc, None """

    def global_density_dec(self, src_raw, tgt_raw, clean_c, clean_m, sev):
        N = tgt_raw.shape[0]
        keep = [1/(1.5*1.5), 1/(2*2), 1/(2.5*2.5), 1/(3*3), 1/(3.5*3.5)][sev - 1]
        n_keep = max(int(round(N * keep)), 1)
        idx = np.random.choice(N, n_keep, replace=False)
        return tgt_raw[idx], None


    def local_density_dec(self,src_raw, tgt_raw, clean_c, clean_m, sev):
        N, C = tgt_raw.shape
        n = N //10
        c = [(1, n),  (2, n), (4, n), (6, n), (8, n)][sev - 1]
        pc = tgt_raw.copy()
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
        N, C = tgt_raw.shape
        n = N // 30 #sur 30
        c = [(2, n), (3, n), (5, n), (7, n), (10, n)][sev - 1]

        pc = tgt_raw.copy()
        for s in range(c[0]):
            np.random.seed(s)
            i = np.random.choice(pc.shape[0], 1)
            picked = pc[i]
            dist = np.sum((pc - picked) ** 2, axis=1, keepdims=True)
            idx = np.argpartition(dist, c[1], axis=0)[:c[1]]
            pc = np.delete(pc, idx.squeeze(), axis=0)
            
        return pc, None

    def occlusion(self, src_raw, tgt_raw, clean_c, clean_m, sev):
        
        instrument_diameters_mm = [6, 12, 18, 24, 30]
        radius = instrument_diameters_mm[sev - 1] / 2.0

        pc = tgt_raw.copy() 

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
        
        cloud_extent = max(np.ptp(u_coords), np.ptp(v_coords))
        offset       = np.random.uniform(-0.1, 0.1) * cloud_extent
        center_2d    = offset * shaft_normal

        coords_2d     = np.stack([u_coords, v_coords], axis=1) - center_2d
        dist_to_shaft = np.abs(coords_2d @ shaft_normal)

        in_shadow = dist_to_shaft < radius
        pc_occluded = pc[~in_shadow]   

        if pc_occluded.shape[0] < 10:
            pc_occluded = pc

        
        return pc_occluded, None

    def get_item(self, index):
        cor = self.cor
        sev = self.sev
        MAP = self.MAP

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
            return None  # degenerate sample, retry in __getitem__

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

            preope_pcd_src = preope_pcd_src + (self.rng.rand(*preope_pcd_src.shape) - 0.5) * self.augment_noise
            intra_liver_pcd_tgt = intra_liver_pcd_tgt + (self.rng.rand(*intra_liver_pcd_tgt.shape) - 0.5) * self.augment_noise
        elif self.data_augmentation:
            # real mode: identity GT, only metric noise
            preope_pcd_src = preope_pcd_src + (self.rng.rand(*preope_pcd_src.shape) - 0.5) * self.augment_noise
            intra_liver_pcd_tgt = intra_liver_pcd_tgt + (self.rng.rand(*intra_liver_pcd_tgt.shape) - 0.5) * self.augment_noise

        pose_R = pose_R.astype(np.float32)
        pose_t = pose_t.astype(np.float32)

        #here R and t are correct, maybe not after

        M = self.pre_model_scale[which_patient] 
        
        #src_u, tgt_u, R_n, t_n, m, s_c, t_c = norm
        clean_c_scaled = np.mean(intra_liver_pcd_tgt, axis=0)
        intra_liver_pcd_tgt = intra_liver_pcd_tgt * 1000 # mm  
        preope_pcd_src = preope_pcd_src * 1000 # mm

        if cor is not None :
            intra_liver_pcd_tgt_cp = intra_liver_pcd_tgt.copy()
            clean_c = np.mean(intra_liver_pcd_tgt, axis=0)
            clean_centered = intra_liver_pcd_tgt - clean_c
            clean_m = np.max(np.sqrt(np.sum(clean_centered ** 2, axis=1)))
            intra_liver_pcd_tgt, cor_transform = self.MAP[cor](preope_pcd_src, intra_liver_pcd_tgt_cp, clean_c, clean_m, sev)
       
        """ norm = self.normalize_pair(preope_pcd_src, intra_liver_pcd_tgt, pose_R, pose_t, M)
        if norm is None:
            return None
        src_u, tgt_u, R_n, t_n, m, s_c, t_c = norm """

        intra_liver_pcd_tgt = intra_liver_pcd_tgt / 1000 # m  
        preope_pcd_src = preope_pcd_src / 1000 # m

        if (not np.isfinite(M)) or M < 1e-8:
            return None
        s_c = preope_pcd_src.mean(axis=0)
        t_c = clean_c_scaled
        src_u = (preope_pcd_src - s_c) / M
        tgt_u = (intra_liver_pcd_tgt - t_c) / M

        R_n = pose_R
        t_n = (pose_R @ s_c + pose_t.reshape(-1) - t_c) / M
        #t_n = (pose_R @ s_c + pose_t.reshape(-1) * 1000.0 - t_c) / (M * 1000.0)
        
        t_norm = t_n.astype(np.float32)
        m = np.float32(M)   # normalization scale (radius of clean pre-op model, m). NB: was `np.float32(M),` (a 1-tuple) before.
        s_c = s_c.astype(np.float32)
        t_c = t_c.astype(np.float32)

        if self.vox_size is not None and self.vox_size > 0:
            src_u = voxel_downsample_np(src_u.astype(np.float32), self.vox_size)
            tgt_u = voxel_downsample_np(tgt_u.astype(np.float32), self.vox_size)

        if src_u.shape[0] == 0 or tgt_u.shape[0] == 0:
            return None

        """ if self.max_points is not None and src_u.shape[0] > self.max_points:
            idx = self.rng.choice(src_u.shape[0], size=self.max_points, replace=False)
            src_u = src_u[idx] """
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
        # normalize the same way as normalize_pair, reusing s_c / t_c / M:
        t_gt_norm = (R_gt @ s_c + t_gt.reshape(-1) - t_c) / m
        transform_for_gt = np.eye(4, dtype=np.float32)
        transform_for_gt[:3, :3] = R_gt
        transform_for_gt[:3, 3]  = t_gt_norm

        """ if self.real_syn == 'syn':
            correspondences = get_correspondences(to_o3d_pcd(src_u), to_o3d_pcd(tgt_u), transform, self.overlap_radius)
            if correspondences.shape[0] < 1000:
                print("correspondences shape: ", correspondences.shape)
                return None """

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
            # (valid for BOTH syn and real).
            item_dict['s_c'] = s_c
            item_dict['t_c'] = t_c

        # Rendering / silhouette-Dice metrics are only defined for the real set,
        # and the variables below (liver_labels, rgbs, img_size, scale, cam_K,
        # bounding_box_center) are ONLY assigned in the real branch. Guarding by
        # `self.real_syn == 'real'` prevents NameErrors during syn-mode testing.
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
            # return self.get_item(idx)

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


def main() :
    args = get_args()
    t = PoseDataset('test', args, data_augmentation=False)

    os.makedirs(args.corrupted_dataset_path, exist_ok=True)

    clean_path = osp.join(args.corrupted_dataset_path, 'clean.npy')
    if not osp.isfile(clean_path): 

        print(f"\nProcessing clean data")

        data_dicts_clean = []
        for index in tqdm(range(len(t))) : 
            data_dict = t.__getitem__(index)
            data_dicts_clean.append(data_dict)

            viz = False
            if viz : 
                    out_dir = osp.join("./viz", 'clean')
                    if not osp.exists(out_dir) : 
                        os.makedirs(out_dir)

                    transform = data_dict['transform']
                    rotation = transform[:3, :3]  # (B, 3, 3)
                    translation = transform[:3, 3]  # (B, 1, 3)
                    gt_points = np.matmul(data_dict['src_points'], rotation.transpose(-1, -2)) + translation
                    
                    out_path1 = osp.join(out_dir, f'clean.png')
                    visualize(data_dict['ref_points'], out_path1) #gt_points , data_dict['src_points']
                    out_path2 = osp.join(out_dir, f'clean_gt.png')
                    visualize(data_dict['ref_points'], out_path2, gt_points)
        
        np.save(clean_path, data_dicts_clean)

    for cor in t.MAP.keys():
        print(f"\nProcessing corruption: {cor}")
        for sev in [1,2,3,4,5]:
            corruption_path = osp.join(args.corrupted_dataset_path, f'data_{cor}_{str(sev)}.npy')
            if not osp.isfile(corruption_path): 

                print(f"  Severity {sev}...")
                data_dicts = []
                t = PoseDataset(mode='test', args=args, data_augmentation=False, cor=cor, sev=sev)

                for index in tqdm(range(len(t))):
                    data_dict = t.__getitem__(index)
                    data_dicts.append(data_dict)

                    viz = False
                    if viz : 
                        out_dir = osp.join("./viz", cor)
                        if not osp.exists(out_dir) : 
                            os.makedirs(out_dir)

                        transform = data_dict['transform']
                        rotation = transform[:3, :3]  # (B, 3, 3)
                        translation = transform[:3, 3]  # (B, 1, 3)
                        gt_points = np.matmul(data_dict['src_points'], rotation.transpose(-1, -2)) + translation
                        
                        out_path1 = osp.join(out_dir, f'{cor}_{sev}.png')
                        visualize(data_dict['ref_points'], out_path1) #gt_points , data_dict['src_points']
                        out_path2 = osp.join(out_dir, f'{cor}_{sev}_gt.png')
                        visualize(data_dict['ref_points'], out_path2, gt_points)


                viz = False
                if viz : 
                    out_dir = osp.join("./viz", cor)
                    if not osp.exists(out_dir) : 
                        os.makedirs(out_dir)

                    transform = data_dict['transform']
                    rotation = transform[:3, :3]  # (B, 3, 3)
                    translation = transform[:3, 3]  # (B, 1, 3)
                    gt_points = np.matmul(data_dict['src_points'], rotation.transpose(-1, -2)) + translation
                    
                    out_path1 = osp.join(out_dir, f'{cor}_{sev}.png')
                    visualize(data_dict['ref_points'], out_path1) #gt_points , data_dict['src_points']
                    out_path2 = osp.join(out_dir, f'{cor}_{sev}_gt.png')
                    visualize(data_dict['ref_points'], out_path2, gt_points)


                np.save(corruption_path, data_dicts)

if __name__ == "__main__":
    main()