import argparse
import os
import os.path as osp
import pickle
import random
from typing import Dict

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

from tqdm import tqdm
import torch
import torch.utils.data

from create_corrupted_dataset import ThreeDMatchPairDataset


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="silico")
    parser.add_argument('--dataset_root', type=str, default="../../P2P/Dataset/")
    parser.add_argument('--corrupted_dataset_path', type=str, default='../datasets')
    parser.add_argument('--COR', type=str, required=True)
    parser.add_argument('--SEV', type=str, required=True)

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

    cor = COR
    sev = SEV
    print(f"\nProcessing corruption: {cor}, sev: {sev}")
    
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
        
    #np.save(corruption_path, data_dicts)
    
    
if __name__ == "__main__":
    main()

    
        
