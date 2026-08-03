import os
import csv
import numpy as np
import torch
from tqdm import tqdm
from extensions.chamfer_distance.chamfer_distance import ChamferDistance
from extensions.earth_movers_distance.emd import EarthMoverDistance

levels = [1, 2, 3, 4, 5]
corruptions = ['clean', 'uniform', 'gaussian', 'background_noise', 'impulse_noise', 'global_density_dec', 'local_density_dec', 'cutout', 'occlusion']
#corruptions = ['scale']
dataset_root = "../datasets_corruptions_p2ilreg"
output_csv = "evaluation_results_all.csv"

CD = ChamferDistance()
EMD = EarthMoverDistance()


def compute_point_cloud_stats(points):
    return {
        "mean_x": float(np.mean(points[:, 0])),
        "mean_y": float(np.mean(points[:, 1])),
        "mean_z": float(np.mean(points[:, 2])),
        "min_x":  float(np.min(points[:, 0])),
        "min_y":  float(np.min(points[:, 1])),
        "min_z":  float(np.min(points[:, 2])),
        "max_x":  float(np.max(points[:, 0])),
        "max_y":  float(np.max(points[:, 1])),
        "max_z":  float(np.max(points[:, 2])),
        "n_points": len(points),
    }


def aggregate_stats(stats_list):
    keys = [k for k in stats_list[0] if k != "n_points"]
    aggregated = {}
    for k in keys:
        vals = [s[k] for s in stats_list]
        aggregated[f"mean_{k}"] = float(np.mean(vals))
        aggregated[f"min_{k}"]  = float(np.min(vals))
        aggregated[f"max_{k}"]  = float(np.max(vals))
    n_points = [s["n_points"] for s in stats_list]
    aggregated["mean_n_points"] = float(np.mean(n_points))
    aggregated["min_n_points"]  = float(np.min(n_points))
    aggregated["max_n_points"]  = float(np.max(n_points))
    return aggregated


def evaluate(data_dicts, label):
    cd1_list, cd2_list, cd_list, emd_list, stats_list = [], [], [], [], []

    pbar = tqdm(data_dicts, desc=label)
    for d in pbar:
        ref_pc   = d['ref_points'][:, :3].astype(np.float32)
        src_pc   = d['src_points'][:, :3].astype(np.float32)
        transform   = d['transform']
        rotation    = transform[:3, :3]
        translation = transform[:3, 3]
        gt_points   = (np.matmul(src_pc, rotation.transpose(-1, -2)) + translation).astype(np.float32)

        if label.startswith('scale_'):
            scale_factor = d['scale']
            ref_inv = ref_pc
            ref_inv_c = np.mean(ref_inv, axis=0)
            ref_inv_scaled = (ref_inv - ref_inv_c)*(1/scale_factor) + ref_inv_c
            ref_pc = (rotation.T @ (ref_inv_scaled - translation).T).T
            gt_points = d['src_points'][:, :3].astype(np.float32)

        ref_tensor = torch.tensor(ref_pc).unsqueeze(0).contiguous().cuda()
        gt_tensor  = torch.tensor(gt_points).unsqueeze(0).contiguous().cuda()

        dist1, dist2 = CD(ref_tensor, gt_tensor)
        cd1 = torch.mean(dist1).item()
        cd2 = torch.mean(dist2).item()
        cd1_list.append(cd1)
        cd2_list.append(cd2)
        cd = (torch.mean(dist1) + torch.mean(dist2)).item()
        cd_list.append(cd)

        emd_dist = EMD(ref_tensor, gt_tensor)
        emd = torch.mean(emd_dist).item()
        emd_list.append(emd)

        stats_list.append(compute_point_cloud_stats(ref_pc))

        pbar.set_postfix({"CD": f"{np.mean(cd_list):.4f}", "EMD": f"{np.mean(emd_list):.4f}"})

    agg_stats = aggregate_stats(stats_list)

    return {
        "label":    label,
        "mean_cd1":  float(np.mean(cd1_list)),
        "std_cd1":   float(np.std(cd1_list)),
        "mean_cd2":  float(np.mean(cd2_list)),
        "std_cd2":   float(np.std(cd2_list)),
        "mean_cd":  float(np.mean(cd_list)),
        "std_cd":   float(np.std(cd_list)),
        "mean_emd": float(np.mean(emd_list)),
        "std_emd":  float(np.std(emd_list)),
        **agg_stats,
    }



#clean_path = os.path.join(dataset_root, "clean.npy")
#clean_dicts = np.load(clean_path, allow_pickle=True)
#result = evaluate(clean_dicts, label="clean")

fieldnames = None
""" if fieldnames is None:
    fieldnames = list(result.keys())
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

with open(output_csv, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writerow(result) """

for corruption in corruptions:
    if corruption == "clean":
        data_path = os.path.join(dataset_root, f"{corruption}.npy")
        data_dicts = np.load(data_path, allow_pickle=True)
        label = f"{corruption}"
        result = evaluate(data_dicts, label=label)

        if fieldnames is None:
            fieldnames = list(result.keys())
            with open(output_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

        with open(output_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(result)

    else : 
        for severity in levels:
            data_path = os.path.join(dataset_root, f"data_{corruption}_{severity}.npy")
            data_dicts = np.load(data_path, allow_pickle=True)
            label = f"{corruption}_s{severity}"
            result = evaluate(data_dicts, label=label)

            if fieldnames is None:
                fieldnames = list(result.keys())
                with open(output_csv, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()

            with open(output_csv, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow(result)

print(f"\nResults saved to {output_csv}")
