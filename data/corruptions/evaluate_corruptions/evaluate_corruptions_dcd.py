import os
import csv
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict

# ============================================================================
# Density-aware Chamfer Distance (DCD)
#   Wu et al., "Density-aware Chamfer Distance as a Comprehensive Metric for
#   Point Cloud Completion", NeurIPS 2021.
#
# Why DCD instead of plain CD: CD is blind to density and to extra/outlier
# points (that is why, in your table, 'background noise' scored as *cleaner*
# than clean). DCD fixes that with a per-target occupancy weight, while staying
# nearest-neighbour cheap (no O(N^3) optimal assignment like EMD).
#
# Properties:
#   * lower value  = more similar  (clean-vs-clean should be the minimum).
#   * range ~ [0, 1].
#   * handles UNEQUAL point counts between the two clouds (EMD cannot).
#   * NN is computed in chunks, so peak memory is O(B * chunk * N), NOT O(B*N^2)
#     -> no need to downsample, and you can likely raise BATCH_SIZE now.
# ============================================================================

levels = [1, 2, 3, 4, 5]
corruptions = ['clean', 'uniform', 'gaussian', 'background_noise', 'impulse_noise', 'global_density_dec', 'local_density_dec', 'cutout', 'occlusion']
# corruptions = ['scale']
dataset_root = "../datasets"
output_csv = "evaluation_results_p2psilico_DCD.csv"

device = torch.device("cuda")

# ----- DCD hyper-parameters -------------------------------------------------
BATCH_SIZE  = 24          # DCD is far lighter than EMD; raise this until OOM
POINT_CHUNK = 2048        # NN query block size; lower it if you ever OOM on huge N
N_LAMBDA    = 1.0         # density-weight exponent (paper default = 1)
ALPHA       = None        # temperature. None -> auto-calibrate from clean.npy once.
                          # IMPORTANT: alpha must match your coordinate scale (see notes).


# ---------------------------------------------------------------------------
# Core ops
# ---------------------------------------------------------------------------
@torch.no_grad()
def _nn_sq(query, ref, chunk):
    """For each point in `query`, squared distance + index of nearest point in `ref`.
    query (B, Nq, 3), ref (B, Nr, 3) -> dist2 (B, Nq), idx (B, Nq) [long].
    Chunked over Nq so the (B, Nq, Nr) matrix is never fully materialised."""
    B, Nq, _ = query.shape
    dist2 = query.new_empty(B, Nq)
    idx = torch.empty(B, Nq, dtype=torch.long, device=query.device)
    r2 = (ref * ref).sum(-1)                                  # (B, Nr)
    for s in range(0, Nq, chunk):
        q = query[:, s:s + chunk, :]                         # (B, c, 3)
        q2 = (q * q).sum(-1, keepdim=True)                   # (B, c, 1)
        cross = torch.bmm(q, ref.transpose(1, 2))            # (B, c, Nr)
        d = q2 - 2.0 * cross + r2.unsqueeze(1)               # (B, c, Nr) squared dist
        d.clamp_(min=0)
        md, mi = d.min(dim=2)
        dist2[:, s:s + chunk] = md
        idx[:, s:s + chunk] = mi
    return dist2, idx


@torch.no_grad()
def dcd(x, gt, alpha, n_lambda=N_LAMBDA, chunk=POINT_CHUNK):
    """Batched DCD. x (B, n_x, 3), gt (B, n_gt, 3) -> (B,) DCD per sample."""
    d_xg, i_xg = _nn_sq(x, gt, chunk)            # each x point -> nearest gt point
    d_gx, i_gx = _nn_sq(gt, x, chunk)            # each gt point -> nearest x point
    B, n_x = i_xg.shape
    n_gt = i_gx.shape[1]

    # occupancy: how many source points map onto each target point (density term)
    count_g = torch.zeros(B, n_gt, device=x.device, dtype=x.dtype).scatter_add_(
        1, i_xg, torch.ones_like(i_xg, dtype=x.dtype))
    w_x = (count_g.gather(1, i_xg).pow(n_lambda) + 1e-6).reciprocal()
    term_x = (1.0 - w_x * torch.exp(-alpha * d_xg)).mean(1)

    count_x = torch.zeros(B, n_x, device=x.device, dtype=x.dtype).scatter_add_(
        1, i_gx, torch.ones_like(i_gx, dtype=x.dtype))
    w_g = (count_x.gather(1, i_gx).pow(n_lambda) + 1e-6).reciprocal()
    term_g = (1.0 - w_g * torch.exp(-alpha * d_gx)).mean(1)

    return 0.5 * (term_x + term_g)               # (B,)


# ---------------------------------------------------------------------------
# Build the (ref, gt) clouds for one sample  (same math as the EMD script)
# ---------------------------------------------------------------------------
def build_pair(d, is_scale):
    ref_pc = d['ref_points'][:, :3].astype(np.float32)
    src_pc = d['src_points'][:, :3].astype(np.float32)
    transform   = d['transform']
    rotation    = transform[:3, :3].astype(np.float32)
    translation = transform[:3, 3].astype(np.float32)

    if is_scale:
        scale_factor = d['scale']
        ref_c = ref_pc.mean(axis=0)
        ref_scaled = (ref_pc - ref_c) * (1.0 / scale_factor) + ref_c
        ref_pc = (ref_scaled - translation) @ rotation        # == (R.T @ x.T).T
        gt_points = src_pc
    else:
        gt_points = src_pc @ rotation.T + translation         # == src @ R.T + t

    return (np.ascontiguousarray(ref_pc, dtype=np.float32),
            np.ascontiguousarray(gt_points, dtype=np.float32))


# ---------------------------------------------------------------------------
# Auto-calibrate alpha from the clean set (done ONCE, then frozen for all
# corruptions so the numbers stay comparable). Sets alpha so that the *median*
# nearest-neighbour squared distance on clean maps to exp(-1) ~ 0.37 -- i.e. the
# exponential spans a useful range instead of saturating at 0 or 1.
# ---------------------------------------------------------------------------
@torch.no_grad()
def calibrate_alpha(data_dicts, n_samples=64):
    sel = np.linspace(0, len(data_dicts) - 1, min(n_samples, len(data_dicts))).astype(int)
    meds = []
    for k in sel:
        ref_pc, gt_points = build_pair(data_dicts[k], is_scale=False)
        x  = torch.from_numpy(ref_pc).unsqueeze(0).to(device)
        gt = torch.from_numpy(gt_points).unsqueeze(0).to(device)
        d_xg, _ = _nn_sq(x, gt, POINT_CHUNK)
        meds.append(float(d_xg.median()))
    median_sq = float(np.median(meds))
    alpha = 1.0 / max(median_sq, 1e-12)
    print(f"[calibrate] median NN squared dist (clean) = {median_sq:.6g}  ->  alpha = {alpha:.6g}")
    return alpha


# ---------------------------------------------------------------------------
# Stats (unchanged behaviour, vectorised)
# ---------------------------------------------------------------------------
def compute_point_cloud_stats(points):
    mn = points.min(axis=0); mx = points.max(axis=0); mean = points.mean(axis=0)
    return {
        "mean_x": float(mean[0]), "mean_y": float(mean[1]), "mean_z": float(mean[2]),
        "min_x":  float(mn[0]),   "min_y":  float(mn[1]),   "min_z":  float(mn[2]),
        "max_x":  float(mx[0]),   "max_y":  float(mx[1]),   "max_z":  float(mx[2]),
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


# ---------------------------------------------------------------------------
# Evaluate one dataset
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(data_dicts, label, alpha):
    is_scale = label.startswith('scale_')

    refs, gts, stats_list = [], [], []
    for d in data_dicts:
        ref_pc, gt_points = build_pair(d, is_scale)
        refs.append(ref_pc)
        gts.append(gt_points)
        stats_list.append(compute_point_cloud_stats(ref_pc))

    dcd_arr = np.empty(len(refs), dtype=np.float64)

    # Bucket by (n_ref, n_gt) so each batch tensor stacks cleanly. DCD itself is
    # fine with n_ref != n_gt; bucketing is only needed to stack equal shapes.
    buckets = defaultdict(list)
    for i in range(len(refs)):
        buckets[(refs[i].shape[0], gts[i].shape[0])].append(i)

    running = []
    pbar = tqdm(total=len(refs), desc=label)
    for idxs in buckets.values():
        for s in range(0, len(idxs), BATCH_SIZE):
            chunk = idxs[s:s + BATCH_SIZE]
            ref_b = torch.from_numpy(np.stack([refs[i] for i in chunk])).to(device, non_blocking=True)
            gt_b  = torch.from_numpy(np.stack([gts[i]  for i in chunk])).to(device, non_blocking=True)

            dist = dcd(ref_b, gt_b, alpha=alpha).cpu().numpy()     # (B,)
            for j, i in enumerate(chunk):
                dcd_arr[i] = dist[j]
            running.extend(dist.tolist())
            pbar.set_postfix({"DCD": f"{np.mean(running):.4f}"})
            pbar.update(len(chunk))
    pbar.close()

    return {
        "label":    label,
        "mean_dcd": float(dcd_arr.mean()),
        "std_dcd":  float(dcd_arr.std()),
        **aggregate_stats(stats_list),
    }


# ---------------------------------------------------------------------------
# Calibrate alpha once (on clean), then run everything with that fixed alpha.
# ---------------------------------------------------------------------------
alpha = ALPHA
if alpha is None:
    clean_dicts = np.load(os.path.join(dataset_root, "clean.npy"), allow_pickle=True)
    alpha = calibrate_alpha(clean_dicts)

fieldnames = None
for corruption in corruptions:
    if corruption == "clean":
        data_dicts = np.load(os.path.join(dataset_root, f"{corruption}.npy"), allow_pickle=True)
        result = evaluate(data_dicts, label=f"{corruption}", alpha=alpha)

        if fieldnames is None:
            fieldnames = list(result.keys())
            with open(output_csv, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=fieldnames).writeheader()
        with open(output_csv, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerow(result)

    else:
        for severity in levels:
            data_path = os.path.join(dataset_root, f"data_{corruption}_{severity}.npy")
            data_dicts = np.load(data_path, allow_pickle=True)
            result = evaluate(data_dicts, label=f"{corruption}_s{severity}", alpha=alpha)

            if fieldnames is None:
                fieldnames = list(result.keys())
                with open(output_csv, "w", newline="") as f:
                    csv.DictWriter(f, fieldnames=fieldnames).writeheader()
            with open(output_csv, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=fieldnames).writerow(result)

print(f"\nResults saved to {output_csv}")