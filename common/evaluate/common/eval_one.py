"""
Evaluate a single results .json file: print the BS-weighted mean ± std of a
chosen metric, optionally filtered by visibility range.

Usage:
    python eval_one_json.py /path/to/results.json
    python eval_one_json.py /path/to/results.json --metric RRE
    python eval_one_json.py /path/to/results.json --metric RTE --vis_lo 0.3 --vis_hi 0.7
"""

import os
import json
import argparse
import numpy as np


# ------------------------------------------------------------------ config
FILE_STATS_DEFAULT = (
    "/home/nbodelot/projects/def-egranger/nbodelot/registration"
    "/pc-registration/silico/data/P2P/Dataset/Deform_mesh_npz_test/stat_svd.npz"
)


# ------------------------------------------------------------------ helpers
def load_vis_mask(path, vis_lo, vis_hi):
    stat = np.load(path)
    vis = stat["vis"]
    return (vis >= vis_lo) & (vis < vis_hi)


def load_json_results(json_path, metric):
    """Return (values, bs) arrays, or (None, None) if file missing/empty.

    `bs` defaults to 1 for any line that doesn't carry a BS field, so the
    weighted mean degenerates to the simple mean for legacy logs.
    """
    if not os.path.isfile(json_path):
        return None, None
    values, bs_values = [], []
    with open(json_path) as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  [warn] skipping malformed line {ln}: {e}")
                continue
            if metric in obj:
                v = obj[metric]
                # unwrap length-1 lists from B=1 dumps
                if isinstance(v, list) and len(v) == 1:
                    v = v[0]
                values.append(v)
                bs_values.append(obj.get("BS", 1))
    if not values:
        return None, None
    return np.array(values, dtype=float), np.array(bs_values, dtype=float)


def apply_mask(arr, bs, mask):
    if arr is None:
        return None, None
    n = min(arr.shape[0], mask.shape[0])
    m = mask[:n]
    return arr[:n][m], bs[:n][m]


def weighted_mean_std(arr, weights):
    """Frequency-weighted mean and std (population formula)."""
    w_sum = float(weights.sum())
    if w_sum <= 0:
        return None, None
    mean = float(np.average(arr, weights=weights))
    var = float(np.average((arr - mean) ** 2, weights=weights))
    return mean, float(np.sqrt(var))


# ------------------------------------------------------------------ main
def main():
    parser = argparse.ArgumentParser(
        description="Print the BS-weighted mean ± std of a metric from one .json results file."
    )
    parser.add_argument("json_path", type=str,
                        help="Path to the results .json (JSONL: one JSON object per line).")
    parser.add_argument("--metric", type=str, default="RE_markers",
                        help="Metric key to read from each line (default: RE_markers).")
    parser.add_argument("--vis_lo", type=float, default=0.0,
                        help="Lower bound of visibility range, inclusive (default: 0.0).")
    parser.add_argument("--vis_hi", type=float, default=1.0,
                        help="Upper bound of visibility range, exclusive (default: 1.0).")
    parser.add_argument("--stat_file", type=str, default=FILE_STATS_DEFAULT,
                        help="Path to stat_svd.npz for visibility filtering. "
                             "Pass 'none' to skip masking.")
    args = parser.parse_args()

    print(f"\nFile   : {args.json_path}")
    print(f"Metric : {args.metric}")

    arr, bs = load_json_results(args.json_path, args.metric)
    if arr is None:
        print("\n  NO DATA (file missing or metric not found in any line).")
        return

    # Optional visibility masking
    if args.stat_file.lower() != "none":
        print(f"Vis    : [{args.vis_lo}, {args.vis_hi})")
        vis_mask = load_vis_mask(args.stat_file, args.vis_lo, args.vis_hi)
        print(f"         {vis_mask.sum()} / {vis_mask.size} samples in range")
        arr, bs = apply_mask(arr, bs, vis_mask)
        if arr is None or arr.size == 0:
            print("\n  NO DATA after visibility masking.")
            return

    mean, std = weighted_mean_std(arr, bs)
    n_iters = int(arr.size)
    n_samples = int(bs.sum())

    print(f"\n  {args.metric} = {mean:.4f} ± {std:.4f}")
    print(f"  iters     = {n_iters}")
    print(f"  weighted N = {n_samples}")
    print(f"  min / median / max = {arr.min():.4f} / {float(np.median(arr)):.4f} / {arr.max():.4f}")


if __name__ == "__main__":
    main()