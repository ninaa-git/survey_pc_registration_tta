import os
import json
import argparse
import numpy as np
import csv

method = "Source_Only"
# ------------------------------------------------------------------ config
base_path = (
    "/home/nbodelot/projects/def-egranger/nbodelot/registration"
    f"/pc-registration/silico/PARENet/output/P2ILReg/{method}/registration"
)


file_stats = (
    "/home/nbodelot/projects/def-egranger/nbodelot/registration"
    "/pc-registration/silico/data/P2P/Dataset/Deform_mesh_npz_test/stat_svd.npz"
)

metric = "RE_markers"
extenstion = "" #"_0.02"

corruptions = [
    "clean",
    "uniform",
    "gaussian",
    "background_noise",
    "impulse_noise",
    "global_density_dec",
    "local_density_dec",
    "cutout",
    "occlusion",
]
col_names = ["uni", "gauss", "backg", "impul",
             "global-den-dec", "local-den-dec", "cut", "clsion"]

severities = (1, 2, 3, 4, 5)

# ------------------------------------------------------------------ helpers
def load_vis_mask(path, vis_lo, vis_hi):
    stat = np.load(path)
    vis  = stat["vis"]
    return (vis >= vis_lo) & (vis < vis_hi)


def load_json_results(json_path):
    """Return (values, bs) arrays, or (None, None) if file missing/empty.

    `bs` defaults to 1 for any line that doesn't carry a BS field, so the
    weighted mean degenerates to the simple mean for legacy logs.
    """
    if not os.path.isfile(json_path):
        return None, None
    values = []
    bs_values = []
    with open(json_path) as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  [warn] skipping malformed line {ln} in {json_path}: {e}")
                continue
            if metric in obj:
                values.append(obj[metric])
                bs_values.append(obj.get("BS", 1))
    if not values:
        return None, None
    return np.array(values, dtype=float), np.array(bs_values, dtype=float)


def apply_mask(arr, bs, mask):
    """Apply the visibility mask to both the value array and the BS array."""
    if arr is None:
        return None, None
    n = min(arr.shape[0], mask.shape[0])
    m = mask[:n]
    return arr[:n][m], bs[:n][m]


def get_json_path(corruption, severity):
    fname = f"test_{corruption}_{severity}{extenstion}.json"
    return os.path.join(base_path, fname)


def weighted_mean_std(arr, weights):
    """Frequency-weighted mean and std (population formula).

    Treats `weights` as the number of samples each value represents.
    """
    w_sum = float(weights.sum())
    if w_sum <= 0:
        return None, None
    mean = float(np.average(arr, weights=weights))
    var  = float(np.average((arr - mean) ** 2, weights=weights))
    return mean, float(np.sqrt(var))


def standard_stats(corruption, severity, vis_mask):
    """Return (mean, std) of RE_markers for the standard model (no adaptation),
    weighted by BS. Returns (None, None) if no data is found."""
    path = get_json_path(corruption, severity)
    arr, bs = load_json_results(path)
    arr, bs = apply_mask(arr, bs, vis_mask)
    if arr is not None and arr.size > 0:
        mean, std = weighted_mean_std(arr, bs)
        n_iters   = int(arr.size)
        n_samples = int(bs.sum())
        print(f"  sev={severity}  {corruption:<25s}"
              f"  RE={mean:.2f} ± {std:.2f}"
              f"  (iters={n_iters}, weighted N={n_samples})")
        return mean, std
    print(f"  sev={severity}  {corruption:<25s}  NO DATA")
    return None, None


# ------------------------------------------------------------------ main
def main():
    parser = argparse.ArgumentParser(
        description="Generate BS-weighted RE_markers CSV for the standard model."
    )
    parser.add_argument("--vis_lo", type=float, default=0.0,
                        help="Lower bound of visibility range, inclusive (default: 0.0)")
    parser.add_argument("--vis_hi", type=float, default=1.0,
                        help="Upper bound of visibility range, exclusive (default: 1.0)")
    args = parser.parse_args()

    print(f"\nVisibility range: [{args.vis_lo}, {args.vis_hi})")
    vis_mask = load_vis_mask(file_stats, args.vis_lo, args.vis_hi)
    print(f"Samples in range: {vis_mask.sum()} / {vis_mask.size}\n")

    # Collect results: table[sev] = list of (mean, std) per corruption
    table = {}
    for sev in severities:
        print(f"--- Severity {sev} ---")
        table[sev] = [standard_stats(c, sev, vis_mask) for c in corruptions]

    # ---- write CSV ----
    vis_tag  = f"vis_{args.vis_lo}_{args.vis_hi}".replace(".", "p")
    out_path = os.path.join(os.path.dirname(__file__), f"{metric}_{method}_{vis_tag}{extenstion}.csv")

    def fmt(mean, std=None):
        if mean is None:
            return "N/A"
        if std is None:
            return f"{mean:.2f}"
        return f"{mean:.2f} ± {std:.2f}"

    # col_acc[i] accumulates BS-weighted means across severities for the Mean row
    col_acc_means = [[] for _ in corruptions]

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Severity"] + col_names + ["Mean"])

        for sev in severities:
            stats = table[sev]                                   # [(mean, std), ...]
            cells = [fmt(mean, std) for mean, std in stats]
            valid_means = [mean for mean, _ in stats if mean is not None]
            row_mean = float(np.mean(valid_means)) if valid_means else None
            row_std  = float(np.std(valid_means))  if valid_means else None
            writer.writerow([f"Sev {sev}"] + cells + [fmt(row_mean, row_std)])
            for i, (mean, _) in enumerate(stats):
                if mean is not None:
                    col_acc_means[i].append(mean)

        # Mean row: mean ± std across the 5 severity means per corruption
        col_means = [float(np.mean(a)) if a else None for a in col_acc_means]
        col_stds  = [float(np.std(a))  if a else None for a in col_acc_means]
        grand_vals = [m for m in col_means if m is not None]
        grand_mean = float(np.mean(grand_vals)) if grand_vals else None
        grand_std  = float(np.std(grand_vals))  if grand_vals else None
        writer.writerow(["Mean"] + [fmt(m, s) for m, s in zip(col_means, col_stds)] + [fmt(grand_mean, grand_std)])

    print(f"\n✓ CSV saved to: {out_path}")


if __name__ == "__main__":
    main()