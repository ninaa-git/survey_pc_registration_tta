from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_SIZES = [0.15, 0.2, 0.25, 0.3]
DEFAULT_SEEDS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
DEFAULT_EPOCHS = 150  # the training logs go up to /150

# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

# Line 1 of every log starts with:
#   [...] [INFO] Command executed: ... --n_val_samples 100 --val_seed 20 ...
_CMD_RX = re.compile(r"--n_val_ratio\s+(\d+).*?--val_seed\s+(\d+)")

# Validation lines look like:
#   [...] [CRIT] [Val] Epoch: 7, loss: ..., RMSE: 0.045, RMSE_std: 0.053, RR: ...
_VAL_RX = re.compile(
    r"\[Val\]\s+Epoch:\s+(\d+),"
    r".*?RMSE:\s+([\d.eE+-]+),"
    r"\s+RMSE_std:\s+([\d.eE+-]+)"
)


def parse_log_header(log_path: str) -> Optional[Tuple[int, int]]:
    """Read the first line of `log_path` and return (n_val_samples, val_seed)."""
    try:
        with open(log_path, "r", errors="replace") as f:
            first_line = f.readline()
    except OSError:
        return None
    m = _CMD_RX.search(first_line)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def parse_val_curve(
    log_path: str, dup_strategy: str = "first"
) -> Dict[int, Tuple[float, float]]:
    """Walk the log and return {epoch: (rmse_mean, rmse_std)}.

    Each epoch has two ``[Val]`` lines (one per rank in distributed training).
    `dup_strategy` controls how they are reconciled:
        * first (default) -- take the first line per epoch (rank 0)
        * last            -- take the last line per epoch
        * mean            -- average the two lines
    """
    by_epoch: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
    with open(log_path, "r", errors="replace") as f:
        for line in f:
            m = _VAL_RX.search(line)
            if m is None:
                continue
            ep = int(m.group(1))
            rmse = float(m.group(2))
            rmse_std = float(m.group(3))
            by_epoch[ep].append((rmse, rmse_std))

    out: Dict[int, Tuple[float, float]] = {}
    for ep, vals in by_epoch.items():
        if dup_strategy == "first":
            out[ep] = vals[0]
        elif dup_strategy == "last":
            out[ep] = vals[-1]
        elif dup_strategy == "mean":
            arr = np.asarray(vals, dtype=np.float64)
            out[ep] = (float(arr[:, 0].mean()), float(arr[:, 1].mean()))
        else:
            raise ValueError(f"Unknown dup_strategy={dup_strategy!r}")
    return out


def discover_logs(log_dir: str) -> Dict[Tuple[int, int], List[str]]:
    """Scan log_dir/*.log and group by (n_val_samples, val_seed)."""
    print(log_dir)
    grouped: Dict[Tuple[int, int], List[str]] = defaultdict(list)
    pattern = os.path.join(log_dir, "*.log")
    for path in sorted(glob.glob(pattern)):
        hdr = parse_log_header(path)
        if hdr is None:
            continue
        grouped[hdr].append(path)
    return grouped


def pick_best_log(
    log_paths: List[str], expected_epochs: int, dup_strategy: str
) -> Tuple[Optional[str], Optional[Dict[int, Tuple[float, float]]]]:
    """Among several logs for the same (size, seed), pick the most complete.

    Preference:
        * a log that reaches `expected_epochs`
        * else the log with the most epochs
    """
    best_path = None
    best_curve: Optional[Dict[int, Tuple[float, float]]] = None
    best_n = -1
    for p in log_paths:
        try:
            curve = parse_val_curve(p, dup_strategy=dup_strategy)
        except OSError:
            continue
        n = len(curve)
        if n > best_n:
            best_path, best_curve, best_n = p, curve, n
    return best_path, best_curve


# ---------------------------------------------------------------------------
# Test JSON aggregation
# ---------------------------------------------------------------------------

def aggregate_test_json(path: str) -> Tuple[float, float, int, int]:
    """Aggregate per-batch RMSE/RMSE_std into one (mean, std).

    Each line of `path` is a JSON object with at least `BS`, `RMSE`, and
    `RMSE_std` (the per-batch mean and within-batch std). The last batch may
    have a smaller `BS`. The combined statistics are computed via the law of
    total variance:

        mean   = sum(BS_i * RMSE_i) / sum(BS_i)
        var    = sum(BS_i * RMSE_std_i^2) / N             # within-batch
               + sum(BS_i * (RMSE_i - mean)^2) / N        # between-batch
        std    = sqrt(var)

    Returns:
        (rmse_mean, rmse_std, n_iterations, n_total_samples)
    """
    bs_list: List[float] = []
    mu_list: List[float] = []
    sigma_list: List[float] = []
    with open(path, "r") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            d = json.loads(raw)
            bs_list.append(float(d["BS"]))
            mu_list.append(float(d["RMSE"]))
            sigma_list.append(float(d["RMSE_std"]))
    if not bs_list:
        raise ValueError(f"empty test file: {path}")

    bs = np.asarray(bs_list, dtype=np.float64)
    mu = np.asarray(mu_list, dtype=np.float64)
    sigma = np.asarray(sigma_list, dtype=np.float64)

    n_total = bs.sum()
    grand_mean = (bs * mu).sum() / n_total
    within = (bs * sigma ** 2).sum() / n_total
    between = (bs * (mu - grand_mean) ** 2).sum() / n_total
    grand_std = float(np.sqrt(within + between))
    return float(grand_mean), grand_std, len(bs), int(n_total)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _color_cycle(n: int):
    """Return a list of n distinct colors from a perceptually-uniform colormap."""
    cmap = plt.get_cmap("viridis")
    return [cmap(i / max(n - 1, 1)) for i in range(n)]


def plot_overlay(
    epochs: np.ndarray,
    means_by_size: Dict[int, np.ndarray],
    stds_by_size: Dict[int, np.ndarray],
    out_path: str,
) -> None:
    """All sizes on one plot, each with its +/- std band."""
    sizes = sorted(means_by_size)
    colors = _color_cycle(len(sizes))
    fig, ax = plt.subplots(figsize=(10, 6))
    for c, sz in zip(colors, sizes):
        m = means_by_size[sz]
        s = stds_by_size[sz]
        ax.plot(epochs, m, color=c, label=f"|val|={sz}", lw=1.6)
        ax.fill_between(epochs, m - s, m + s, color=c, alpha=0.18, linewidth=0)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation RMSE")
    ax.set_title("Validation RMSE per validset size (mean +/- std across seeds)")
    ax.legend(ncol=2, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_grid(
    epochs: np.ndarray,
    means_by_size: Dict[int, np.ndarray],
    stds_by_size: Dict[int, np.ndarray],
    out_path: str,
) -> None:
    """One subplot per size for cleaner band comparison."""
    sizes = sorted(means_by_size)
    n = len(sizes)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 3.4 * rows),
                             sharex=True, sharey=True, squeeze=False)
    colors = _color_cycle(n)
    # Use a common y-limit across panels for easier visual comparison.
    all_low = min((means_by_size[s] - stds_by_size[s]).min() for s in sizes)
    all_high = max((means_by_size[s] + stds_by_size[s]).max() for s in sizes)
    pad = 0.05 * (all_high - all_low + 1e-9)
    for i, sz in enumerate(sizes):
        ax = axes[i // cols, i % cols]
        m = means_by_size[sz]
        s = stds_by_size[sz]
        ax.plot(epochs, m, color=colors[i], lw=1.6)
        ax.fill_between(epochs, m - s, m + s, color=colors[i], alpha=0.25,
                        linewidth=0)
        ax.set_title(f"|val|={sz}  (mean band-width={s.mean():.4g})")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation RMSE")
        ax.set_ylim(all_low - pad, all_high + pad)
        ax.grid(True, alpha=0.3)
    # hide any unused panels
    for j in range(n, rows * cols):
        axes[j // cols, j % cols].axis("off")
    fig.suptitle("Validation curves per size (mean +/- std across seeds)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_best_epoch_distribution(
    sizes: List[int],
    best_epochs_by_size: Dict[int, List[int]],
    expected_epochs: int,
    out_path: str,
) -> None:
    """Box + scatter plot of the chosen best epoch per validset size.

    Each dot is one seed. The box shows median / IQR across seeds.
    A narrow spread means the model-selection epoch is stable; a wide spread
    means it is sensitive to the random seed (i.e. the validset is too small
    to reliably locate the best checkpoint).
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = _color_cycle(len(sizes))

    data_for_box = [best_epochs_by_size.get(sz, []) for sz in sizes]
    # Only draw boxplot for sizes that have data
    nonempty = [(i, d) for i, d in enumerate(data_for_box) if len(d) > 1]
    if nonempty:
        positions = [i + 1 for i, _ in nonempty]
        bp = ax.boxplot(
            [d for _, d in nonempty],
            positions=positions,
            widths=0.45,
            patch_artist=True,
            medianprops=dict(color="black", lw=2),
            whiskerprops=dict(lw=1.2),
            capprops=dict(lw=1.2),
            flierprops=dict(marker=""),
        )
        for patch, (i, _) in zip(bp["boxes"], nonempty):
            patch.set_facecolor(colors[i])
            patch.set_alpha(0.4)

    # Overlay individual seed dots with horizontal jitter
    rng = np.random.default_rng(0)
    for i, sz in enumerate(sizes):
        epochs_for_sz = best_epochs_by_size.get(sz, [])
        if not epochs_for_sz:
            continue
        jitter = rng.uniform(-0.18, 0.18, size=len(epochs_for_sz))
        ax.scatter(
            np.full(len(epochs_for_sz), i + 1) + jitter,
            epochs_for_sz,
            color=colors[i], s=55, zorder=3, edgecolors="black", linewidths=0.5,
            label=f"|val|={sz}  "
                  f"(μ={np.mean(epochs_for_sz):.0f}, "
                  f"σ={np.std(epochs_for_sz, ddof=1) if len(epochs_for_sz) > 1 else 0:.1f})"
        )

    ax.set_xticks(range(1, len(sizes) + 1))
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_xlabel("Validset size")
    ax.set_ylabel("Best epoch selected")
    ax.set_ylim(0, expected_epochs + 5)
    ax.set_title("Distribution of best-model epoch across seeds\n"
                 "(narrow spread = stable model selection)")
    ax.legend(ncol=2, loc="best", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_test_bars(
    sizes: List[int],
    means: np.ndarray,
    stds: np.ndarray,
    out_path: str,
) -> None:
    """Bar plot of test RMSE mean +/- std per size."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    xs = np.arange(len(sizes))
    ax.bar(xs, means, yerr=stds, capsize=6, color="#3a7ca5",
           edgecolor="black", alpha=0.85)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_xlabel("Validation-set size")
    ax.set_ylabel("Test RMSE")
    ax.set_title("Test RMSE of best-by-val model: mean +/- std across seeds")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def write_csv(path: str, header: List[str], rows: List[List]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> int:
    base = Path(args.base).resolve()
    log_dir = base / "output" / "P2PSilico" / "Source_Only" / "logs"
    test_dir = base / "output" / "P2PSilico" / "Source_Only" / "ft_validation"
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not log_dir.is_dir():
        print(f"[ERROR] log dir not found: {log_dir}", file=sys.stderr)
        return 2
    if not test_dir.is_dir():
        print(f"[ERROR] test dir not found: {test_dir}", file=sys.stderr)
        return 2

    sizes = args.sizes
    seeds = args.seeds
    expected_epochs = args.epochs
    start_epoch = max(1, args.start_epoch)
    epochs_axis = np.arange(1, expected_epochs + 1, dtype=int)

    # ---- index logs by (size, seed) ----------------------------------------
    print(f"[info] indexing logs in {log_dir}")
    grouped = discover_logs(str(log_dir))
    print(f"[info] found {sum(len(v) for v in grouped.values())} logs "
          f"in {len(grouped)} (size,seed) groups")

    # ---- per-(size,seed) val curve & test stats ---------------------------
    # val_rmse[size][seed] = np.ndarray of length expected_epochs (NaN if missing)
    val_rmse: Dict[int, Dict[int, np.ndarray]] = {sz: {} for sz in sizes}
    val_rmse_std: Dict[int, Dict[int, np.ndarray]] = {sz: {} for sz in sizes}
    # test_rmse[size][seed] = float
    test_rmse: Dict[int, Dict[int, float]] = {sz: {} for sz in sizes}
    test_rmse_std: Dict[int, Dict[int, float]] = {sz: {} for sz in sizes}
    test_iters: Dict[int, Dict[int, int]] = {sz: {} for sz in sizes}
    # best_epoch[size][seed] = int  (1-based epoch of the last minimum val RMSE)
    best_epoch: Dict[int, Dict[int, int]] = {sz: {} for sz in sizes}
    best_epoch_rmse: Dict[int, Dict[int, float]] = {sz: {} for sz in sizes}

    skipped_runs: List[str] = []

    for sz in sizes:
        for sd in seeds:
            key = (sz, sd)

            # --- training log -> val curve --------------------------------
            log_paths = grouped.get(key, [])
            if not log_paths:
                skipped_runs.append(
                    f"  (size={sz}, seed={sd}): no log found")
                continue
            best_path, curve = pick_best_log(
                log_paths, expected_epochs, dup_strategy=args.dup_strategy)
            if curve is None or len(curve) == 0:
                skipped_runs.append(
                    f"  (size={sz}, seed={sd}): empty/unparseable log "
                    f"({best_path})")
                continue
            n_eps = len(curve)
            if n_eps < expected_epochs:
                skipped_runs.append(
                    f"  (size={sz}, seed={sd}): only {n_eps}/"
                    f"{expected_epochs} epochs in {best_path} (kept anyway)")
            arr_m = np.full(expected_epochs, np.nan, dtype=np.float64)
            arr_s = np.full(expected_epochs, np.nan, dtype=np.float64)
            for ep, (m, s) in curve.items():
                if 1 <= ep <= expected_epochs:
                    arr_m[ep - 1] = m
                    arr_s[ep - 1] = s
            # Mask out epochs before start_epoch so ALL calculations
            # (best epoch, mean/std curves, band width) only use the
            # epochs the user cares about.
            if start_epoch > 1:
                arr_m[:start_epoch - 1] = np.nan
                arr_s[:start_epoch - 1] = np.nan
            val_rmse[sz][sd] = arr_m
            val_rmse_std[sz][sd] = arr_s

            # Best epoch = last epoch achieving the minimum val RMSE
            # (mirrors a training loop that saves whenever val improves)
            valid_mask = ~np.isnan(arr_m)
            if valid_mask.any():
                min_val = np.nanmin(arr_m)
                # last index where RMSE equals the minimum
                last_min_idx = int(np.where(arr_m == min_val)[0][-1])
                best_epoch[sz][sd] = last_min_idx + 1          # 1-based
                best_epoch_rmse[sz][sd] = float(min_val)

            # --- test JSON -> aggregated mean/std -------------------------
            test_path = test_dir / f"test_{sz}_{sd}.json"
            if not test_path.is_file():
                skipped_runs.append(
                    f"  (size={sz}, seed={sd}): missing {test_path.name}")
                continue
            try:
                m_t, s_t, n_iter, _ = aggregate_test_json(str(test_path))
            except Exception as e:  # noqa: BLE001 -- robust scan
                skipped_runs.append(
                    f"  (size={sz}, seed={sd}): error reading "
                    f"{test_path.name}: {e}")
                continue
            test_rmse[sz][sd] = m_t
            test_rmse_std[sz][sd] = s_t
            test_iters[sz][sd] = n_iter

    # ---- consistency check on test iteration counts -----------------------
    all_iters = [n for d in test_iters.values() for n in d.values()]
    if all_iters:
        unique_iters = sorted(set(all_iters))
        if len(unique_iters) > 1:
            print(f"[WARN] test files have different iteration counts: "
                  f"{unique_iters} -- they may not be evaluating the same "
                  f"test set", file=sys.stderr)
        else:
            print(f"[info] all test files have {unique_iters[0]} iterations")

    if skipped_runs:
        print("[WARN] issues found in some runs:", file=sys.stderr)
        for line in skipped_runs:
            print(line, file=sys.stderr)

    # ---- best-by-val test selection ---------------------------------------
    # The user's protocol: select the best model VIA the validset, and measure
    # that model's performance on the testset. The test JSON is already that
    # measurement (it is run on the model picked at the end of training, which
    # in turn keeps the best checkpoint by val). So `test_rmse[sz][sd]` is the
    # quantity we want to summarise across seeds.

    test_summary_rows: List[List] = []
    test_means_by_size: List[float] = []
    test_stds_by_size: List[float] = []
    valid_sizes_for_test: List[int] = []
    for sz in sizes:
        per_seed = [test_rmse[sz][sd] for sd in seeds if sd in test_rmse[sz]]
        per_seed_std_within = [test_rmse_std[sz][sd]
                               for sd in seeds if sd in test_rmse_std[sz]]
        if not per_seed:
            test_summary_rows.append([sz, 0, "", "", ""]
                                     + [""] * len(seeds))
            continue
        arr = np.asarray(per_seed, dtype=np.float64)
        mu = float(arr.mean())
        sd_ = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
        within_avg = float(np.mean(per_seed_std_within))
        # One column per seed — empty string if that seed is missing
        per_seed_cols = [
            f"{test_rmse[sz][sd]:.6g}" if sd in test_rmse[sz] else ""
            for sd in seeds
        ]
        test_summary_rows.append([sz, len(per_seed),
                                  f"{mu:.6g}", f"{sd_:.6g}",
                                  f"{within_avg:.6g}"]
                                 + per_seed_cols)
        test_means_by_size.append(mu)
        test_stds_by_size.append(sd_)
        valid_sizes_for_test.append(sz)

    per_seed_headers = [f"rmse_seed_{sd}" for sd in seeds]
    write_csv(str(out_dir / "test_summary.csv"),
              ["valid_size", "n_seeds_used", "test_rmse_mean_across_seeds",
               "test_rmse_std_across_seeds", "avg_within_test_rmse_std"]
              + per_seed_headers,
              test_summary_rows)

    if valid_sizes_for_test:
        plot_test_bars(valid_sizes_for_test,
                       np.asarray(test_means_by_size),
                       np.asarray(test_stds_by_size),
                       str(out_dir / "test_means.png"))

    # ---- best epoch analysis -----------------------------------------------
    # For each (size, seed): which epoch was selected as best checkpoint?
    # Stability of this choice across seeds reveals whether the validset is
    # large enough to consistently locate the optimal stopping point.
    best_epoch_per_seed_rows: List[List] = []
    best_epoch_summary_rows: List[List] = []
    best_epochs_by_size: Dict[int, List[int]] = {}

    for sz in sizes:
        epochs_for_sz = []
        for sd in seeds:
            if sd not in best_epoch[sz]:
                continue
            ep = best_epoch[sz][sd]
            rmse = best_epoch_rmse[sz][sd]
            best_epoch_per_seed_rows.append([sz, sd, ep, f"{rmse:.6g}"])
            epochs_for_sz.append(ep)
        best_epochs_by_size[sz] = epochs_for_sz
        if epochs_for_sz:
            arr = np.asarray(epochs_for_sz, dtype=np.float64)
            mu_ep = float(arr.mean())
            std_ep = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
            best_epoch_summary_rows.append(
                [sz, len(epochs_for_sz), f"{mu_ep:.1f}", f"{std_ep:.2f}",
                 int(arr.min()), int(arr.max())])
        else:
            best_epoch_summary_rows.append([sz, 0, "", "", "", ""])

    write_csv(str(out_dir / "best_epoch_per_seed.csv"),
              ["valid_size", "seed", "best_epoch", "best_val_rmse"],
              best_epoch_per_seed_rows)
    write_csv(str(out_dir / "best_epoch_summary.csv"),
              ["valid_size", "n_seeds", "mean_best_epoch", "std_best_epoch",
               "min_best_epoch", "max_best_epoch"],
              best_epoch_summary_rows)
    plot_best_epoch_distribution(
        sizes, best_epochs_by_size, expected_epochs,
        str(out_dir / "best_epoch_distribution.png"))

    # Print console summary
    print("\n=== Best-epoch selection stability across seeds ===")
    print(f"{'size':>6} {'n':>3} {'mean epoch':>12} {'std epoch':>10} "
          f"{'min':>5} {'max':>5}   (narrow std = stable selection)")
    for sz, n, mu_s, std_s, mn, mx in best_epoch_summary_rows:
        if n == 0:
            print(f"{sz:>6} {n:>3}   (no data)")
            continue
        print(f"{sz:>6} {n:>3} {mu_s:>12} {std_s:>10} {mn:>5} {mx:>5}")

    # ---- val-curve aggregation across seeds -------------------------------
    means_by_size: Dict[int, np.ndarray] = {}
    stds_by_size: Dict[int, np.ndarray] = {}
    band_summary_rows: List[List] = []
    val_curves_rows: List[List] = []
    val_curves_per_seed_rows: List[List] = []

    for sz in sizes:
        seed_curves      = [val_rmse[sz][sd]     for sd in seeds if sd in val_rmse[sz]]
        seed_curves_std  = [val_rmse_std[sz][sd] for sd in seeds if sd in val_rmse_std[sz]]
        if not seed_curves:
            band_summary_rows.append([sz, 0, "", ""])
            continue
        stack     = np.vstack(seed_curves)      # (n_seeds, expected_epochs)
        stack_std = np.vstack(seed_curves_std)  # (n_seeds, expected_epochs)
        with np.errstate(invalid="ignore"):
            ep_mean = np.nanmean(stack, axis=0)
            # Cross-seed std: how much the mean RMSE varies between seeds
            # at each epoch → measures stability of the validation signal.
            ep_std_across = np.nanstd(stack, axis=0, ddof=1) if stack.shape[0] > 1 \
                else np.zeros(stack.shape[1])
            # Within-run std: the RMSE_std reported inside each log line
            # (spread within a single evaluation run).
            ep_std_within = np.nanmean(stack_std, axis=0)
        means_by_size[sz] = ep_mean
        stds_by_size[sz]  = ep_std_across  # used for the fill_between band

        # Scalar summaries: mean over epochs (ignoring NaN / masked epochs)
        avg_std_across = float(np.nanmean(ep_std_across))
        avg_std_within = float(np.nanmean(ep_std_within))
        band_summary_rows.append([sz, stack.shape[0],
                                   f"{avg_std_across:.6g}",
                                   f"{avg_std_within:.6g}"])

        for i, ep in enumerate(epochs_axis):
            if ep < start_epoch:
                continue
            val_curves_rows.append(
                [sz, int(ep),
                 "" if np.isnan(ep_mean[i]) else f"{ep_mean[i]:.6g}",
                 "" if np.isnan(ep_std_across[i]) else f"{ep_std_across[i]:.6g}"])
        for sd in seeds:
            if sd not in val_rmse[sz]:
                continue
            arr = val_rmse[sz][sd]
            for i, ep in enumerate(epochs_axis):
                if ep < start_epoch:
                    continue
                val_curves_per_seed_rows.append(
                    [sz, sd, int(ep),
                     "" if np.isnan(arr[i]) else f"{arr[i]:.6g}"])

    write_csv(str(out_dir / "val_band_summary.csv"),
              ["valid_size", "n_seeds_used",
               "mean_per_epoch_std_across_seeds",   # cross-seed: stability indicator
               "mean_per_epoch_std_within_run"],     # within-run: from log RMSE_std
              band_summary_rows)
    write_csv(str(out_dir / "val_curves.csv"),
              ["valid_size", "epoch", "val_rmse_mean", "val_rmse_std"],
              val_curves_rows)
    write_csv(str(out_dir / "val_curves_per_seed.csv"),
              ["valid_size", "seed", "epoch", "val_rmse"],
              val_curves_per_seed_rows)

    # ---- plot val curves --------------------------------------------------
    if means_by_size:
        # epochs_axis is already NaN-masked before start_epoch in the arrays;
        # slice the axis to match so the x-axis starts at start_epoch.
        ep_mask = epochs_axis >= start_epoch
        epochs_plot = epochs_axis[ep_mask]
        means_plot  = {sz: arr[ep_mask] for sz, arr in means_by_size.items()}
        stds_plot   = {sz: arr[ep_mask] for sz, arr in stds_by_size.items()}

        plot_overlay(epochs_plot, means_plot, stds_plot,
                     str(out_dir / "val_curves.png"))
        plot_grid(epochs_plot, means_plot, stds_plot,
                  str(out_dir / "val_curves_grid.png"))

    # ---- console summary --------------------------------------------------
    print("\n=== Test RMSE across seeds (best-by-val) ===")
    print(f"{'size':>6} {'n':>3} {'mean':>14} {'std (cross-seed)':>18}  "
          f"{'avg within-test std':>20}")
    for row in test_summary_rows:
        sz, n_used, mean_s, std_s, within_s = row[0], row[1], row[2], row[3], row[4]
        if n_used == 0:
            print(f"{sz:>6} {n_used:>3}   (no data)")
            continue
        print(f"{sz:>6} {n_used:>3} {mean_s:>14} {std_s:>18}  {within_s:>20}")

    print("\n=== Validation curve band width per size ===")
    print(f"{'size':>6} {'n':>3} {'std across seeds':>20} {'std within run':>18}"
          f"   (cross-seed: stability | within-run: eval noise)")
    for sz, n_used, std_across, std_within in band_summary_rows:
        if n_used == 0:
            print(f"{sz:>6} {n_used:>3}   (no data)")
            continue
        print(f"{sz:>6} {n_used:>3} {std_across:>20} {std_within:>18}")

    # ---- best-size suggestion ---------------------------------------------
    if valid_sizes_for_test:
        best_test_idx = int(np.argmin(test_stds_by_size))
        print(f"\n[suggest] smallest cross-seed test std: |val|="
              f"{valid_sizes_for_test[best_test_idx]} "
              f"(std={test_stds_by_size[best_test_idx]:.6g})")
    band_pairs = [(sz, float(b)) for sz, n, b, _ in band_summary_rows
                  if isinstance(b, str) and b]
    if band_pairs:
        best_band_sz, best_band = min(band_pairs, key=lambda x: x[1])
        print(f"[suggest] smallest mean per-epoch val std: |val|="
              f"{best_band_sz} (mean per-epoch std={best_band:.6g})")

    print(f"\n[done] wrote outputs to {out_dir}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyse the impact of validation-set size on model "
                    "selection.")
    parser.add_argument(
        "--base", required=True,
        help="Project root containing output/P2PSilico/Source_Only/logs/ "
             "and output/P2PSilico/ft_validation/.")
    parser.add_argument(
        "--out", default="./analysis",
        help="Output directory for CSV/PNG (created if missing).")
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=DEFAULT_SIZES,
        help=f"Validation-set sizes to consider (default: {DEFAULT_SIZES}).")
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
        help=f"Seeds to consider (default: {DEFAULT_SEEDS}).")
    parser.add_argument(
        "--epochs", type=int, default=DEFAULT_EPOCHS,
        help=f"Expected number of training epochs (default: {DEFAULT_EPOCHS}).")
    parser.add_argument(
        "--start_epoch", type=int, default=1,
        help="First epoch to include in plots and CSV output (default: 1). "
             "E.g. --start_epoch 20 skips the first 19 epochs in all plots.")
    parser.add_argument(
        "--dup-strategy", choices=("first", "last", "mean"), default="first",
        help="Each epoch logs two [Val] lines (one per distributed rank); "
             "how to reconcile them. Default: first (rank 0).")
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())