import os
import os.path as osp
import time

import numpy as np
import torch

from config import make_cfg
from dataset import train_valid_data_loader
from tta.PEA_TTA.model import create_model
from pareconv.engine.logger import Logger
from pareconv.utils.torch import to_cuda
from pareconv.utils.data import precompute_neibors
from common_utils.parser import make_parser


# Stages in backbone order: coarse → fine
STAGE_KEYS = [
    'ref_feats_c',   # coarse RI head (post-VNStdFeature)
    'ref_feats_f',   # fine   RI headz
]

STAGE_LABELS = {
    'ref_feats_c': 'RI-head-c',
    'ref_feats_f': 'RI-head-f',
}


# ─────────────────────────────────────────────────────────────────────────────
def _collect_stats_online(model, cfg, loader, stage_keys, n_batches=None):
    """
    Compute per-stage distribution statistics in a single pass over the data,
    accumulating sufficient statistics (sum, sum-of-squares, outer products)
    without storing all raw features.

    Returns:
        dict  stage_key -> {'mean': [C], 'var': [C], 'cov': [C,C],
                            'cov_sqrt': [C,C], 'cov_invsqrt': [C,C]}
    """
    model.eval()

    # Accumulators: sum, sum-of-squares, outer-product sum, count
    acc = {k: {'s1': None, 's2': None, 'outer': None, 'n': 0}
           for k in stage_keys}

    total = n_batches if n_batches is not None else len(loader)
    from tqdm import tqdm
    for batch_idx, data_dict in tqdm(enumerate(loader), total=total,
                                     desc="  collecting"):
        if n_batches is not None and batch_idx >= n_batches:
            break

        data_dict = to_cuda(data_dict)

        with torch.no_grad():
            out = model.forward_out_intermediate(data_dict)

        for key in stage_keys:
            if key not in out:
                continue
            feats = out[key]          # [N, C]  — still on GPU
            N = feats.shape[0]

            s1    = feats.sum(dim=0)                    # [C]
            s2    = (feats ** 2).sum(dim=0)             # [C]
            outer = feats.T @ feats                     # [C, C]

            a = acc[key]
            if a['s1'] is None:
                a['s1']    = s1
                a['s2']    = s2
                a['outer'] = outer
            else:
                a['s1']    += s1
                a['s2']    += s2
                a['outer'] += outer
            a['n'] += N

    # Finalise statistics
    results = {}
    for key in stage_keys:
        a = acc[key]
        if a['s1'] is None:
            print(f"  Warning: no data collected for {key}.")
            continue

        n    = a['n']
        mean = (a['s1'] / n).cpu().numpy()                  # [C]
        var  = (a['s2'] / n - (a['s1'] / n) ** 2           # E[x²] - E[x]²
                ).cpu().numpy()
        var  = np.maximum(var, 0.0)                         # numerical safety

        # Cov = (ΣxxT)/n - μμT
        mean_t = torch.from_numpy(mean).to(a['outer'].device)
        cov_t  = a['outer'] / n - torch.outer(mean_t, mean_t)
        cov    = cov_t.cpu().numpy()                        # [C, C]

        # Symmetrise to suppress floating-point asymmetry
        cov = (cov + cov.T) / 2.0

        # Eigendecomposition  Σ = V Λ Vᵀ
        eigvals, eigvecs = np.linalg.eigh(cov)
        eps = 1e-6
        eigvals_pos = np.maximum(eigvals, eps)

        # Σ^{1/2}  = V Λ^{1/2} Vᵀ
        cov_sqrt    = eigvecs * np.sqrt(eigvals_pos)[None, :] @ eigvecs.T
        # Σ^{-1/2} = V Λ^{-1/2} Vᵀ
        cov_invsqrt = eigvecs * (1.0 / np.sqrt(eigvals_pos))[None, :] @ eigvecs.T

        results[key] = {
            'mean':        mean,
            'var':         var,
            'cov':         cov,
            'cov_sqrt':    cov_sqrt,
            'cov_invsqrt': cov_invsqrt,
        }

    return results


# ─────────────────────────────────────────────────────────────────────────────
def runner(args, config):
    log_file = osp.join(config.log_dir,
                        'pea_stats-{}.log'.format(time.strftime('%Y%m%d-%H%M%S')))
    logger = Logger(log_file=log_file)

    # ── Model ────────────────────────────────────────────────────────────────
    model = create_model(config).cuda()
    checkpoint = torch.load(args.snapshot)
    load_result = model.load_state_dict(checkpoint['model'], strict=False)
    if load_result.missing_keys:
        logger.info(f"Missing keys: {load_result.missing_keys}")
    model.eval()
    logger.info(f"Loaded checkpoint: {args.snapshot}")

    # ── Source loader ────────────────────────────────────────────────────────
    logger.info("Loading source data …")
    src_loader, _, cfg_neighbors = train_valid_data_loader(config, False)
    config.backbone.num_neighbors = cfg_neighbors

    # ── Collect SOURCE statistics only ───────────────────────────────────────
    n_batches = getattr(args, 'n_batches', None)   # None = full dataset
    force     = getattr(args, 'force', False)

    src_cache = "intermediate_features/pea_stats/source_feats.pth"

    if os.path.exists(src_cache) and not force:
        logger.info(f"Loading cached source stats: {src_cache}")
        results = torch.load(src_cache, weights_only=False)
    else:
        logger.info("Collecting source statistics …")
        src_stats_dict = _collect_stats_online(
            model, config, src_loader, STAGE_KEYS, n_batches
        )
        # Save in the format expected by test.py:
        #   { stage_key: { 'src': stats_dict } }
        results = {
            key: {'src': src_stats_dict[key]}
            for key in STAGE_KEYS
            if key in src_stats_dict
        }
        os.makedirs(osp.dirname(src_cache), exist_ok=True)
        torch.save(results, src_cache)
        logger.info(f"Source stats cached: {src_cache}")

    # ── Log a summary ────────────────────────────────────────────────────────
    header = f"{'Stage':<22}  {'‖μ_src‖':>12}  {'mean(var)':>12}"
    logger.info("=" * len(header))
    logger.info(header)
    logger.info("-" * len(header))
    for key in STAGE_KEYS:
        if key not in results:
            continue
        s = results[key]['src']
        label = STAGE_LABELS.get(key, key)
        logger.info(
            f"{label:<22}  "
            f"{np.linalg.norm(s['mean']):>12.4f}  "
            f"{float(s['var'].mean()):>12.6f}"
        )
    logger.info("=" * len(header))

# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = make_parser().parse_args()
    cfg  = make_cfg()
    runner(args, cfg)