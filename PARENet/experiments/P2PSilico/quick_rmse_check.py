"""quick_rmse_check.py — compare Source_Only vs LN_TTA RMSE on one corruption.

Run from silico/PARENet/experiments/P2PSilico/:

    python quick_rmse_check.py --corruption uniform --severity 3 \
        --snapshot ../../ckpts/P2PSilico/best.pth.tar \
        --ln_momentum 0.03 --n_pairs 50

Prints a side-by-side summary so you can quickly judge whether LN_TTA helps.
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))           # PARENet/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../"))              # experiments/P2PSilico/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tta/LN_TTA/"))     # LN_TTA/
sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                "../../../../common"))                           # silico/common/

import numpy as np
import torch

from config import make_cfg
from model import create_model
from loss import Evaluator
from dataset import corrupted_test_data_loader
from ln_tta import LN_TTA_Manager
from generate_stats import collect_source_ln_stats
from pareconv.utils.torch import to_cuda
from pareconv.modules.transformer.rpe_transformer_LN import AdaptiveLayerNorm, TTAContext


def _clear_ln(root):
    for _, m in root.named_modules():
        if type(m) is AdaptiveLayerNorm:
            m.mu_source_ref = m.sigma2_source_ref = None
            m.mu_source_src = m.sigma2_source_src = None
            m.mu_target_ref = m.m2_target_ref = m.sigma2_target_ref = None
            m.mu_target_src = m.m2_target_src = m.sigma2_target_src = None
            m.n_target_ref  = m.n_target_src = 0.0


def _evaluate(model, loader, evaluator, n_pairs, label, with_tta, manager=None):
    metrics = {"RMSE": [], "RRE": [], "RTE": [], "IR": []}
    model.eval()

    with torch.no_grad():
        for i, data in enumerate(loader):
            if len(metrics["RMSE"]) >= n_pairs:
                break
            data = to_cuda(data)
            try:
                if with_tta and manager is not None:
                    out = manager.adapt(model, data)
                else:
                    out = model(data)
                res = evaluator(out, data)
                for k in metrics:
                    metrics[k].append(float(res[k]))
            except Exception as e:
                print(f"  [skip] pair {i}: {e}")

    n = len(metrics["RMSE"])
    print(f"\n{'─'*55}")
    print(f"  {label}   (n={n} pairs)")
    print(f"{'─'*55}")
    for k, vals in metrics.items():
        arr = np.array(vals)
        print(f"  {k:<6}  mean={arr.mean():.4f}   median={np.median(arr):.4f}")
    print(f"{'─'*55}")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone",    default="PARENet")
    parser.add_argument("--dataset",     default="P2PSilico")
    parser.add_argument("--method",      default="LN_TTA")
    parser.add_argument("--corruption",  required=True)
    parser.add_argument("--severity",    type=int, required=True)
    parser.add_argument("--snapshot",    default=None)
    parser.add_argument("--ln_momentum", type=float, default=0.03)
    parser.add_argument("--n_pairs",     type=int, default=50,
                        help="Number of test pairs per condition (default 50)")
    args = parser.parse_args()

    cfg = make_cfg()
    cfg.test.batch_size  = 1
    cfg.test.num_workers = 0
    torch.set_float32_matmul_precision("high")

    loader, cfg.backbone.num_neighbors = corrupted_test_data_loader(args, cfg)
    evaluator = Evaluator(cfg).cuda()

    # ── build model ──────────────────────────────────────────────────────────
    ctx = TTAContext()
    model = create_model(cfg).cuda()
    model.transformer.transformer.ctx = ctx
    for layer in model.transformer.transformer.layers:
        if hasattr(layer, "attention") and hasattr(layer.attention, "norm"):
            if hasattr(layer.attention.norm, "ctx"):
                layer.attention.norm.ctx = ctx
            if hasattr(layer.output, "norm") and hasattr(layer.output.norm, "ctx"):
                layer.output.norm.ctx = ctx

    if args.snapshot:
        import torch as _torch
        ckpt = _torch.load(args.snapshot, map_location="cpu", weights_only=False)
        state = ckpt["model"] if "model" in ckpt else ckpt
        state = {k.replace("._orig_mod", ""): v for k, v in state.items()}
        model.load_state_dict(state, strict=True)
        print(f"[quick_check] Loaded checkpoint: {args.snapshot}")

    # ── Source-Only (no adaptation) ───────────────────────────────────────────
    _clear_ln(model.transformer.transformer)
    src_metrics = _evaluate(model, loader, evaluator, args.n_pairs,
                            label=f"Source_Only  [{args.corruption}_{args.severity}]",
                            with_tta=False)

    # ── LN_TTA ───────────────────────────────────────────────────────────────
    src_stats_path = os.path.join("intermediate_features", "ln_stats",
                                  "ln_source_stats.pth")
    source_stats = collect_source_ln_stats(
        model, cfg, args, src_stats_path, force_regenerate=False
    )
    _clear_ln(model.transformer.transformer)
    manager = LN_TTA_Manager(
        model.transformer, source_stats, ctx, "oracle",
        layers_to_adapt=None,
        momentum=args.ln_momentum,
    )
    ln_metrics = _evaluate(model, loader, evaluator, args.n_pairs,
                           label=f"LN_TTA       [{args.corruption}_{args.severity}]",
                           with_tta=True, manager=manager)

    # ── Delta ─────────────────────────────────────────────────────────────────
    print("\n  Δ (LN_TTA − Source_Only)  negative = improvement")
    for k in ("RMSE", "RRE", "RTE", "IR"):
        src_m = np.mean(src_metrics[k])
        ln_m  = np.mean(ln_metrics[k])
        sign  = "✓" if (ln_m < src_m) else "✗"
        print(f"  {k:<6}  {src_m:.4f} → {ln_m:.4f}   Δ={ln_m-src_m:+.4f}  {sign}")


if __name__ == "__main__":
    main()
