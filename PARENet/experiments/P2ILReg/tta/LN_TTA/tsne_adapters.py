import os.path as osp
import torch

from config import make_cfg
from common_utils.parser import make_parser
from pareconv.utils.torch import to_cuda

import tsne_core as core
from loss import Evaluator


def _tag(args, cfg):
    return f"{cfg.dataset_name}_{args.corruption}_{args.severity}"


def run(args, cfg):
    from model import create_model
    from dataset import viz_data_loaders
    from ln_tta import LN_TTA_Manager
    from generate_stats import collect_source_ln_stats
    from pareconv.modules.transformer.rpe_transformer_LN import AdaptiveLayerNorm, TTAContext

    n_clouds   = args.viz_max_clouds
    sample_idx = args.viz_sample
    tag        = _tag(args, cfg)
    core.print_viz_window(sample_idx, n_clouds)
    cfg.test.batch_size  = 1
    cfg.test.num_workers = 0

    ctx   = TTAContext()
    model = create_model(cfg).cuda()
    model.transformer.transformer.ctx = ctx
    for layer in model.transformer.transformer.layers:
        if hasattr(layer, "attention") and hasattr(layer.attention, "norm"):
            if hasattr(layer.attention.norm, "ctx"):
                layer.attention.norm.ctx = ctx
            if hasattr(layer.output, "norm") and hasattr(layer.output.norm, "ctx"):
                layer.output.norm.ctx = ctx
    if getattr(args, "snapshot", None):
        core.load_checkpoint(model, args.snapshot)

    sel      = lambda name, m: type(m) is AdaptiveLayerNorm
    root     = model.transformer.transformer
    # forward_fn: hooks live on `root` (inner transformer), but the entry
    # point for a full forward pass is the complete PARE_Net `model`.
    fwd      = lambda data: model(data)

    def clear():
        for _, m in root.named_modules():
            if type(m) is AdaptiveLayerNorm:
                m.mu_source_ref = m.sigma2_source_ref = None
                m.mu_source_src = m.sigma2_source_src = None
                m.mu_target_ref = m.m2_target_ref = m.sigma2_target_ref = None
                m.mu_target_src = m.m2_target_src = m.sigma2_target_src = None
                m.n_target_ref  = m.n_target_src = 0.0

    clean_loader, cor_loader, cfg.backbone.num_neighbors = viz_data_loaders(args, cfg)

    clear()
    F_clean = core.collect_with_hook(root, clean_loader, sel, to_cuda, n_clouds,
                                     sample_idx=sample_idx, forward_fn=fwd)
    clear()
    F_cor   = core.collect_with_hook(root, cor_loader,   sel, to_cuda, n_clouds,
                                     sample_idx=sample_idx, forward_fn=fwd)
    clear()

    # ── No-adapt RMSE — before any LN_TTA_Manager is registered ──────────
    evaluator = Evaluator(cfg).cuda()

    @torch.no_grad()
    def _rmse_1(ldr):
        return core.collect_rmse_metrics(
            fwd, lambda out, data: evaluator(out, data), ldr, to_cuda, 1,
            sample_idx=sample_idx,
        )

    rmse_no_adapt = _rmse_1(cor_loader)

    # ── Adapted feature collection (registers LN_TTA_Manager hooks) ──────
    stats = collect_source_ln_stats(
        model, cfg, args,
        osp.join("intermediate_features", "ln_stats", "ln_source_stats.pth"),
        force_regenerate=False,
    )
    LN_TTA_Manager(
        model.transformer, stats, ctx, "oracle",
        getattr(args, "layers_to_adapt", None),
        getattr(args, "ln_momentum", 0.01),
    )
    F_adapt = core.collect_with_hook(root, cor_loader, sel, to_cuda, n_clouds,
                                     sample_idx=sample_idx, forward_fn=fwd)

    # ── Adapted RMSE — manager warmed up, 1 pair ──────────────────────────
    rmse_adapted = _rmse_1(cor_loader)

    core.plot_groups(F_clean, F_cor, F_adapt, tag, "LN_TTA", out_dir=cfg.viz_dir,
                     rmse={"no_adapt": rmse_no_adapt, "adapted": rmse_adapted})


def main():
    args = make_parser().parse_args()
    cfg  = make_cfg()
    torch.set_float32_matmul_precision("high")
    run(args, cfg)


if __name__ == "__main__":
    main()
