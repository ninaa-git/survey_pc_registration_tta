import os.path as osp
import torch

from config import make_cfg
from common_utils.parser import make_parser
from pareconv.utils.torch import to_cuda

import tsne_core as core


def _tag(args, cfg):
    return f"{cfg.dataset_name}_{args.corruption}_{args.severity}"


def run(args, cfg):
    from tta.PEA_TTA.model import create_model
    from dataset import viz_data_loaders

    n_clouds   = args.viz_max_clouds
    sample_idx = args.viz_sample
    tag        = _tag(args, cfg)
    core.print_viz_window(sample_idx, n_clouds)
    cfg.test.batch_size  = 1
    cfg.test.num_workers = 0

    model = create_model(cfg).cuda()
    if getattr(args, "snapshot", None):
        core.load_checkpoint(model, args.snapshot)

    clean_loader, cor_loader, cfg.backbone.num_neighbors = viz_data_loaders(args, cfg)

    # ── 1 & 2: alignment NOT installed → raw ref coarse fed to transformer ──
    F_clean = core.collect_transformer_input(model, clean_loader, to_cuda, n_clouds, sample_idx)
    F_cor   = core.collect_transformer_input(model, cor_loader,   to_cuda, n_clouds, sample_idx)

    # ── 3: install WCT, then the SAME hook captures the aligned ref coarse ──
    stats_file = osp.join(args.align_stats_dir, "source_feats.pth")
    stage = torch.load(stats_file, weights_only=False)["ref_feats_c"]
    model.set_alignment_stats(
        src_stats = stage["src"],
        weight    = max(getattr(args, "align_weight", 1.0), 1.0),
        momentum  = getattr(args, "align_momentum", 0.02),
    )
    F_adapt = core.collect_transformer_input(model, cor_loader, to_cuda, n_clouds, sample_idx)

    core.plot_groups(F_clean, F_cor, F_adapt, tag, "PEA_TTA", out_dir=cfg.viz_dir)


def main():
    args = make_parser().parse_args()
    cfg  = make_cfg()
    torch.set_float32_matmul_precision("high")
    run(args, cfg)


if __name__ == "__main__":
    main()