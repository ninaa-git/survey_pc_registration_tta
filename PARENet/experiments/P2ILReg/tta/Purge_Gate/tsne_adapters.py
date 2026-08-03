import torch
import numpy as np

from config import make_cfg
from common_utils.parser import make_parser
from pareconv.utils.torch import to_cuda

import tsne_core as core

PURGE_SIZES = [0.0, 0.1, 0.5]


def _tag(args, cfg):
    return f"{cfg.dataset_name}_{args.corruption}_{args.severity}"


def _divergence_split(feats_pad, mask, src_stats, purge_size, rng=None):
    """Return (kept_feats, purged_feats) using the model's exact criterion.

    feats_pad : [B, N, C]  pre-transformer ref features (b_ref_feats_c_pad)
    mask      : [B, N]     bool, valid token positions
    src_stats : dict with 'mean' [C] and 'std' [C] tensors
    purge_size: float in [0, 1)

    Per-token z-score is applied before returning (matches collect_transformer_input).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    device = feats_pad.device
    source_mean = src_stats['mean'].to(feats_pad.dtype).to(device).view(1, 1, -1)
    source_std  = src_stats['std'].to(feats_pad.dtype).to(device).view(1, 1, -1)

    diff   = feats_pad - source_mean
    div    = torch.linalg.norm(diff / (source_std + 1e-8), dim=-1)   # [B, N]
    div    = div.masked_fill(~mask, float("-inf"))

    n_valid = mask.sum(1)                                             # [B]
    k       = (n_valid.float() * purge_size).long()                  # [B]

    kept_list, purged_list = [], []
    for b in range(feats_pad.shape[0]):
        f_b = feats_pad[b]                                            # [N, C]
        # per-token z-score to match collect_transformer_input
        fm  = f_b.mean(-1, keepdim=True)
        fs  = f_b.std(-1, keepdim=True)
        f_b = (f_b - fm) / (fs + 1e-6)

        order      = torch.argsort(div[b], descending=True)
        purged_idx = order[:k[b]]
        kept_idx   = order[k[b]:n_valid[b]]

        kept_list.append(f_b[kept_idx].cpu().numpy())
        if k[b] > 0:
            purged_list.append(f_b[purged_idx].cpu().numpy())

    return kept_list, purged_list


@torch.no_grad()
def run(args, cfg):
    from tta.Purge_Gate.model import create_model
    from dataset import viz_data_loaders
    from generate_stats import load_or_collect

    n_clouds   = args.viz_max_clouds
    sample_idx = args.viz_sample
    tag        = _tag(args, cfg)
    core.print_viz_window(sample_idx, n_clouds)
    cfg.test.batch_size  = 1
    cfg.test.num_workers = 0

    feat = args.feature

    model = create_model(cfg).cuda()
    if getattr(args, "snapshot", None):
        core.load_checkpoint(model, args.snapshot)
    model.eval()

    # ── Clean baseline ────────────────────────────────────────────────────
    clean_loader, cor_loader, cfg.backbone.num_neighbors = viz_data_loaders(args, cfg)
    F_clean = core.collect_transformer_input(model, clean_loader, to_cuda, n_clouds, sample_idx)

    # ── Source stats for the purge divergence criterion ───────────────────
    src_stats = load_or_collect(
        model, cfg,
        f"intermediate_features/purge_gate_stats/{args.stats_mode}_{feat}_intermediates.pth",
        [feat], mode=args.stats_mode, force_regenerate=False,
    )[feat]

    # ── Corrupted: adaptive purge selection ───────────────────────────────
    kept_list, purged_list = [], []
    chosen_sizes = []
    rng = np.random.default_rng(0)

    for i, data in core.iter_viz_batches(cor_loader, sample_idx, n_clouds):
        data = to_cuda(data)

        # --- try each purge size, keep the best IR_final ---
        best_ir = -1.0
        best_ps = 0.0
        for ps in PURGE_SIZES:
            try:
                out = model.forward_prototype_purge(data, feat, src_stats, ps)
                if out.get("empty"):
                    continue
                ir = float(out["batched"]["ir_final_per_pair"][0])
                if not np.isnan(ir) and ir > best_ir:
                    best_ir = ir
                    best_ps = ps
            except Exception as e:
                print(f"  [Purge_Gate] purge_size={ps} failed on batch {i}: {e}")

        chosen_sizes.append(best_ps)

        # --- extract kept/purged tokens with the winning ratio ---
        try:
            int_out = model.forward_out_intermediate(data)
        except Exception as e:
            print(f"  [Purge_Gate] forward_out_intermediate failed on batch {i}: {e}")
            continue

        feats_pad = int_out["b_ref_feats_c_pad"].float()    # [B, N, C]
        mask      = int_out["b_ref_masks_c"].bool()         # [B, N]

        k_list, p_list = _divergence_split(feats_pad, mask, src_stats, best_ps, rng=rng)
        kept_list.extend(k_list)
        purged_list.extend(p_list)

    # ── Print chosen purge-size distribution ─────────────────────────────
    print(f"\n[Purge_Gate | {tag}] chosen purge_size per sample ({len(chosen_sizes)} total):")
    for ps in PURGE_SIZES:
        n = chosen_sizes.count(ps)
        bar = "█" * n
        print(f"  {ps:.1f}  {bar} ({n})")

    if not kept_list:
        print("  [Purge_Gate] no kept tokens collected — aborting plot")
        return

    F_kept   = np.concatenate(kept_list)
    F_purged = np.concatenate(purged_list) if purged_list else np.zeros((1, F_kept.shape[1]))
    core.plot_selection(F_clean, F_kept, F_purged, tag, out_dir=cfg.viz_dir)


def main():
    args = make_parser().parse_args()
    cfg  = make_cfg()
    torch.set_float32_matmul_precision("high")
    run(args, cfg)


if __name__ == "__main__":
    main()
