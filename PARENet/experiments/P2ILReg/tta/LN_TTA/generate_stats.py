import os
import os.path as osp
from collections import defaultdict

import torch
from tqdm import tqdm

from config import make_cfg
from model import create_model
from pareconv.utils.torch import clean_checkpoint_state_dict, to_cuda
from pareconv.utils.data import precompute_neibors
from pareconv.modules.transformer.rpe_transformer_LN import AdaptiveLayerNorm, TTAContext

from dataset import train_valid_data_loader


# ---------------------------------------------------------------------------
# Source collector: POST-AFFINE y = gamma*h + beta
# ---------------------------------------------------------------------------

class _PostAffineCollector:
    """
    Hooks AdaptiveLayerNorm OUTPUT.
    Captures post-affine y for source stats (E[y], Var[y] per feature j).
    stream : 'ref' | 'src'
    """

    def __init__(self, rpe_transformer, ctx: TTAContext, stream: str):
        assert stream in ('ref', 'src')
        self._ctx    = ctx
        self._stream = stream
        self._hooks  = []
        self._sum    = {}
        self._sum_sq = {}
        self._count  = defaultdict(int)
        self._names  = {}

        for name, module in rpe_transformer.named_modules():
            if type(module) is AdaptiveLayerNorm:
                self._names[id(module)] = name
                self._hooks.append(module.register_forward_hook(self._hook_fn))

        print(f"[LN-stats] PostAffineCollector (stream={stream}): "
              f"hooked {len(self._names)} AdaptiveLayerNorm layers")

    def _hook_fn(self, module, input, output):
        # ref stream fires when _processing_corrupted=True
        if self._stream == 'ref' and not self._ctx._processing_corrupted:
            return
        # src stream fires when _processing_corrupted=False
        if self._stream == 'src' and self._ctx._processing_corrupted:
            return

        name   = self._names[id(module)]
        y      = output.detach().float()
        y_flat = y.reshape(-1, y.shape[-1])   # (N, d)

        mask = getattr(self._ctx, '_current_mask', None)
        if mask is not None:
            valid_flat = (~mask).reshape(-1).bool()
            y_flat = y_flat[valid_flat]

        if name not in self._sum:
            self._sum[name]    = y_flat.sum(dim=0).cpu()
            self._sum_sq[name] = y_flat.pow(2).sum(dim=0).cpu()
        else:
            self._sum[name]    += y_flat.sum(dim=0).cpu()
            self._sum_sq[name] += y_flat.pow(2).sum(dim=0).cpu()
        self._count[name] += y_flat.shape[0]

    @torch.no_grad()
    def get_stats(self, prefix: str):
        stats = {}
        for name in self._sum:
            n      = self._count[name]
            mu     = self._sum[name] / n                                # (d,)
            sigma2 = (self._sum_sq[name] / n - mu.pow(2)).clamp(min=0) # (d,)
            stats[name] = {
                f'm_{prefix}':      mu.mean().item(),             # scalar (for logging)
                f's_{prefix}':      sigma2.mean().sqrt().item(),  # scalar (for logging)
                f'mu_{prefix}':     mu,                           # (d,)
                f'sigma2_{prefix}': sigma2,                       # (d,)
            }
        return stats

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ---------------------------------------------------------------------------
# collect_source_ln_stats  — post-affine y, ref and src streams separately
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_source_ln_stats(model, cfg, args, save_path, force_regenerate=False):
    """
    Run the clean training set through the model and collect post-affine y
    statistics for every AdaptiveLayerNorm, separately per stream.
    """
    if osp.exists(save_path) and not force_regenerate:
        print(f"[LN-stats] Loading cached source stats from {save_path}")
        return torch.load(save_path, map_location='cpu')

    os.makedirs(osp.dirname(save_path), exist_ok=True)
    print("[LN-stats] Collecting SOURCE stats (post-affine y, ref+src separately)...")

    syn_or_real = cfg.data.dataset
    cfg.data.dataset = 'syn'
    source_loader, _, cfg.backbone.num_neighbors = train_valid_data_loader(cfg, False)
    cfg.data.dataset = syn_or_real
    ctx = model.transformer.transformer.ctx

    # Clear any previously injected stats so AdaptiveLayerNorm falls back to
    # plain super().forward(x) during collection.
    for _, module in model.transformer.transformer.named_modules():
        if type(module) is AdaptiveLayerNorm:
            module.mu_source_ref     = None
            module.sigma2_source_ref = None
            module.mu_source_src     = None
            module.sigma2_source_src = None
            module.mu_target_ref     = None
            module.sigma2_target_ref = None
            module.mu_target_src     = None
            module.sigma2_target_src = None

    col_ref = _PostAffineCollector(model.transformer.transformer, ctx, stream='ref')
    col_src = _PostAffineCollector(model.transformer.transformer, ctx, stream='src')
    model.eval()

    for data_dict in tqdm(source_loader, desc="Source LN stats"):
        data_dict = to_cuda(data_dict)
        _ = model(data_dict)

    stats_ref = col_ref.get_stats(prefix='source_ref')
    stats_src = col_src.get_stats(prefix='source_src')
    col_ref.remove()
    col_src.remove()

    def _sort_key(n):
        parts = n.split('.')
        return int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else n

    stats = {}
    for name in sorted(set(stats_ref) | set(stats_src), key=_sort_key):
        stats[name] = {}
        if name in stats_ref:
            stats[name].update(stats_ref[name])
        if name in stats_src:
            stats[name].update(stats_src[name])

    torch.save(stats, save_path)
    print(f"[LN-stats] Saved {len(stats)} layers -> {save_path}")
    for name, s in stats.items():
            ref_part = (f"ref  m={s['m_source_ref']:.4f} s={s['s_source_ref']:.4f}"
                        if 'm_source_ref' in s else "ref  (no data)")
            src_part = (f"src  m={s['m_source_src']:.4f} s={s['s_source_src']:.4f}"
                        if 'm_source_src' in s else "src  (no data)")
            print(f"  {name}:  {ref_part}  |  {src_part}")
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    from common_utils.parser import make_parser
    args = make_parser().parse_args()
    cfg  = make_cfg()

    model = create_model(cfg).cuda()
    if args.snapshot is not None:
        model.load_state_dict(
            clean_checkpoint_state_dict(
                torch.load(args.snapshot, map_location='cpu')
            )
        )
    model.eval()

    src_path = osp.join('intermediate_features', 'ln_stats', 'ln_source_stats.pth')
    collect_source_ln_stats(model, cfg, args, src_path, force_regenerate=True)