import torch
from typing import Dict, List, Optional
from pareconv.modules.transformer.rpe_transformer_LN import AdaptiveLayerNorm, TTAContext


def _collect_adaptive_lns(geo_transformer) -> Dict[str, AdaptiveLayerNorm]:
    result = {}
    for name, module in geo_transformer.transformer.named_modules():
        if type(module) is AdaptiveLayerNorm:
            result[name] = module
    return result


def _layer_index(ln_name: str) -> Optional[int]:
    """Extract the transformer layer index from a dotted name like 'layers.2.attention.norm'."""
    parts = ln_name.split('.')
    if len(parts) >= 2 and parts[0] == 'layers':
        try:
            return int(parts[1])
        except ValueError:
            pass
    return None


class LN_TTA_Manager:
    def __init__(
        self,
        transformer,
        source_stats: Dict,
        ctx: TTAContext,
        mode: str = 'oracle',
        layers_to_adapt: Optional[List[int]] = None,
        momentum: float = 0.01,
    ):
        self.transformer     = transformer
        self.ctx             = ctx
        self.mode            = mode
        self.layers_to_adapt = layers_to_adapt
        self.momentum        = momentum
        self.adaptive_lns    = _collect_adaptive_lns(transformer)

        print(f"[LN-TTA] {len(self.adaptive_lns)} AdaptiveLayerNorm layers found.")
        print(f"[LN-TTA] layers_to_adapt={layers_to_adapt}  (None = all)")
        print(f"[LN-TTA] Target stats: online Welford inside AdaptiveLayerNorm.forward()")
        
        self._inject_source_stats(source_stats)

    def _should_adapt(self, ln_name: str) -> bool:
        """Return True unless layers_to_adapt is set and this layer is not in it."""
        if self.layers_to_adapt is None:
            return True
        layer_idx = _layer_index(ln_name)
        if layer_idx is None:
            return False
        return layer_idx in self.layers_to_adapt

    def _inject_source_stats(self, source_stats: Dict):
        adapted = 0
        skipped = 0

        for ln_name, ln in self.adaptive_lns.items():
            ln.mu_target_ref     = None
            ln.sigma2_target_ref = None
            ln.mu_target_src     = None
            ln.sigma2_target_src = None
            ln.n_target_ref      = 0.0
            ln.n_target_src      = 0.0

            if not self._should_adapt(ln_name):
                # Clear all stats so AdaptiveLayerNorm falls back to plain forward
                ln.mu_source_ref     = None
                ln.sigma2_source_ref = None
                ln.mu_source_src     = None
                ln.sigma2_source_src = None
                skipped += 1
                continue

            if ln_name not in source_stats:
                print(f"[LN-TTA] WARNING: no source stats for '{ln_name}' — skipping")
                ln.mu_source_ref = ln.sigma2_source_ref = None
                ln.mu_source_src = ln.sigma2_source_src = None
                skipped += 1
                continue

            src = source_stats[ln_name]
            ln.mu_source_ref     = src['mu_source_ref']      # E[y] source, ref stream
            ln.sigma2_source_ref = src['sigma2_source_ref']  # Var[y] source, ref stream
            ln.mu_source_src     = src['mu_source_src']      # E[y] source, src stream
            ln.sigma2_source_src = src['sigma2_source_src']  # Var[y] source, src stream
            ln.momentum = self.momentum

            print(f"[LN-TTA]  {ln_name}: "
                  f"s_source_ref={src['s_source_ref']:.4f}  "
                  f"s_source_src={src['s_source_src']:.4f}")
            adapted += 1

        print(f"[LN-TTA] Adapted: {adapted}  Skipped: {skipped}")

    @torch.no_grad()
    def adapt(self, model, data_dict):
        return model(data_dict)