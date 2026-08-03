import torch
import torch.nn as nn

from tta.Source_Only.model import _build_gather_idx


def _gather_padded(stacked, gather_idx):
    """
    Gather a stacked (N, C) tensor into a padded (B, max_len, C) layout via
    ``gather_idx`` (B, max_len). Out-of-range slots map to a sentinel row of
    zeros appended at the end.
    """
    N, C = stacked.shape
    sentinel = stacked.new_zeros(1, C)
    extended = torch.cat([stacked, sentinel], dim=0)   # (N+1, C)
    return extended[gather_idx]                         # (B, max_len, C)


# ---------------------------------------------------------------------------
# MLP decoder
# ---------------------------------------------------------------------------

class _ReconDecoder(nn.Module):
    """MLP: (..., in_dim) → (..., output_pts, 3)."""

    def __init__(self, in_dim: int, hidden_dim: int = 256, output_pts: int = 512):
        super().__init__()
        self.output_pts = output_pts
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_dim, output_pts * 3),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, global_feat: torch.Tensor) -> torch.Tensor:
        out = self.net(global_feat)
        return out.view(*out.shape[:-1], self.output_pts, 3)


# ---------------------------------------------------------------------------
# Reconstruction auxiliary branch
# ---------------------------------------------------------------------------

class RecAux(nn.Module):
    """
    Reconstruction auxiliary branch.

    `self.backbone` IS `primary_model.backbone` (same Python object —
    no copy). Gradients from RecAuxLoss flow through it into θ_shar.
    """

    def __init__(self, primary_model: nn.Module, cfg):
        super().__init__()

        self.backbone = primary_model.backbone

        # ri_feats_c dim: 8 * (init_dim // 3) * 3 — same as the original.
        coarse_dim = 8 * (cfg.backbone.init_dim // 3) * 3

        self.decoder = _ReconDecoder(coarse_dim, hidden_dim=256, output_pts=512)

    def forward(self, data_dict: dict):
        """
        Returns:
            (rec_ref, rec_src), each of shape (B, output_pts, 3).
        """
        _, _, _, ri_feats_c, _ = self.backbone(data_dict)

        B          = data_dict['batch_size']
        device     = ri_feats_c.device
        N_c_total  = ri_feats_c.shape[0]

        # Build padded gather indices for ref and src coarse features.
        ref_gather, ref_valid = _build_gather_idx(
            data_dict['ref_lens_c'], B, int(data_dict['ref_max_c']),
            data_dict['ref_off_c'], device, sentinel_row=N_c_total,
        )
        src_gather, src_valid = _build_gather_idx(
            data_dict['src_lens_c'], B, int(data_dict['src_max_c']),
            data_dict['src_off_c'], device, sentinel_row=N_c_total,
        )

        ref_feats_pad = _gather_padded(ri_feats_c, ref_gather)   # (B, max_ref_c, C)
        src_feats_pad = _gather_padded(ri_feats_c, src_gather)   # (B, max_src_c, C)

        NEG_INF = torch.finfo(ref_feats_pad.dtype).min
        ref_feats_pad = ref_feats_pad.masked_fill(~ref_valid.unsqueeze(-1), NEG_INF)
        src_feats_pad = src_feats_pad.masked_fill(~src_valid.unsqueeze(-1), NEG_INF)

        ref_global = ref_feats_pad.max(dim=1)[0]   # (B, coarse_dim)
        src_global = src_feats_pad.max(dim=1)[0]   # (B, coarse_dim)

        rec_ref = self.decoder(ref_global)         # (B, output_pts, 3)
        rec_src = self.decoder(src_global)         # (B, output_pts, 3)
        return rec_ref, rec_src


# ---------------------------------------------------------------------------
# Reconstruction loss
# ---------------------------------------------------------------------------

class RecAuxLoss(nn.Module):
    """
    Mean Chamfer-distance reconstruction loss over the whole batch.

        loss = (1/2) · mean_b [ chamfer(rec_ref_b, X_ref_b)
                              + chamfer(rec_src_b, X_src_b) ]

    Targets (X_ref, X_src) are detached — gradients flow only through
    the reconstructions back into the decoder and the shared backbone.

    chamfer(a, b) = mean_p min_n ‖a_p − b_n‖²  +  mean_n min_p ‖a_p − b_n‖²
    """

    def forward(self, recs, data_dict: dict) -> torch.Tensor:
        rec_ref, rec_src = recs        # each (B, P, 3)
        B      = data_dict['batch_size']
        device = rec_ref.device

        points0 = data_dict['points'][0].detach()       # (N_0_total, 3)
        N_total = points0.shape[0]

        ref_lens = data_dict['ref_lens']
        src_lens = data_dict['src_lens']
        ref_max  = int(ref_lens.max().item()) if torch.is_tensor(ref_lens) else max(ref_lens)
        src_max  = int(src_lens.max().item()) if torch.is_tensor(src_lens) else max(src_lens)

        ref_gather, ref_valid = _build_gather_idx(
            ref_lens, B, ref_max, data_dict['ref_off'], device, sentinel_row=N_total,
        )
        src_gather, src_valid = _build_gather_idx(
            src_lens, B, src_max, data_dict['src_off'], device, sentinel_row=N_total,
        )

        ref_target = _gather_padded(points0, ref_gather)   # (B, max_N_ref, 3)
        src_target = _gather_padded(points0, src_gather)   # (B, max_N_src, 3)

        loss_ref = self._batched_chamfer(rec_ref, ref_target, ref_valid)
        loss_src = self._batched_chamfer(rec_src, src_target, src_valid)
        # Mean over the {ref, src} pair so the scale matches the original
        # per-cloud chamfer loss.
        return 0.5 * (loss_ref + loss_src)

    @staticmethod
    def _batched_chamfer(rec, target_pad, target_valid):
        """
        rec:           (B, P, 3)        — dense, all valid
        target_pad:    (B, max_N, 3)    — padded with arbitrary values
        target_valid:  (B, max_N) bool  — True at real positions

        Returns: scalar = mean_b [ mean_p min_n d(b,p,n)
                                 + mean_{valid n} min_p d(b,p,n) ]
        """
        # Squared L2 pairwise distance: (B, P, max_N)
        rec_sq = (rec * rec).sum(-1, keepdim=True)                       # (B, P, 1)
        tgt_sq = (target_pad * target_pad).sum(-1, keepdim=True)         # (B, max_N, 1)
        ab     = torch.bmm(rec, target_pad.transpose(1, 2))              # (B, P, max_N)
        dist   = rec_sq + tgt_sq.transpose(1, 2) - 2.0 * ab              # (B, P, max_N)

        # rec → nearest valid target: mask invalid targets with +inf so
        # they're never picked as the nearest neighbour.
        invalid = ~target_valid.unsqueeze(1)                             # (B, 1, max_N)
        dist_a  = dist.masked_fill(invalid, float('inf'))
        chamfer_a = dist_a.min(dim=2)[0].mean(dim=1)                     # (B,)

        # target → nearest rec: rec side is dense, no masking needed for
        # the inner min; we only need to average over valid target positions.
        min_per_n = dist.min(dim=1)[0]                                   # (B, max_N)
        valid_f   = target_valid.float()
        chamfer_b = (min_per_n * valid_f).sum(dim=1) / valid_f.sum(dim=1).clamp(min=1)

        return (chamfer_a + chamfer_b).mean()