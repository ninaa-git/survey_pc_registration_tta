import torch
import torch.nn as nn
import torch.nn.functional as F

from pareconv.modules.dual_matching import PointDualMatching
from pareconv.modules.geotransformer import (
    GeometricTransformer_LN, #LN
    SuperPointTargetGenerator,
    BatchedSuperPointMatching,
)
from pareconv.modules.ops import batched_point_to_node_partition
from pareconv.modules.registration import (
    BatchedHypothesisProposer,
    batched_get_node_correspondences,
)

from backbone import PAREConvFPN

from pareconv.modules.transformer.rpe_transformer_LN import TTAContext #LN


# ---------------------------------------------------------------------------
# Gather-scatter helpers
# ---------------------------------------------------------------------------

def _build_gather_idx(lens, B, max_len, base_off, device, sentinel_row,
                      include_sentinel_col=False):
    """Map padded slots to source rows.

    For each pair b:
        gather_idx[b, n] = base_off[b] + n   if n < lens[b]
                         = sentinel_row      otherwise

    If include_sentinel_col=True, output is [B, max_len + 1] with column
    max_len always pointing to the sentinel (used by fine layout).
    """
    lens_t = lens if torch.is_tensor(lens) else torch.tensor(lens, device=device)
    if lens_t.device != device:
        lens_t = lens_t.to(device)
    if isinstance(base_off, list):
        offs = torch.tensor(base_off, device=device, dtype=torch.long)
    else:
        offs = base_off.to(device).long()

    cols_count = max_len + 1 if include_sentinel_col else max_len
    cols = torch.arange(cols_count, device=device).unsqueeze(0).expand(B, -1)
    valid = cols < lens_t.unsqueeze(1)
    gather_idx = torch.where(
        valid, cols + offs.unsqueeze(1), torch.full_like(cols, sentinel_row),
    )
    return gather_idx, valid


def _gather_padded_pair(stacked_a, stacked_b, gather_idx):
    """Gather two stacked tensors into padded layout in one fused op.

    stacked_a: [N, Ca], stacked_b: [N, *Fb] →
        padded_a [B, max_len, Ca],
        padded_b [B, max_len, *Fb].
    """
    Ca = stacked_a.shape[-1]
    fb_shape = stacked_b.shape[1:]
    Fb = 1
    for s in fb_shape:
        Fb *= s
    sb_flat = stacked_b.reshape(stacked_b.shape[0], Fb) if len(fb_shape) > 1 else stacked_b
    fused = torch.cat([stacked_a, sb_flat], dim=-1)
    sentinel = fused.new_zeros(1, Ca + Fb)
    extended = torch.cat([fused, sentinel], dim=0)
    pad = extended[gather_idx]
    pad_a = pad[..., :Ca].contiguous()
    pad_b = pad[..., Ca:].contiguous()
    if len(fb_shape) > 1:
        pad_b = pad_b.reshape(*pad.shape[:-1], *fb_shape)
    return pad_a, pad_b


def _gather_padded_fine(feats_f, m_scores, points_f, re_feats_f, gather_idx):
    """Gather four fine-level tensors into padded layout with sentinel slot.

    Returns feats_pad, scores_pad, points_pad, re_pad of shape
    [B, max_len + 1, ...]. Column max_len is always the sentinel (zeros).
    """
    Cf = feats_f.shape[-1]
    re_shape = re_feats_f.shape[1:]
    Cre = 1
    for s in re_shape:
        Cre *= s

    re_flat = re_feats_f.reshape(feats_f.shape[0], Cre)
    fused = torch.cat([feats_f, m_scores.unsqueeze(-1), points_f, re_flat], dim=-1)
    sentinel = fused.new_zeros(1, fused.shape[-1])
    extended = torch.cat([fused, sentinel], dim=0)
    pad = extended[gather_idx]

    feats_pad  = pad[..., :Cf].contiguous()
    scores_pad = pad[..., Cf:Cf + 1].squeeze(-1).contiguous()
    points_pad = pad[..., Cf + 1:Cf + 1 + 3].contiguous()
    re_pad = pad[..., Cf + 1 + 3:].reshape(*pad.shape[:-1], *re_shape).contiguous()
    return feats_pad, scores_pad, points_pad, re_pad


def _stack_transforms(transforms, B, device):
    """Normalize the data_dict['transform'] field to a [B, 4, 4] tensor."""
    if isinstance(transforms, list):
        ts = transforms
    elif isinstance(transforms, torch.Tensor) and transforms.dim() >= 1 \
            and transforms.shape[0] == B and B > 1:
        ts = [transforms[i] for i in range(B)]
    else:
        ts = [transforms]
    ts = [t.detach().to(device) if isinstance(t, torch.Tensor) else t for t in ts]
    return torch.stack(ts, dim=0)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class PARE_Net(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_points_in_patch = cfg.model.num_points_in_patch
        self.matching_radius = cfg.model.ground_truth_matching_radius

        self.backbone = PAREConvFPN(
            cfg.backbone.init_dim, cfg.backbone.output_dim, cfg.backbone.kernel_size,
            cfg.backbone.share_nonlinearity, cfg.backbone.conv_way,
            cfg.backbone.use_xyz, cfg.fine_matching.use_encoder_re_feats,
        )
        ctx = TTAContext()
        self.transformer = GeometricTransformer_LN(
            cfg.geotransformer.input_dim, cfg.geotransformer.output_dim,
            cfg.geotransformer.hidden_dim, cfg.geotransformer.num_heads,
            cfg.geotransformer.blocks, cfg.geotransformer.sigma_d,
            cfg.geotransformer.sigma_a, cfg.geotransformer.angle_k,
            reduction_a=cfg.geotransformer.reduction_a,
            ctx=ctx,
        )
        self.coarse_target = SuperPointTargetGenerator(
            cfg.coarse_matching.num_targets, cfg.coarse_matching.overlap_threshold,
        )
        self.coarse_matching = BatchedSuperPointMatching(
            cfg.coarse_matching.num_correspondences, cfg.coarse_matching.dual_normalization,
        )
        self.fine_matching = BatchedHypothesisProposer(
            cfg.fine_matching.topk, cfg.fine_matching.acceptance_radius,
            confidence_threshold=cfg.fine_matching.confidence_threshold,
            num_hypotheses=cfg.fine_matching.num_hypotheses,
            num_refinement_steps=cfg.fine_matching.num_refinement_steps,
            compute_ir_final=True,
        )
        self.point_matching = PointDualMatching(dim=cfg.backbone.output_dim // 3 * 3)

    def forward(self, data_dict):
        B = data_dict['batch_size']
        device = data_dict['features'].device

        # ---- 1. Backbone ----
        re_feats_f, feats_f, _, feats_c, m_scores = self.backbone(data_dict)

        points_c = data_dict['points'][-1].detach()
        points_f = data_dict['points'][1].detach()
        points   = data_dict['points'][0].detach()

        # Per-pair lengths: GPU for math, CPU for slicing without syncs
        ref_lens_c = data_dict['ref_lens_c'].to(device)
        src_lens_c = data_dict['src_lens_c'].to(device)
        ref_lens_f = data_dict['ref_lens_f'].to(device)
        src_lens_f = data_dict['src_lens_f'].to(device)
        src_lens_cpu   = data_dict['src_lens'].tolist()
        ref_lens_f_cpu = data_dict['ref_lens_f'].tolist()
        src_lens_f_cpu = data_dict['src_lens_f'].tolist()

        ref_off_c = data_dict['ref_off_c'].tolist()
        src_off_c = data_dict['src_off_c'].tolist()
        ref_off_f = data_dict['ref_off_f'].tolist()
        src_off_f = data_dict['src_off_f'].tolist()
        src_off   = data_dict['src_off'].tolist()

        transforms_b = _stack_transforms(data_dict['transform'], B, device)

        # ---- 2. Coarse padded layout + transformer ----
        ref_max_c  = data_dict['ref_max_c']
        src_max_c  = data_dict['src_max_c']
        ref_mask_c = data_dict['ref_mask_c'].to(device)
        src_mask_c = data_dict['src_mask_c'].to(device)

        N_c_total = feats_c.shape[0]
        ref_gather_c, _ = _build_gather_idx(
            ref_lens_c, B, ref_max_c, ref_off_c, device, sentinel_row=N_c_total,
        )
        src_gather_c, _ = _build_gather_idx(
            src_lens_c, B, src_max_c, src_off_c, device, sentinel_row=N_c_total,
        )
        ref_feats_c_pad, ref_points_c_pad = _gather_padded_pair(feats_c, points_c, ref_gather_c)
        src_feats_c_pad, src_points_c_pad = _gather_padded_pair(feats_c, points_c, src_gather_c)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            ref_feats_c_t, src_feats_c_t, _ = self.transformer(
                ref_points_c_pad, src_points_c_pad,
                ref_feats_c_pad,  src_feats_c_pad,
                ref_masks=ref_mask_c, src_masks=src_mask_c,
            )

        ref_feats_c_norm = F.normalize(ref_feats_c_t.float(), p=2, dim=2)
        src_feats_c_norm = F.normalize(src_feats_c_t.float(), p=2, dim=2)

        feats_f    = feats_f.float()
        re_feats_f = re_feats_f.float()
        m_scores   = m_scores.float()

        # ---- 3. Fine padded layout (with sentinel slot at index max_f) ----
        ref_max_f = max(ref_lens_f_cpu)
        src_max_f = max(src_lens_f_cpu)
        N_f_total = feats_f.shape[0]

        ref_gather_f, ref_mask_f_full = _build_gather_idx(
            ref_lens_f, B, ref_max_f, ref_off_f, device,
            sentinel_row=N_f_total, include_sentinel_col=True,
        )
        src_gather_f, src_mask_f_full = _build_gather_idx(
            src_lens_f, B, src_max_f, src_off_f, device,
            sentinel_row=N_f_total, include_sentinel_col=True,
        )
        ref_mask_f = ref_mask_f_full[:, :ref_max_f]
        src_mask_f = src_mask_f_full[:, :src_max_f]

        (ref_feats_f_pad, ref_m_scores_pad,
         ref_points_f_pad, re_ref_feats_f_pad) = _gather_padded_fine(
            feats_f, m_scores, points_f, re_feats_f, ref_gather_f,
        )
        (src_feats_f_pad, src_m_scores_pad,
         src_points_f_pad, re_src_feats_f_pad) = _gather_padded_fine(
            feats_f, m_scores, points_f, re_feats_f, src_gather_f,
        )

        # ---- 4. Patch partition (batched) ----
        K = self.num_points_in_patch
        _, ref_node_masks_b, ref_node_knn_indices_b, ref_node_knn_masks_b = \
            batched_point_to_node_partition(
                ref_points_f_pad[:, :ref_max_f], ref_points_c_pad, K, ref_mask_f, ref_mask_c,
            )
        _, src_node_masks_b, src_node_knn_indices_b, src_node_knn_masks_b = \
            batched_point_to_node_partition(
                src_points_f_pad[:, :src_max_f], src_points_c_pad, K, src_mask_f, src_mask_c,
            )

        # KNN data per coarse point: [B, N_c_max, K, ...]
        b_idx_full = torch.arange(B, device=device).view(B, 1, 1)
        ref_node_knn_points_b   = ref_points_f_pad[b_idx_full,    ref_node_knn_indices_b]
        src_node_knn_points_b   = src_points_f_pad[b_idx_full,    src_node_knn_indices_b]
        ref_node_knn_feats_b    = ref_feats_f_pad[b_idx_full,     ref_node_knn_indices_b]
        src_node_knn_feats_b    = src_feats_f_pad[b_idx_full,     src_node_knn_indices_b]
        ref_node_knn_scores_b   = ref_m_scores_pad[b_idx_full,    ref_node_knn_indices_b]
        src_node_knn_scores_b   = src_m_scores_pad[b_idx_full,    src_node_knn_indices_b]
        re_ref_node_knn_feats_b = re_ref_feats_f_pad[b_idx_full,  ref_node_knn_indices_b]
        re_src_node_knn_feats_b = re_src_feats_f_pad[b_idx_full,  src_node_knn_indices_b]

        # ---- 5. GT correspondences (batched) ----
        gt_corr_per_pair, gt_overlap_per_pair = batched_get_node_correspondences(
            ref_points_c_pad, src_points_c_pad,
            ref_node_knn_points_b, src_node_knn_points_b,
            transforms_b, self.matching_radius,
            ref_node_masks_b, src_node_masks_b,
            ref_node_knn_masks_b, src_node_knn_masks_b,
        )

        # ---- 6. Predicted coarse correspondences (no_grad) ----
        with torch.no_grad():
            ref_corr_idx_pad, src_corr_idx_pad, _, corr_real_mask = self.coarse_matching(
                ref_feats_c_norm, src_feats_c_norm, ref_mask_c, src_mask_c,
            )

        # ---- 7. Per-pair correspondence selection ----
        ref_corr_idx_list, src_corr_idx_list, P_per_pair_list = self._select_corrs(
            B, gt_corr_per_pair, gt_overlap_per_pair,
            ref_corr_idx_pad, src_corr_idx_pad, corr_real_mask,
        )

        P_max = max(P_per_pair_list) if P_per_pair_list else 0
        if P_max == 0:
            return {'batch_size': B, 'batched': None, 'empty': True}

        # ---- 8. Pad correspondences to [B, P_max] ----
        P_total_cpu = sum(P_per_pair_list)
        P_per_pair = torch.tensor(P_per_pair_list, device=device)
        all_ref_idx = torch.cat(ref_corr_idx_list)
        all_src_idx = torch.cat(src_corr_idx_list)
        batch_ids = torch.repeat_interleave(torch.arange(B, device=device), P_per_pair)
        offsets = torch.zeros(B + 1, dtype=torch.long, device=device)
        offsets[1:] = P_per_pair.cumsum(0)
        within = torch.arange(P_total_cpu, device=device) - offsets[batch_ids]

        ref_corr_idx_padded = torch.zeros(B, P_max, dtype=torch.long, device=device)
        src_corr_idx_padded = torch.zeros(B, P_max, dtype=torch.long, device=device)
        pair_corr_masks     = torch.zeros(B, P_max, dtype=torch.bool, device=device)
        ref_corr_idx_padded[batch_ids, within] = all_ref_idx
        src_corr_idx_padded[batch_ids, within] = all_src_idx
        pair_corr_masks[batch_ids, within] = True

        # Gather KNN data per correspondence: [B, P_max, K, ...]
        b_idx_corr = torch.arange(B, device=device).view(B, 1).expand(-1, P_max)
        ref_corr_knn_points  = ref_node_knn_points_b[b_idx_corr,    ref_corr_idx_padded]
        ref_corr_knn_feats   = ref_node_knn_feats_b[b_idx_corr,     ref_corr_idx_padded]
        ref_corr_knn_scores  = ref_node_knn_scores_b[b_idx_corr,    ref_corr_idx_padded]
        ref_corr_knn_masks   = ref_node_knn_masks_b[b_idx_corr,     ref_corr_idx_padded]
        re_ref_corr_knn_feats = re_ref_node_knn_feats_b[b_idx_corr, ref_corr_idx_padded]
        src_corr_knn_points  = src_node_knn_points_b[b_idx_corr,    src_corr_idx_padded]
        src_corr_knn_feats   = src_node_knn_feats_b[b_idx_corr,     src_corr_idx_padded]
        src_corr_knn_scores  = src_node_knn_scores_b[b_idx_corr,    src_corr_idx_padded]
        src_corr_knn_masks   = src_node_knn_masks_b[b_idx_corr,     src_corr_idx_padded]
        re_src_corr_knn_feats = re_src_node_knn_feats_b[b_idx_corr, src_corr_idx_padded]

        # Mask padded correspondence slots
        ref_corr_knn_masks = ref_corr_knn_masks & pair_corr_masks.unsqueeze(2)
        src_corr_knn_masks = src_corr_knn_masks & pair_corr_masks.unsqueeze(2)

        # ---- 9. Point matching (only on real correspondences) ----
        valid_flat = pair_corr_masks.view(-1)

        def flat(t):
            return t.reshape(B * P_max, *t.shape[2:])[valid_flat]

        flat_ref_pts   = flat(ref_corr_knn_points)
        flat_src_pts   = flat(src_corr_knn_points)
        flat_ref_feats = flat(ref_corr_knn_feats)
        flat_src_feats = flat(src_corr_knn_feats)
        flat_ref_s     = flat(ref_corr_knn_scores)
        flat_src_s     = flat(src_corr_knn_scores)
        flat_ref_m     = flat(ref_corr_knn_masks)
        flat_src_m     = flat(src_corr_knn_masks)
        flat_re_ref    = flat(re_ref_corr_knn_feats)
        flat_re_src    = flat(re_src_corr_knn_feats)

        real_matching_scores = self.point_matching(
            flat_ref_feats, flat_src_feats,
            flat_ref_s,     flat_src_s,
            flat_ref_m,     flat_src_m,
        )

        # Scatter back to padded layout for the hypothesis proposer
        full_matching_scores = real_matching_scores.new_zeros(B * P_max, K, K)
        full_matching_scores[valid_flat] = real_matching_scores
        matching_scores_padded = full_matching_scores.view(B, P_max, K, K)

        # ---- 10. Hypothesis proposer ----
        hyp_outputs = self.fine_matching(
            ref_corr_knn_points, src_corr_knn_points,
            re_ref_corr_knn_feats, re_src_corr_knn_feats,
            ref_corr_knn_masks, src_corr_knn_masks,
            matching_scores_padded,
            pair_corr_masks=pair_corr_masks,
        )
        # hyp_outputs[b] = (ref_corr_pts, src_corr_pts, corr_scores, est_T,
        #                   hypotheses, re_ref_corr_feats, re_src_corr_feats)
        est_T_b = torch.stack([h[3] for h in hyp_outputs], dim=0)

        ir_final_per_pair = torch.stack([h[-1].detach().float() for h in hyp_outputs])
        valid = ~torch.isnan(ir_final_per_pair)
        self._ir_final_n_minus_1 = (
            ir_final_per_pair[valid].mean() if valid.any()
            else torch.tensor(float('nan'), device=ir_final_per_pair.device)
        )

        # ---- 11. Build the `batched` dict consumed by loss/evaluator ----
        batched = {
            # Coarse loss
            'ref_feats_c_padded':  ref_feats_c_norm,
            'src_feats_c_padded':  src_feats_c_norm,
            'ref_mask_c':          ref_mask_c,
            'src_mask_c':          src_mask_c,
            'gt_corr_per_pair':    gt_corr_per_pair,
            'gt_overlap_per_pair': gt_overlap_per_pair,

            # Evaluator coarse precision
            'ref_corr_idx_pad':    ref_corr_idx_padded,
            'src_corr_idx_pad':    src_corr_idx_padded,
            'pair_corr_masks':     pair_corr_masks,

            # Fine loss inputs (flat over real correspondences)
            'ref_node_corr_knn_points':   flat_ref_pts,
            'src_node_corr_knn_points':   flat_src_pts,
            'ref_node_corr_knn_masks':    flat_ref_m,
            'src_node_corr_knn_masks':    flat_src_m,
            'ref_node_corr_knn_scores':   flat_ref_s,
            'src_node_corr_knn_scores':   flat_src_s,
            're_ref_node_corr_knn_feats': flat_re_ref,
            're_src_node_corr_knn_feats': flat_re_src,
            'matching_scores':            real_matching_scores,
            'corr_to_pair':               batch_ids,
            'transforms':                 transforms_b,

            # Evaluator registration metrics
            'estimated_transforms':     est_T_b,
            'ref_corr_points_per_pair': [h[0] for h in hyp_outputs],
            'src_corr_points_per_pair': [h[1] for h in hyp_outputs],
            'ir_final_per_pair':        ir_final_per_pair,
            'src_points_per_pair':      [points[src_off[b]:src_off[b] + src_lens_cpu[b]]
                                         for b in range(B)],
        }

        return {'batch_size': B, 'batched': batched}

    def _select_corrs(self, B, gt_corr_per_pair, gt_overlap_per_pair,
                      ref_corr_idx_pad, src_corr_idx_pad, corr_real_mask):
        """Build per-pair correspondence index lists.

        Training: replace predictions with GT-derived targets where GT exists;
        fall back to predicted for pairs with no GT.
        Eval: use predicted (validity-masked) correspondences.
        """
        ref_list, src_list, lens = [], [], []
        if self.training:
            for b in range(B):
                gt_idx, gt_overl = gt_corr_per_pair[b], gt_overlap_per_pair[b]
                if gt_idx.numel() == 0:
                    valid = corr_real_mask[b]
                    ref_list.append(ref_corr_idx_pad[b][valid])
                    src_list.append(src_corr_idx_pad[b][valid])
                    lens.append(int(valid.sum().item()))
                    continue
                ref_t, src_t, _ = self.coarse_target(gt_idx, gt_overl)
                ref_list.append(ref_t)
                src_list.append(src_t)
                lens.append(ref_t.shape[0])
        else:
            valid_counts = corr_real_mask.sum(dim=1).tolist()
            for b in range(B):
                valid = corr_real_mask[b]
                ref_list.append(ref_corr_idx_pad[b][valid])
                src_list.append(src_corr_idx_pad[b][valid])
            lens = valid_counts
        return ref_list, src_list, lens


def create_model(config):
    return PARE_Net(config)