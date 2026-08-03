import numpy as np
import torch
import torch.nn as nn

from pareconv.modules.loss import WeightedCircleLoss
from pareconv.modules.ops.transformation import apply_transform
from pareconv.modules.registration.metrics import isotropic_transform_error
from pareconv.modules.ops.pairwise_distance import pairwise_distance


def cal_error(gt, pred, print_error=False):
    '''
    Modified from https://github.com/zixinyang9109/P2P
    '''
    diff = np.linalg.norm(gt - pred, axis=1)
    diff_mean = np.mean(diff); diff_std = np.std(diff); diff_max = np.max(diff)
    diff = pred - gt
    RE = np.sqrt(np.sum(diff * diff) / len(diff))
    if print_error:
        print("mean error: %0.2f, max: %0.2f, std: %0.2f, RE: %0.2f" % (diff_mean, diff_max, diff_std, RE))
    return diff_mean, diff_max, diff_std, RE


# ---------------------------------------------------------------------------
# Coarse loss (batched)
# ---------------------------------------------------------------------------

class CoarseMatchingLossBatched(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.weighted_circle_loss = WeightedCircleLoss(
            cfg.coarse_loss.positive_margin,
            cfg.coarse_loss.negative_margin,
            cfg.coarse_loss.positive_optimal,
            cfg.coarse_loss.negative_optimal,
            cfg.coarse_loss.log_scale,
        )
        self.positive_overlap = cfg.coarse_loss.positive_overlap

    def forward(self, batched):
        ref_feats = batched['ref_feats_c_padded']
        src_feats = batched['src_feats_c_padded']
        ref_mask  = batched['ref_mask_c']
        src_mask  = batched['src_mask_c']
        gt_corr_list  = batched['gt_corr_per_pair']
        gt_ovlap_list = batched['gt_overlap_per_pair']

        B, N, _ = ref_feats.shape
        M = src_feats.shape[1]
        device = ref_feats.device

        feat_dists = torch.sqrt(pairwise_distance(ref_feats, src_feats, normalized=True))

        # Vectorized sparse scatter into [B, N, M] dense overlap matrix
        sizes = [gi.shape[0] for gi in gt_corr_list]
        if sum(sizes) == 0:
            overlaps = torch.zeros(B, N, M, device=device, dtype=feat_dists.dtype)
        else:
            sizes_t = torch.tensor(sizes, device=device)
            all_corr = torch.cat(gt_corr_list, dim=0)         # [G_total, 2]
            all_ovl  = torch.cat(gt_ovlap_list, dim=0)        # [G_total]
            batch_ids = torch.repeat_interleave(torch.arange(B, device=device), sizes_t)
            overlaps = torch.zeros(B, N, M, device=device, dtype=feat_dists.dtype)
            overlaps[batch_ids, all_corr[:, 0], all_corr[:, 1]] = all_ovl

        pair_mask = ref_mask.unsqueeze(2) & src_mask.unsqueeze(1)
        overlaps = overlaps * pair_mask.float()
        feat_dists_for_loss = feat_dists.masked_fill(~pair_mask, 0.0)

        pos_masks = overlaps > self.positive_overlap
        neg_masks = (overlaps == 0) & pair_mask
        pos_scales = torch.sqrt(overlaps * pos_masks.float())

        loss = self.weighted_circle_loss(pos_masks, neg_masks, feat_dists_for_loss, pos_scales)
        return loss


# ---------------------------------------------------------------------------
# Fine loss (batched)
# ---------------------------------------------------------------------------

class FineMatchingLossBatched(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.positive_radius = cfg.fine_loss.positive_radius
        self.negative_radius = cfg.fine_loss.negative_radius
        self.positive_margin = cfg.fine_loss.positive_margin
        self.negative_margin = cfg.fine_loss.negative_margin

    def forward(self, batched):
        ref_pts  = batched['ref_node_corr_knn_points']
        src_pts  = batched['src_node_corr_knn_points']
        ref_m    = batched['ref_node_corr_knn_masks']
        src_m    = batched['src_node_corr_knn_masks']
        ref_s    = batched['ref_node_corr_knn_scores']
        src_s    = batched['src_node_corr_knn_scores']
        re_ref   = batched['re_ref_node_corr_knn_feats']
        re_src   = batched['re_src_node_corr_knn_feats']
        ms       = batched['matching_scores']
        c2p      = batched['corr_to_pair']
        Ts       = batched['transforms']

        if ref_pts.shape[0] == 0:
            zero = torch.zeros((), device=Ts.device, dtype=Ts.dtype)
            return zero, zero

        per_corr_T = Ts[c2p]
        src_pts_aligned = apply_transform(src_pts, per_corr_T)
        dists = pairwise_distance(ref_pts, src_pts_aligned)

        gt_masks = ref_m.unsqueeze(2) & src_m.unsqueeze(1)
        gt_corr_map = (dists < self.positive_radius ** 2) & gt_masks
        slack_row = (gt_corr_map.sum(2) == 0) & ref_m
        slack_col = (gt_corr_map.sum(1) == 0) & src_m

        eps = 1e-7
        if gt_corr_map.any():
            pos_term = ms[gt_corr_map].clamp(min=eps).log().mean()
        else:
            pos_term = torch.zeros((), device=ms.device, dtype=ms.dtype)
        if slack_row.any():
            row_term = (1 - ref_s)[slack_row].clamp(min=eps).log().mean()
        else:
            row_term = torch.zeros((), device=ms.device, dtype=ms.dtype)
        if slack_col.any():
            col_term = (1 - src_s)[slack_col].clamp(min=eps).log().mean()
        else:
            col_term = torch.zeros((), device=ms.device, dtype=ms.dtype)
        fine_ri_loss = -(pos_term + 0.5 * row_term + 0.5 * col_term)

        neg_map = (dists > self.negative_radius ** 2) & gt_masks
        fine_re_loss = self._fine_re_loss(re_ref, re_src, gt_corr_map, neg_map, c2p, Ts)
        return fine_ri_loss, fine_re_loss

    def _fine_re_loss(self, ref_feats, src_feats, gt_corr_map, neg_map, c2p, Ts):
        device = ref_feats.device
        rotations = Ts[:, :3, :3]

        def take(mask, ref_f, src_f):
            corr_idx, ref_k, src_k = torch.nonzero(mask, as_tuple=True)
            if corr_idx.numel() == 0:
                return torch.zeros((), device=device, dtype=ref_f.dtype)
            r = ref_f[corr_idx, ref_k]
            s = src_f[corr_idx, src_k]
            R_per = rotations[c2p[corr_idx]]
            s_rot = torch.einsum('bck,blk->bcl', s, R_per)
            return r, s_rot

        pos = take(gt_corr_map, ref_feats, src_feats)
        if isinstance(pos, tuple):
            r, s_rot = pos
            pos_loss = torch.relu(torch.norm(s_rot - r, 2, -1) - self.positive_margin).mean()
        else:
            pos_loss = pos

        neg = take(neg_map, ref_feats, src_feats)
        if isinstance(neg, tuple):
            r, s_rot = neg
            neg_loss = torch.relu(self.negative_margin - torch.norm(s_rot - r, 2, -1)).mean()
        else:
            neg_loss = neg

        return pos_loss + neg_loss


# ---------------------------------------------------------------------------
# Overall loss
# ---------------------------------------------------------------------------

class OverallLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.coarse_loss_b = CoarseMatchingLossBatched(cfg)
        self.fine_loss_b   = FineMatchingLossBatched(cfg)
        self.weight_coarse_loss  = cfg.loss.weight_coarse_loss
        self.weight_fine_ri_loss = cfg.loss.weight_fine_ri_loss
        self.weight_fine_re_loss = cfg.loss.weight_fine_re_loss

    def forward(self, output_dict, data_dict):
        if 'batched' not in output_dict or output_dict.get('empty', False):
            # Empty / degenerate batch
            device = data_dict['features'].device if 'features' in data_dict else torch.device('cuda')
            zero = torch.zeros((), device=device)
            return {'loss': zero, 'c_loss': zero, 'f_ri_loss': zero, 'f_re_loss': zero}

        batched = output_dict['batched']
        c_loss = self.coarse_loss_b(batched)
        f_ri_loss, f_re_loss = self.fine_loss_b(batched)
        loss = (self.weight_coarse_loss * c_loss
                + self.weight_fine_ri_loss * f_ri_loss
                + self.weight_fine_re_loss * f_re_loss)
        return {
            'loss':      loss,
            'c_loss':    c_loss,
            'f_ri_loss': f_ri_loss,
            'f_re_loss': f_re_loss,
        }


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class Evaluator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.acceptance_overlap = cfg.eval.acceptance_overlap
        self.acceptance_radius = cfg.eval.acceptance_radius
        self.acceptance_rmse = cfg.eval.rmse_threshold
        self.feat_rre_threshold = cfg.eval.feat_rre_threshold

    @torch.no_grad()
    def forward(self, output_dict, data_dict):
        is_test = ('src_markers' in data_dict)
        if 'batched' not in output_dict or output_dict.get('empty', False):
            raise RuntimeError("OverallLoss received an empty batch (no output for point cloud).")

        if is_test:
            return self._test_per_pair(output_dict, data_dict)
        return self._batched(output_dict, data_dict)

    @torch.no_grad()
    def _batched(self, output_dict, data_dict):
        batched = output_dict['batched']
        B = output_dict['batch_size']
        device = batched['ref_feats_c_padded'].device

        # ---- PIR (coarse precision) — vectorized ----
        ref_mask_c = batched['ref_mask_c']
        src_mask_c = batched['src_mask_c']
        N_c = ref_mask_c.shape[1]
        M_c = src_mask_c.shape[1]
        gt_corr_list  = batched['gt_corr_per_pair']
        gt_ovlap_list = batched['gt_overlap_per_pair']

        sizes = [gi.shape[0] for gi in gt_corr_list]
        if sum(sizes) > 0:
            sizes_t = torch.tensor(sizes, device=device)
            all_corr = torch.cat(gt_corr_list, dim=0)
            all_ovl  = torch.cat(gt_ovlap_list, dim=0)
            keep = all_ovl > self.acceptance_overlap
            kept_batch = torch.repeat_interleave(torch.arange(B, device=device), sizes_t)[keep]
            kept_corr  = all_corr[keep]
            gt_dense = torch.zeros(B, N_c, M_c, device=device)
            if kept_corr.numel() > 0:
                gt_dense[kept_batch, kept_corr[:, 0], kept_corr[:, 1]] = 1.0
        else:
            gt_dense = torch.zeros(B, N_c, M_c, device=device)

        ref_corr_idx = batched['ref_corr_idx_pad']        # [B, P_max]
        src_corr_idx = batched['src_corr_idx_pad']
        pair_corr_masks = batched['pair_corr_masks']      # [B, P_max]
        b_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, ref_corr_idx.shape[1])
        hits = gt_dense[b_idx, ref_corr_idx, src_corr_idx] * pair_corr_masks.float()
        real_count = pair_corr_masks.sum(1).clamp(min=1).float()
        PIR = (hits.sum(1) / real_count).mean()

        # ---- RRE / RTE — fully batched ----
        T_b = batched['transforms']              # [B, 4, 4]
        est_b = batched['estimated_transforms']  # [B, 4, 4]
        rre_per_pair, rte_per_pair = isotropic_transform_error(T_b, est_b, reduction='none')   # [B], [B]
        RRE = rre_per_pair.mean()
        RTE = rte_per_pair.mean()

        # ---- IR / RMSE / RR — small per-pair work (variable corr counts) ----
        ref_corr_pts = batched['ref_corr_points_per_pair']
        src_corr_pts = batched['src_corr_points_per_pair']
        src_pts_per_pair = batched['src_points_per_pair']

        ir_list = []
        rmse_list = []
        recall_list = []
        for b in range(B):
            T = T_b[b]
            est = est_b[b]
            r_pts = ref_corr_pts[b]
            s_pts = src_corr_pts[b]
            if s_pts.shape[0] == 0:
                ir = torch.zeros((), device=device)
            else:
                s_pts_a = apply_transform(s_pts, T)
                d = torch.linalg.norm(r_pts - s_pts_a, dim=1)
                ir = (d < self.acceptance_radius).float().mean()
            ir_list.append(ir)
            sp = src_pts_per_pair[b]
            realign = torch.matmul(torch.inverse(T), est)
            r_sp = apply_transform(sp, realign)
            rmse = torch.linalg.norm(r_sp - sp, dim=1).mean()
            rmse_list.append(rmse)
            recall_list.append((rmse < self.acceptance_rmse).float())

        IR = torch.stack(ir_list).mean()
        rmse_stack = torch.stack(rmse_list)
        RMSE = rmse_stack.mean()
        RMSE_std = rmse_stack.std()
        RR = torch.stack(recall_list).mean()

        return {
            'PIR': PIR, 'IR': IR, 'RRE': RRE, 'RTE': RTE, 'RMSE': RMSE, 'RMSE_std': RMSE_std, 'RR': RR,
        }

    @torch.no_grad()
    def _test_per_pair(self, output_dict, data_dict):
        # Test mode (markers / s_c / t_c / m).
        batched = output_dict['batched']
        B = output_dict['batch_size']
        device = batched['ref_feats_c_padded'].device

        # PIR
        ref_mask_c = batched['ref_mask_c']
        src_mask_c = batched['src_mask_c']
        N_c = ref_mask_c.shape[1]
        M_c = src_mask_c.shape[1]
        gt_corr_list  = batched['gt_corr_per_pair']
        gt_ovlap_list = batched['gt_overlap_per_pair']
        sizes = [gi.shape[0] for gi in gt_corr_list]
        if sum(sizes) > 0:
            sizes_t = torch.tensor(sizes, device=device)
            all_corr = torch.cat(gt_corr_list, dim=0)
            all_ovl  = torch.cat(gt_ovlap_list, dim=0)
            keep = all_ovl > self.acceptance_overlap
            kept_batch = torch.repeat_interleave(torch.arange(B, device=device), sizes_t)[keep]
            kept_corr  = all_corr[keep]
            gt_dense = torch.zeros(B, N_c, M_c, device=device)
            if kept_corr.numel() > 0:
                gt_dense[kept_batch, kept_corr[:, 0], kept_corr[:, 1]] = 1.0
        else:
            gt_dense = torch.zeros(B, N_c, M_c, device=device)
        ref_corr_idx = batched['ref_corr_idx_pad']
        src_corr_idx = batched['src_corr_idx_pad']
        pair_corr_masks = batched['pair_corr_masks']
        b_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, ref_corr_idx.shape[1])
        hits = gt_dense[b_idx, ref_corr_idx, src_corr_idx] * pair_corr_masks.float()
        real_count = pair_corr_masks.sum(1).clamp(min=1).float()
        PIR = (hits.sum(1) / real_count).mean()

        T_b = batched['transforms']
        est_b = batched['estimated_transforms']

        s_c_list = data_dict['s_c'] if isinstance(data_dict['s_c'], list) else [data_dict['s_c']]
        t_c_list = data_dict['t_c'] if isinstance(data_dict['t_c'], list) else [data_dict['t_c']]
        m_list = data_dict['m'] if isinstance(data_dict['m'], list) else [data_dict['m']]
        src_markers_list = data_dict['src_markers'] if isinstance(data_dict['src_markers'], list) else [data_dict['src_markers']]
        tgt_markers_list = data_dict['tgt_markers'] if isinstance(data_dict['tgt_markers'], list) else [data_dict['tgt_markers']]

        ref_corr_pts = batched['ref_corr_points_per_pair']
        src_corr_pts = batched['src_corr_points_per_pair']
        src_pts_per_pair = batched['src_points_per_pair']
        if 'ir_final_per_pair' in batched:
            ir_final_per_pair = batched['ir_final_per_pair']
        else:
            ir_final_per_pair = torch.zeros(B, device=device)

        ir_list = []; rre_list = []; rte_list = []; rmse_list = []; re_markers_list = []; ir_final_list = []
        for b in range(B):
            T = T_b[b]; est = est_b[b]
            r_pts = ref_corr_pts[b]; s_pts = src_corr_pts[b]; sp = src_pts_per_pair[b]
            s_c = torch.as_tensor(s_c_list[b], dtype=torch.float32, device=device)
            t_c = torch.as_tensor(t_c_list[b], dtype=torch.float32, device=device)
            m   = m_list[b]
            R = est[:3, :3]
            t_norm = est[:3, 3]
            T_R = T[:3, :3]
            T_t_norm = T[:3, 3]
            T_real = torch.eye(4, dtype=T.dtype, device=device)
            T_real[:3, :3] = T_R
            T_real[:3, 3] = T_t_norm * m + t_c - torch.matmul(T_R, s_c)
            est_real = torch.eye(4, dtype=est.dtype, device=device)
            est_real[:3, :3] = R
            est_real[:3, 3] = t_norm * m + t_c - torch.matmul(R, s_c)

            rre, rte = isotropic_transform_error(T_real, est_real)
            rre_list.append(rre); rte_list.append(rte)

            if s_pts.shape[0] == 0:
                ir_list.append(torch.zeros((), device=device))
            else:
                s_pts_a = apply_transform(s_pts, T)
                d = torch.linalg.norm(r_pts - s_pts_a, dim=1)
                ir_list.append((d < self.acceptance_radius).float().mean())

            realign = torch.matmul(torch.inverse(T), est)
            realigned_src = apply_transform(sp, realign)
            real_world_src = sp * m + s_c
            real_world_realigned = realigned_src * m + s_c
            rmse = torch.linalg.norm(real_world_realigned - real_world_src, dim=1).mean()
            rmse_list.append(rmse)

            # Marker error
            src_markers = torch.as_tensor(src_markers_list[b], dtype=torch.float32, device=device) if not torch.is_tensor(src_markers_list[b]) else src_markers_list[b].float()
            tgt_markers = torch.as_tensor(tgt_markers_list[b], dtype=torch.float32, device=device) if not torch.is_tensor(tgt_markers_list[b]) else tgt_markers_list[b].float()
            est_R = est[:3, :3].float(); est_t = est[:3, 3].float()
            realigned_mkrs = torch.matmul(src_markers, est_R.transpose(-1, -2)) + est_t
            tgt_mkrs_real = tgt_markers.cpu().numpy() * float(m) + t_c.cpu().numpy()
            realigned_mkrs_real = realigned_mkrs.cpu().numpy() * float(m) + t_c.cpu().numpy()
            _, _, _, re_markers = cal_error(tgt_mkrs_real, realigned_mkrs_real, False)
            re_markers_list.append(re_markers)

            if ir_final_per_pair is not None:
                ir_final_list.append(ir_final_per_pair[b].cpu().numpy())

        return {
            'PIR': PIR,
            'IR':  torch.stack(ir_list).mean(),
            'IR_std': torch.stack(ir_list).std(),
            'RRE': torch.stack(rre_list).mean(),
            'RTE': torch.stack(rte_list).mean(),
            'RMSE': torch.stack(rmse_list).mean(),
            'RE_markers': float(np.mean(re_markers_list)),
            'IR_final': float(np.mean(ir_final_list)) if ir_final_list else 0.0,
        }