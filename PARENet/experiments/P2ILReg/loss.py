import numpy as np
import torch
import torch.nn as nn

from pareconv.modules.loss import WeightedCircleLoss
from pareconv.modules.ops.transformation import apply_transform
from pareconv.modules.registration.metrics import isotropic_transform_error
from pareconv.modules.ops.pairwise_distance import pairwise_distance





def cal_error(gt, pred, print_error=False):
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
            raise RuntimeError("OverallLoss received an empty batch (no output for point cloud).")

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

class BinaryDiceLoss(nn.Module):
    r"""Dice loss of binary class
    From https://github.com/junzastar/Self-P2IR/
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        intersection = torch.sum(torch.mul(predict, target), dim=1)
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth
        dice = (2. * intersection + self.smooth) / den

        loss = 1 - dice

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))

class Evaluator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.acceptance_overlap = cfg.eval.acceptance_overlap
        self.acceptance_radius = cfg.eval.acceptance_radius
        self.acceptance_rmse = cfg.eval.rmse_threshold
        self.feat_rre_threshold = cfg.eval.feat_rre_threshold
        if cfg.data.dataset == 'real':
            from utils.mesh_render import MeshRender
            self.renderer = MeshRender(cfg)
        
    @torch.no_grad()
    def forward(self, output_dict, data_dict):
        is_test = ('liver_label' in data_dict)
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

        def _m(i):
            v = data_dict['m']
            if isinstance(v, (list, tuple)):
                return float(v[i])
            if torch.is_tensor(v) or isinstance(v, np.ndarray):
                vv = v.reshape(-1)
                return float(vv[i] if vv.shape[0] > 1 else vv[0])
            return float(v)  # B == 1, unwrapped scalar

        # ---- RRE (deg) / RTE (mm) — fully batched ----
        T_b   = batched['transforms']              # [B, 4, 4]
        est_b = batched['estimated_transforms']    # [B, 4, 4]
        num_transforms = T_b.shape[0]
        RRE_sum = T_b.new_zeros(())
        RTE_sum = T_b.new_zeros(())
        for i in range(num_transforms):
            gt_rot, pred_rot   = T_b[i, :3, :3], est_b[i, :3, :3]
            gt_trans, pred_trans = T_b[i, :3, 3], est_b[i, :3, 3]
            m = _m(i)

            trace = torch.clamp(torch.trace(gt_rot.t() @ pred_rot), -1.0, 3.0)
            RRE_sum = RRE_sum + torch.rad2deg(torch.arccos((trace - 1.0) / 2.0))   # degrees
            RTE_sum = RTE_sum + torch.linalg.norm(gt_trans - pred_trans) * m * 1000.0  # mm

        RRE = RRE_sum / num_transforms
        RTE = RTE_sum / num_transforms

        # ---- IR / RMSE (mm) / RR — per-pair (variable corr counts) ----
        ref_corr_pts     = batched['ref_corr_points_per_pair']
        src_corr_pts     = batched['src_corr_points_per_pair']
        src_pts_per_pair = batched['src_points_per_pair']
        ref_pts_per_pair = batched['ref_points_per_pair']
        if 'ir_final_per_pair' in batched:
            ir_final_per_pair = batched['ir_final_per_pair']
        else:
            ir_final_per_pair = None

        ir_list, rmse_list, recall_list, ir_final_list = [], [], [], []
        for b in range(B):
            T, est = T_b[b], est_b[b]
            r_pts, s_pts = ref_corr_pts[b], src_corr_pts[b]
            if s_pts.shape[0] == 0:
                ir = torch.zeros((), device=device)
            else:
                d = torch.linalg.norm(r_pts - apply_transform(s_pts, T), dim=1)
                ir = (d < self.acceptance_radius).float().mean()
            ir_list.append(ir)

            sp = src_pts_per_pair[b]
            realign = torch.matmul(torch.inverse(T), est)
            rmse_norm = torch.linalg.norm(apply_transform(sp, realign) - sp, dim=1).mean()  # normalized
            recall_list.append((rmse_norm < self.acceptance_rmse).float())  # threshold is normalized
            rmse_list.append(rmse_norm * _m(b) * 1000.0)                    # report in mm
            if ir_final_per_pair is not None:
                ir_final_list.append(ir_final_per_pair[b].cpu().numpy())

        IR = torch.stack(ir_list).mean()
        rmse_stack = torch.stack(rmse_list)
        RMSE = rmse_stack.mean()
        RMSE_std = rmse_stack.std()
        RR = torch.stack(recall_list).mean()
        IR_final = float(np.mean(ir_final_list)) if ir_final_list else 0.0
        return {
            'PIR': PIR, 'IR': IR, 'RRE': RRE, 'RTE': RTE, 'RMSE': RMSE, 'RMSE_std': RMSE_std, 'RR': RR,
            'IR_final': IR_final,
        }

    @torch.no_grad()
    def _test_per_pair(self, output_dict, data_dict):
        from pareconv.extensions.chamfer_distance.chamfer_distance import ChamferDistance
        batched = output_dict['batched']
        B = output_dict['batch_size']
        device = batched['ref_feats_c_padded'].device

        # PIR / IR / RRE / RTE / RMSE — same as batched train_val

        # PIR (vectorized)
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

        def pp(key, b):
            v = data_dict[key]
            if isinstance(v, (list, tuple)):
                return v[b]          # B > 1
            if B == 1:
                return v             # B == 1
            return v[b]              

        src_pts_per_pair = batched['src_points_per_pair'] 
        if 'ref_points_per_pair' in batched:
            tgt_pts_per_pair = batched['ref_points_per_pair']
        else:
            print("Using ref_corr_points_per_pair for tgt_pts_per_pair")
            tgt_pts_per_pair = batched['ref_corr_points_per_pair']
        if 'ir_final_per_pair' in batched:
            ir_final_per_pair = batched['ir_final_per_pair']
        else:
            ir_final_per_pair = None

        DSC = BinaryDiceLoss()
        CD = ChamferDistance()

        cd_list = []
        dice_list = []
        ir_final_list = []
        for b in range(B):
            est = est_b[b]                       # [4,4] normalized: src_u -> tgt_u
            sp = src_pts_per_pair[b]             # [Ns,3] normalized source
            tgt = tgt_pts_per_pair[b]            # [Nt,3] normalized target

            s_c = torch.as_tensor(pp('s_c', b), dtype=torch.float32, device=device).reshape(3)
            t_c = torch.as_tensor(pp('t_c', b), dtype=torch.float32, device=device).reshape(3)
            m   = float(pp('m', b))

            R = est[:3, :3]
            t_norm = est[:3, 3]
            est_real = torch.eye(4, dtype=est.dtype, device=device)
            est_real[:3, :3] = R
            est_real[:3, 3] = m * t_norm + t_c - torch.matmul(R, s_c)

            real_world_src = sp * m + s_c                                 # [Ns,3] physical source
            real_world_tgt = tgt * m + t_c                               # [Nt,3] physical target
            registered_src = apply_transform(real_world_src, est_real)   # [Ns,3] registered source

            # ---------------- Chamfer distance (mm) ----------------
            dist1, dist2 = CD(registered_src.unsqueeze(0), real_world_tgt.unsqueeze(0))
            cd_mm = (dist1.mean() + dist2.mean()) * 1000.0               # mm
            cd_list.append(cd_mm.item())

            # ---------------- silhouette Dice ----------------
            ocv2blender = torch.as_tensor(pp('ocv2blender', b), dtype=torch.float32, device=device)        # [3,3]
            bbx_center  = torch.as_tensor(pp('bbx_center', b), dtype=torch.float32, device=device).reshape(3)
            scale       = torch.as_tensor(pp('scale', b), dtype=torch.float32, device=device).reshape(-1)[0]
            cam_k       = torch.as_tensor(pp('cam_k', b), dtype=torch.float32, device=device)              # [3,3]
            img_size    = pp('img_size', b)                                                                # [2]
            liver_label = torch.as_tensor(pp('liver_label', b), dtype=torch.float32, device=device)

            verts = torch.matmul(registered_src, ocv2blender.t())        # [Ns,3]
            verts = (verts - bbx_center) * (1.0 / scale) + bbx_center
            verts = verts.unsqueeze(0)                                   # [1,Ns,3] (batched)

            Pose = {
                'rot':   torch.eye(3, device=device).unsqueeze(0),       # [1,3,3]
                'trans': torch.zeros(3, 1, device=device).unsqueeze(0),  # [1,3,1]
            }
            rgbs = torch.zeros_like(verts)                               
            mask, _ = self.renderer(verts, Pose, rgbs, [img_size], cam_k.unsqueeze(0), None)

            label = (liver_label > 0).float()
            if label.dim() == 2:
                label = label.unsqueeze(0)                               # [1,H,W]
            elif label.dim() == 3 and label.shape[-1] in (3, 4):
                label = (label.sum(dim=-1) > 0).float().unsqueeze(0)
            dice = (1.0 - DSC(mask, label)) * 100.0                      # percentage
            dice_list.append(dice.item())
            if ir_final_per_pair is not None:
                ir_final_list.append(ir_final_per_pair[b].cpu().numpy())
        return {
            'CD': float(np.mean(cd_list)) if cd_list else 0.0,
            'DICE': float(np.mean(dice_list)) if dice_list else 0.0,
            'IR_final': float(np.mean(ir_final_list)) if ir_final_list else 0.0,
        }