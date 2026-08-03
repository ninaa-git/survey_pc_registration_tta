import pdb
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from pareconv.modules.ops import apply_transform
from pareconv.modules.registration import WeightedProcrustes, solve_local_rotations

class BatchedHypothesisProposer(nn.Module):
    '''
    Modified from https://github.com/yaorz97/PARENet  
    - BatchedHypothesisProposer: batching the hypothesis proposer
    - Add of IR for Purge Gate
    '''
    def __init__(
        self,
        k: int,
        acceptance_radius: float,
        confidence_threshold: float = 0.025,
        num_hypotheses: int = 1000,
        num_refinement_steps: int = 5,
        compute_ir_final: bool = False,
    ):
        super().__init__()
        self.k = k
        self.acceptance_radius = acceptance_radius
        self.confidence_threshold = confidence_threshold
        self.num_hypotheses = num_hypotheses
        self.num_refinement_steps = num_refinement_steps
        self.procrustes = WeightedProcrustes(return_transform=True)
        self.compute_ir_final = compute_ir_final

    @torch.no_grad()
    def forward(
        self,
        ref_knn_points,        # [B, P_max, K, 3]
        src_knn_points,        # [B, P_max, K, 3]
        re_ref_knn_feats,      # [B, P_max, K, D, 3]
        re_src_knn_feats,      # [B, P_max, K, D, 3]
        ref_knn_masks,         # [B, P_max, K]
        src_knn_masks,         # [B, P_max, K]
        score_mat,             # [B, P_max, K, K]
        pair_corr_masks=None,  # [B, P_max] — False for padded correspondences
    ):
        B, P_max, K = ref_knn_masks.shape
        device = ref_knn_points.device

        if pair_corr_masks is None:
            pair_corr_masks = torch.ones(B, P_max, dtype=torch.bool, device=device)

        knn_mask_mat = ref_knn_masks.unsqueeze(3) & src_knn_masks.unsqueeze(2)   # [B, P, K, K]
        knn_mask_mat = knn_mask_mat & pair_corr_masks.unsqueeze(2).unsqueeze(3)

        # Source-side topk for voters
        src_topk_scores, src_topk_indices = score_mat.topk(k=self.k, dim=2)       # [B, P, k, K]
        src_score_mat = torch.zeros_like(score_mat)
        b_idx = torch.arange(B, device=device).view(B, 1, 1, 1).expand(-1, P_max, self.k, K)
        p_idx = torch.arange(P_max, device=device).view(1, P_max, 1, 1).expand(B, -1, self.k, K)
        c_idx = torch.arange(K, device=device).view(1, 1, 1, K).expand(B, P_max, self.k, -1)
        src_score_mat[b_idx, p_idx, src_topk_indices, c_idx] = src_topk_scores
        voter_corr_mat = (src_score_mat > self.confidence_threshold) & knn_mask_mat

        per_pair_budget = min(self.num_hypotheses, P_max * K * K)
        flat_scores = score_mat.reshape(B, -1)                                    # [B, P*K*K]
        flat_validity = knn_mask_mat.reshape(B, -1)
        masked_flat = flat_scores * flat_validity.float()
        h_scores, h_flat = masked_flat.topk(k=per_pair_budget, dim=1, largest=True)
        h_valid = h_scores > 0                                                    # [B, H_max]

        H_max = per_pair_budget
        h_p   = h_flat // (K * K)                                                 # [B, H_max]
        rem   = h_flat %  (K * K)
        h_ref = rem // K
        h_src = rem %  K

        # ---- 2. Gather hypothesis source data: feats and points ----
        b_idx_h = torch.arange(B, device=device).unsqueeze(1).expand(-1, H_max)   # [B, H_max]
        h_ref_points = ref_knn_points[b_idx_h, h_p, h_ref]                        # [B, H_max, 3]
        h_src_points = src_knn_points[b_idx_h, h_p, h_src]                        # [B, H_max, 3]
        h_ref_feats  = re_ref_knn_feats[b_idx_h, h_p, h_ref]                      # [B, H_max, D, 3]
        h_src_feats  = re_src_knn_feats[b_idx_h, h_p, h_src]                      # [B, H_max, D, 3]

        # ---- 3. Generate transformations from rotation-equivariant features ----
        BH = B * H_max
        flat_ref_feats = h_ref_feats.reshape(BH, *h_ref_feats.shape[2:])
        flat_src_feats = h_src_feats.reshape(BH, *h_src_feats.shape[2:])
        flat_ref_pts   = h_ref_points.reshape(BH, 3)
        flat_src_pts   = h_src_points.reshape(BH, 3)

        rotations = solve_local_rotations(flat_src_feats, flat_ref_feats)         # [BH, 3, 3]
        aligned_src = torch.einsum('bmn,bn->bm', rotations, flat_src_pts)         # [BH, 3]
        translations = flat_ref_pts - aligned_src                                 # [BH, 3]
        hypotheses_flat = torch.eye(4, device=device).unsqueeze(0).repeat(BH, 1, 1)
        hypotheses_flat[:, :3, :3] = rotations
        hypotheses_flat[:, :3, 3]  = translations
        hypotheses = hypotheses_flat.view(B, H_max, 4, 4)                         # [B, H_max, 4, 4]

        # ---- 4. Build voter set per pair, pad to V_max ----
        v_count_per_pair = voter_corr_mat.view(B, -1).sum(dim=1)                  # [B]
        V_max = int(v_count_per_pair.max().item())

        if V_max == 0:
            # Fallback: use hypothesis set as voters
            V_max = H_max
            v_ref_points = h_ref_points
            v_src_points = h_src_points
            v_scores     = h_scores
            v_valid      = h_valid
        else:
            v_flat_validity = voter_corr_mat.view(B, -1).float()                  # [B, P*K*K]
            v_flat_scores   = score_mat.view(B, -1) * v_flat_validity
            v_top_scores, v_top_flat = v_flat_scores.topk(k=V_max, dim=1, largest=True)
            v_valid = v_top_scores > 0                                            # [B, V_max]
            v_p   = v_top_flat // (K * K)
            v_rem = v_top_flat %  (K * K)
            v_ref = v_rem // K
            v_src = v_rem %  K
            b_idx_v = torch.arange(B, device=device).unsqueeze(1).expand(-1, V_max)
            v_ref_points = ref_knn_points[b_idx_v, v_p, v_ref]                    # [B, V_max, 3]
            v_src_points = src_knn_points[b_idx_v, v_p, v_src]
            v_scores     = v_top_scores                                            # [B, V_max]

        # ---- 5. Apply each pair's H hypotheses to that pair's V voters ----
        R = hypotheses[..., :3, :3]                                                # [B, H, 3, 3]
        t = hypotheses[..., :3, 3]                                                 # [B, H, 3]
        # [B, V, 3] -> rotate by [B, H, 3, 3] -> [B, H, V, 3]
        aligned = torch.einsum('bhij,bvj->bhvi', R, v_src_points) + t.unsqueeze(2) # [B, H, V, 3]
        residuals = torch.linalg.norm(v_ref_points.unsqueeze(1) - aligned, dim=-1)  # [B, H, V]

        # Inlier per voter (only count real voters and real hypotheses)
        inlier_mask = (residuals < self.acceptance_radius)                         # [B, H, V]
        inlier_mask = inlier_mask & v_valid.unsqueeze(1) & h_valid.unsqueeze(2)
        # Inlier ratio per (b, h): count over V_real (per pair)
        v_real_count = v_valid.sum(dim=1, keepdim=True).clamp(min=1).float()       # [B, 1]
        inlier_count = inlier_mask.sum(dim=2).float()                              # [B, H]
        ir = inlier_count / v_real_count                                           # [B, H]
        # Mask invalid hypotheses out of the argmax
        ir = ir.masked_fill(~h_valid, -1.0)
        best_h_idx = ir.argmax(dim=1)                                              # [B]
        # Gather best hypothesis's inlier mask per pair: [B, V]
        best_inlier_mask = inlier_mask[torch.arange(B, device=device), best_h_idx]
        # Per-pair best inlier ratio (used to detect "no inliers")
        best_inlier_count = inlier_count[torch.arange(B, device=device), best_h_idx]
        no_inliers_pair = best_inlier_count <= 0                                   # [B]

        # Adaptive radius fallback for pairs with zero inliers
        if no_inliers_pair.any():
            r_for_mean = residuals.masked_fill(~v_valid.unsqueeze(1), 0.0)
            r_means = r_for_mean.sum(dim=2) / v_real_count                         # [B, H]
            r_means = r_means.masked_fill(~h_valid, float('inf'))
            alt_best_h = r_means.argmin(dim=1)                                     # [B]
            for b in torch.nonzero(no_inliers_pair, as_tuple=True)[0].tolist():
                alt_h = alt_best_h[b].item()
                # k-th smallest residual at this hypothesis
                rs = residuals[b, alt_h][v_valid[b]]                               # real voters only
                if rs.numel() == 0:
                    continue
                k_use = min(self.k, rs.shape[0]) - 1
                adaptive_radius = rs.sort()[0][k_use]
                new_inliers = residuals[b, alt_h] < adaptive_radius
                new_inliers = new_inliers & v_valid[b]
                best_h_idx[b] = alt_h
                best_inlier_mask[b] = new_inliers

        # ---- 6. Per-pair refinement via WeightedProcrustes (already batched) ----
        cur_corr_scores = v_scores * best_inlier_mask.float()
        cur_corr_scores = cur_corr_scores * v_valid.float()

        estimated_transform = self.procrustes(v_src_points, v_ref_points, cur_corr_scores)  # [B, 4, 4]
        for _ in range(self.num_refinement_steps - 1):
            # Apply current transform per pair: [B, V, 3]
            aligned_src_voters = apply_transform(v_src_points, estimated_transform)    # [B, V, 3]
            res = torch.linalg.norm(v_ref_points - aligned_src_voters, dim=-1)         # [B, V]
            inlier = (res < self.acceptance_radius) & v_valid
            cur_corr_scores = v_scores * inlier.float() * v_valid.float()
            estimated_transform = self.procrustes(v_src_points, v_ref_points, cur_corr_scores)

        if self.compute_ir_final:
            aligned_final = apply_transform(v_src_points, estimated_transform)        # [B, V, 3]
            res_final = torch.linalg.norm(v_ref_points - aligned_final, dim=-1)       # [B, V]
            inl_final = (res_final < self.acceptance_radius) & v_valid                # [B, V]
            v_real = v_valid.sum(dim=1).clamp(min=1).float()                          # [B]
            n_inl = inl_final.sum(dim=1).float()                                      # [B]
            ir_final = n_inl / v_real

        # ---- 7. Build per-pair output (split lists) ----
        outputs = []
        for b in range(B):
            valid_h_b = h_valid[b]                                                  # [H_max]
            n_h_b = int(valid_h_b.sum().item())
            if n_h_b == 0:
                # No hypotheses
                D = re_ref_knn_feats.shape[-2]
                dummy_pts   = torch.zeros((0, 3), device=device)
                dummy_score = torch.zeros((0,), device=device)
                dummy_hyp   = torch.eye(4, device=device).unsqueeze(0)
                dummy_feat  = torch.zeros((0, D, 3), device=device)
                outputs.append((
                    dummy_pts, dummy_pts, dummy_score,
                    torch.eye(4, device=device),
                    dummy_hyp, dummy_feat, dummy_feat,
                ))
                continue
            global_ref_corr_points_b = h_ref_points[b][valid_h_b]
            global_src_corr_points_b = h_src_points[b][valid_h_b]
            global_corr_scores_b     = h_scores[b][valid_h_b]
            ref_corr_feats_b         = h_ref_feats[b][valid_h_b]
            src_corr_feats_b         = h_src_feats[b][valid_h_b]
            transformation_hypotheses_b = hypotheses[b][valid_h_b]
            estimated_transform_b    = estimated_transform[b]
            if self.compute_ir_final:
                ir_final_b = ir_final[b]
            else:
                ir_final_b = None

            outputs.append((
                global_ref_corr_points_b, global_src_corr_points_b, global_corr_scores_b,
                estimated_transform_b,
                transformation_hypotheses_b,
                ref_corr_feats_b, src_corr_feats_b,
                ir_final_b if self.compute_ir_final else None,
            ))

        return outputs


class HypothesisProposer(nn.Module):
    def __init__(
        self,
        k: int,
        acceptance_radius: float,
        confidence_threshold: float = 0.025,
        num_hypotheses: int = 1000,
        num_refinement_steps: int = 5,
    ):
        r"""Point Matching with Local-to-Global Registration.

        Args:
            k (int): top-k selection for matching.
            acceptance_radius (float): acceptance radius for LGR.
            confidence_threshold (float=0.05): ignore matches whose scores are below this threshold.
            correspondence_limit (optional[int]=None): maximal number of verification correspondences.
            num_refinement_steps (int=5): number of refinement steps.
        """
        super(HypothesisProposer, self).__init__()
        self.k = k
        self.acceptance_radius = acceptance_radius
        self.confidence_threshold = confidence_threshold
        self.num_hypotheses = num_hypotheses
        self.num_refinement_steps = num_refinement_steps
        self.procrustes = WeightedProcrustes(return_transform=True)

    def compute_correspondence_matrix(self, score_mat, ref_knn_masks, src_knn_masks):
        """Compute matching matrix and score matrix for each patch correspondence."""
        mask_mat = torch.logical_and(ref_knn_masks.unsqueeze(2), src_knn_masks.unsqueeze(1))

        batch_size, ref_length, src_length = score_mat.shape
        batch_indices = torch.arange(batch_size, device=score_mat.device)

        # correspondences from reference side
        ref_topk_scores, ref_topk_indices = score_mat.topk(k=self.k, dim=2)  # (B, N, K)
        ref_batch_indices = batch_indices.view(batch_size, 1, 1).expand(-1, ref_length, self.k)  # (B, N, K)
        ref_indices = torch.arange(ref_length, device=score_mat.device).view(1, ref_length, 1).expand(batch_size, -1, self.k)  # (B, N, K)
        ref_score_mat = torch.zeros_like(score_mat)
        ref_score_mat[ref_batch_indices, ref_indices, ref_topk_indices] = ref_topk_scores

        # correspondences from source side
        src_topk_scores, src_topk_indices = score_mat.topk(k=self.k, dim=1)  # (B, K, N)
        src_batch_indices = batch_indices.view(batch_size, 1, 1).expand(-1, self.k, src_length)  # (B, K, N)
        src_indices = torch.arange(src_length, device=score_mat.device).view(1, 1, src_length).expand(batch_size, self.k, -1)  # (B, K, N)
        src_score_mat = torch.zeros_like(score_mat)
        src_score_mat[src_batch_indices, src_topk_indices, src_indices] = src_topk_scores
        # correspondences used to vote for hypotheses
        voter_corr_mat = torch.logical_or(torch.gt(src_score_mat, self.confidence_threshold), torch.gt(src_score_mat, self.confidence_threshold))

        # top-k hypotheses used to generate hypotheses
        num_correspondences = min(self.num_hypotheses, mask_mat.sum())
        corr_scores, corr_indices = score_mat.reshape(-1).topk(k=num_correspondences, largest=True)
        batch_sel_indices = corr_indices // (score_mat.shape[1] * score_mat.shape[2])
        ref_sel_indices0 = corr_indices % (score_mat.shape[1] * score_mat.shape[2])
        ref_sel_indices = ref_sel_indices0 // (score_mat.shape[2])
        src_sel_indices = ref_sel_indices0 % score_mat.shape[1]
        corr_mat = torch.zeros_like(mask_mat, device=mask_mat.device)
        corr_mat[batch_sel_indices, ref_sel_indices, src_sel_indices] = True

        corr_mat = torch.logical_and(corr_mat, mask_mat)
        voter_corr_mat = torch.logical_and(voter_corr_mat, mask_mat)
        return corr_mat, voter_corr_mat


    def recompute_correspondence_scores(self, ref_corr_points, src_corr_points, corr_scores, estimated_transform):
        aligned_src_corr_points = apply_transform(src_corr_points, estimated_transform)
        corr_residuals = torch.linalg.norm(ref_corr_points - aligned_src_corr_points, dim=1)
        inlier_masks = torch.lt(corr_residuals, self.acceptance_radius)
        new_corr_scores = corr_scores * inlier_masks.float()
        return new_corr_scores


    def extract_fine_transforms(self, ref_corr_feats, src_corr_feats, ref_corr_points, src_corr_points):
        point_rotations = solve_local_rotations(src_corr_feats, ref_corr_feats) # B 3 3
        aligned_src_points = torch.einsum('bmn, bn->bm', point_rotations, src_corr_points)
        t = ref_corr_points - aligned_src_points
        transforms = torch.eye(4, device=ref_corr_feats.device).unsqueeze(0).repeat(t.shape[0], 1, 1)
        transforms[:, :3, :3] = point_rotations
        transforms[:, :3, 3] = t
        return transforms

    def feature_based_hypothesis_proposer(self, ref_knn_points,
                                          src_knn_points,
                                          ref_knn_feats,
                                          src_knn_feats,
                                          score_mat,
                                          corr_mat,
                                          voter_corr_mat):
        # extract dense correspondences
        batch_indices, ref_indices, src_indices = torch.nonzero(corr_mat, as_tuple=True)
        global_ref_corr_points = ref_knn_points[batch_indices, ref_indices]
        global_src_corr_points = src_knn_points[batch_indices, src_indices]
        global_corr_scores = score_mat[batch_indices, ref_indices, src_indices]
        ref_corr_feats, src_corr_feats = ref_knn_feats[batch_indices, ref_indices], src_knn_feats[batch_indices, src_indices]

        # guard 1: no hypotheses at all
        if global_ref_corr_points.shape[0] == 0:
            device = ref_knn_points.device
            estimated_transform = torch.eye(4, device=device)
            dummy_points = torch.zeros((0, 3), device=device)
            dummy_scores = torch.zeros((0,), device=device)
            dummy_hyp    = torch.eye(4, device=device).unsqueeze(0)
            dummy_feats  = torch.zeros((0, ref_knn_feats.shape[-2], ref_knn_feats.shape[-1]), device=device)
            return dummy_points, dummy_points, dummy_scores, estimated_transform, dummy_hyp, dummy_feats, dummy_feats


        # build verification set
        batch_v_indices, ref_v_indices, src_v_indices = torch.nonzero(voter_corr_mat, as_tuple=True)
        ref_corr_points = ref_knn_points[batch_v_indices, ref_v_indices]
        src_corr_points = src_knn_points[batch_v_indices, src_v_indices]
        corr_scores = score_mat[batch_v_indices, ref_v_indices, src_v_indices]

        # guard 2: no voters → fall back to hypothesis set itself
        if ref_corr_points.shape[0] == 0:
            ref_corr_points = global_ref_corr_points
            src_corr_points = global_src_corr_points
            corr_scores     = global_corr_scores

        # generate hypotheses using rotation-equivarint features
        transformation_hypotheses = self.extract_fine_transforms(ref_corr_feats, src_corr_feats, global_ref_corr_points, global_src_corr_points)

        # select the hypothesis with the most supporter
        batch_aligned_src_corr_points = apply_transform(src_corr_points.unsqueeze(0), transformation_hypotheses)
        batch_corr_residuals = torch.linalg.norm(ref_corr_points.unsqueeze(0) - batch_aligned_src_corr_points, dim=2)
        batch_inlier_masks = torch.lt(batch_corr_residuals, self.acceptance_radius)  # (P, N)

        if batch_corr_residuals.numel() == 0:
            batch_aligned_src_corr_points = apply_transform(global_src_corr_points.unsqueeze(0), transformation_hypotheses)
            batch_corr_residuals = torch.linalg.norm(global_ref_corr_points.unsqueeze(0) - batch_aligned_src_corr_points, dim=2)
 
        batch_inlier_masks = torch.lt(batch_corr_residuals, self.acceptance_radius)  # (P, N)

        if batch_inlier_masks.numel() == 0 or batch_inlier_masks.sum() == 0:
            sorted_residuals, _ = batch_corr_residuals.sort(dim=1)  # (P, N) ascending
            best_index = sorted_residuals.mean(dim=1).argmin()
            k = min(self.k, sorted_residuals.shape[1]) - 1
            adaptive_radius = sorted_residuals[best_index, k]
            batch_inlier_masks = torch.lt(batch_corr_residuals, adaptive_radius)

        ir = batch_inlier_masks.float().mean(dim=1)
        best_index = ir.argmax()
        cur_corr_scores = corr_scores * batch_inlier_masks[best_index].float()

        # global refinement
        estimated_transform = self.procrustes(src_corr_points, ref_corr_points, cur_corr_scores)
        for _ in range(self.num_refinement_steps - 1):
            cur_corr_scores = self.recompute_correspondence_scores(ref_corr_points, src_corr_points, corr_scores, estimated_transform)
            estimated_transform = self.procrustes(src_corr_points, ref_corr_points, cur_corr_scores)

        return global_ref_corr_points, global_src_corr_points, global_corr_scores, estimated_transform, transformation_hypotheses, ref_corr_feats, src_corr_feats,


    def forward(
        self,
        ref_knn_points,
        src_knn_points,
        re_ref_knn_feats,
        re_src_knn_feats,
        ref_knn_masks,
        src_knn_masks,
        score_mat,

    ):
        r"""Point Matching Module forward propagation with Local-to-Global registration.

        Args:
            ref_knn_points (Tensor): (N, K, 3)
            src_knn_points (Tensor): (N, K, 3)
            re_ref_knn_feats (Tensor): (N, K, D, 3)
            re_src_knn_feats (Tensor): (N, K, D, 3)
            ref_knn_masks (BoolTensor): (N, K)
            src_knn_masks (BoolTensor): (N, K)
            score_mat (Tensor): (B, K, K)
        Returns:
            ref_corr_points: (Tensor) (C, 3)
            src_corr_points: (Tensor) (C, 3)
            corr_scores: (Tensor) (C,)
            estimated_transform: (Tensor) (4, 4)
            hypotheses: (Tensor) (N, 4, 4)
            ref_corr_feats: (Tensor) (N, D, 3)
            src_corr_feats: (Tensor) (N, D, 3)
        """

        corr_mat, voter_corr_mat = self.compute_correspondence_matrix(score_mat, ref_knn_masks, src_knn_masks)  # (B, K, K)

        ref_corr_points, src_corr_points, corr_scores, estimated_transform, hypotheses, ref_corr_feats, src_corr_feats, \
            = self.feature_based_hypothesis_proposer(
            ref_knn_points,
            src_knn_points,
            re_ref_knn_feats,
            re_src_knn_feats,
            score_mat,
            corr_mat,
            voter_corr_mat
        )
        return ref_corr_points, src_corr_points, corr_scores, estimated_transform, hypotheses, ref_corr_feats, src_corr_feats,

