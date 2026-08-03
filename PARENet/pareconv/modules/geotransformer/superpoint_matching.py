import torch
import torch.nn as nn
from pareconv.modules.ops import pairwise_distance


class BatchedSuperPointMatching(nn.Module):
    def __init__(self, num_correspondences, dual_normalization=True):
        super().__init__()
        self.num_correspondences = num_correspondences
        self.dual_normalization = dual_normalization

    def forward(self, ref_feats, src_feats, ref_masks, src_masks):
        """
        ref_feats:  [B, N_max, C]
        src_feats:  [B, M_max, C]
        ref_masks:  [B, N_max] bool, True for valid
        src_masks:  [B, M_max] bool, True for valid

        Returns:
            ref_corr_indices: [B, k] long, indices into N_max (padded)
            src_corr_indices: [B, k] long, indices into M_max
            corr_scores:      [B, k]
            corr_masks:       [B, k] bool, True for real (non-zero-score) entries
        """
        B, N, C = ref_feats.shape
        M = src_feats.shape[1]

        # Compute -dist^2 / e^... → matching scores in padded space
        neg_distances = -pairwise_distance(ref_feats, src_feats, normalized=True)  # [B, N, M]
        matching_scores = torch.exp(neg_distances)

        # Zero out padded rows/cols
        pair_mask = ref_masks.unsqueeze(2) & src_masks.unsqueeze(1)        # [B, N, M]
        matching_scores = matching_scores * pair_mask.float()

        if self.dual_normalization:
            # Per-pair dual normalization. clamp denominators to avoid 0/0.
            # Note: a fully-padded row has sum=0; clamping to eps would make
            # score/eps a huge value if the row had any tiny nonzero numerics.
            # We instead use 1.0 as denominator for those (the row is already
            # zeroed by pair_mask, so score/1=0 stays zero).
            ref_sum = matching_scores.sum(dim=2, keepdim=True)
            src_sum = matching_scores.sum(dim=1, keepdim=True)
            ref_sum = torch.where(ref_sum > 0, ref_sum, torch.ones_like(ref_sum))
            src_sum = torch.where(src_sum > 0, src_sum, torch.ones_like(src_sum))
            matching_scores = (matching_scores / ref_sum) * (matching_scores / src_sum)
            matching_scores = matching_scores * pair_mask.float()

        # Per-pair top-k from flattened (N*M) scores
        flat = matching_scores.view(B, -1)                                  # [B, N*M]
        k = min(self.num_correspondences, flat.shape[1])
        corr_scores, corr_indices = flat.topk(k=k, dim=1, largest=True)     # [B, k]

        ref_corr_indices = corr_indices // M                                # [B, k]
        src_corr_indices = corr_indices %  M                                # [B, k]

        # Real iff the picked score is > 0 (zero score == padded entry got top-k'd
        # because the pair had fewer than k real candidates)
        corr_masks = corr_scores > 0

        return ref_corr_indices, src_corr_indices, corr_scores, corr_masks

class SuperPointMatching(nn.Module):
    def __init__(self, num_correspondences, dual_normalization=True):
        super(SuperPointMatching, self).__init__()
        self.num_correspondences = num_correspondences
        self.dual_normalization = dual_normalization

    def forward(self, ref_feats, src_feats, ref_masks=None, src_masks=None):
        r"""Extract superpoint correspondences.

        Args:
            ref_feats (Tensor): features of the superpoints in reference point cloud.
            src_feats (Tensor): features of the superpoints in source point cloud.
            ref_masks (BoolTensor=None): masks of the superpoints in reference point cloud (False if empty).
            src_masks (BoolTensor=None): masks of the superpoints in source point cloud (False if empty).

        Returns:
            ref_corr_indices (LongTensor): indices of the corresponding superpoints in reference point cloud.
            src_corr_indices (LongTensor): indices of the corresponding superpoints in source point cloud.
            corr_scores (Tensor): scores of the correspondences.
        """
        if ref_masks is None:
            ref_masks = torch.ones(ref_feats.shape[0], dtype=torch.bool, device=ref_feats.device)
        if src_masks is None:
            src_masks = torch.ones(src_feats.shape[0], dtype=torch.bool, device=src_feats.device)

        ref_indices = torch.nonzero(ref_masks, as_tuple=True)[0]
        src_indices = torch.nonzero(src_masks, as_tuple=True)[0]
        ref_feats = ref_feats[ref_indices]
        src_feats = src_feats[src_indices]
        # select top-k proposals
        neg_distances = -pairwise_distance(ref_feats, src_feats, normalized=True)
        matching_scores = torch.exp(neg_distances)
        if self.dual_normalization:
            ref_matching_scores = matching_scores / matching_scores.sum(dim=1, keepdim=True)
            src_matching_scores = matching_scores / matching_scores.sum(dim=0, keepdim=True)
            matching_scores = ref_matching_scores * src_matching_scores
        
        num_correspondences = min(self.num_correspondences, matching_scores.numel())
        corr_scores, corr_indices = matching_scores.view(-1).topk(k=num_correspondences, largest=True)
        ref_sel_indices = corr_indices // matching_scores.shape[1]
        src_sel_indices = corr_indices % matching_scores.shape[1]
        ref_corr_indices = ref_indices[ref_sel_indices]
        src_corr_indices = src_indices[src_sel_indices]

        return ref_corr_indices, src_corr_indices, corr_scores

