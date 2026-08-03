import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PointDualMatching(nn.Module):
    def __init__(self, dim):
        """point dual matching"""
        super(PointDualMatching, self).__init__()
        self.proj1 = nn.Linear(dim, dim, True)
        self.inf = np.inf

    def forward(self, ref_node_corr_knn_feats, src_node_corr_knn_feats, ref_node_corr_knn_scores, src_node_corr_knn_scores, ref_node_corr_knn_masks, src_node_corr_knn_masks, return_entropy=False):
        """point dual matching forward.
        Args:
            ref_node_corr_knn_feats: torch.Tensor (N, k, D)
            src_node_corr_knn_feats: torch.Tensor (N, k, D)
            ref_node_corr_knn_scores: torch.Tensor (N, k)
            src_node_corr_knn_scores: torch.Tensor (N, k)
            ref_node_corr_knn_masks: torch.bool (N, k)
            src_node_corr_knn_masks: torch.bool (N, k)

        Returns:
            matching_scores: torch.Tensor (N, k, k)
        """
        m_ref_feats, m_src_feats = self.proj1(ref_node_corr_knn_feats), self.proj1(src_node_corr_knn_feats)

        scores = torch.einsum('bnd,bmd->bnm', m_ref_feats, m_src_feats)  # (P, K, K)
        scores = scores / m_ref_feats.shape[-1] ** 0.5

        batch_size, num_row, num_col = scores.shape
        device = scores.device
        padded_row_masks = torch.zeros(size=(batch_size, num_row, num_col), device=device)
        padded_row_masks.masked_fill_(~src_node_corr_knn_masks[:, None, :], float('-inf'))

        padded_col_masks = torch.zeros(size=(batch_size, num_row, num_col), device=device)
        padded_col_masks.masked_fill_(~ref_node_corr_knn_masks[:, :, None], float('-inf'))
        matching_scores = F.softmax(scores + padded_row_masks, -1) * F.softmax(scores + padded_col_masks, 1)
        matching_scores = matching_scores * ref_node_corr_knn_scores[:, :, None] * src_node_corr_knn_scores[:, None, :]

        if return_entropy:
            p_row = F.softmax(scores + padded_row_masks, dim=-1)   # (N, k, k)
            p_col = F.softmax(scores + padded_col_masks, dim=1)    # (N, k, k)

            # Mask out invalid positions before summing entropy
            valid_src = src_node_corr_knn_masks[:, None, :].expand_as(p_row)  # (N, k, k) — src validity for each ref point
            valid_ref = ref_node_corr_knn_masks[:, :, None].expand_as(p_col)  # (N, k, k) — ref validity for each src point

            H_ref = -(p_row * torch.log(p_row + 1e-8) * valid_src).sum(dim=2)  # (N, k)
            H_src = -(p_col * torch.log(p_col + 1e-8) * valid_ref).sum(dim=1)  # (N, k)

            # Normalize by log(valid_k) so entropy is in [0, 1]
            valid_src_k = src_node_corr_knn_masks.float().sum(dim=1).clamp(min=2)  # (N,)
            valid_ref_k = ref_node_corr_knn_masks.float().sum(dim=1).clamp(min=2)  # (N,)

            H_ref_norm = H_ref / torch.log(valid_src_k)[:, None]  # (N, k)
            H_src_norm = H_src / torch.log(valid_ref_k)[:, None]  # (N, k)

            corr_norm_entropy = (H_ref_norm.mean() + H_src_norm.mean()) / 2
        if return_entropy :
            return matching_scores, corr_norm_entropy
        else :
            return matching_scores

    def __repr__(self):
        format_string = self.__class__.__name__
        return format_string
