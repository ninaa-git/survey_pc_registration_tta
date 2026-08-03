import torch
import torch.nn as nn

from pareconv.modules.ops import pairwise_distance
from skimage.filters import threshold_otsu

class SuperPointMatching_k(nn.Module):
    def __init__(self, num_correspondences, dual_normalization=True):
        super(SuperPointMatching_k, self).__init__()
        self.num_correspondences = num_correspondences
        self.dual_normalization = dual_normalization

    def forward(self, ref_feats, src_feats, ref_masks=None, src_masks=None,  return_entropy=False):
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
            ref_masks = torch.ones(size=(ref_feats.shape[0],), dtype=torch.bool).cuda()
        if src_masks is None:
            src_masks = torch.ones(size=(src_feats.shape[0],), dtype=torch.bool).cuda()
        # remove empty patch
        #n_src_total = src_masks.shape[0]
        #n_ref_total = ref_masks.shape[0]
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

        if return_entropy:
            neg_distances_clamped = neg_distances.clamp(min=-50, max=0)

            ref_corr_proba = torch.softmax(neg_distances_clamped, dim=1)
            ref_corr_entropy = -(ref_corr_proba * torch.log(ref_corr_proba.clamp(min=1e-8))).sum(dim=1)
            
            n_src = neg_distances.shape[1] #n_src_total
            if n_src > 1:
                ref_corr_norm_entropy = ref_corr_entropy / torch.log(
                    torch.tensor(n_src, dtype=torch.float, device=ref_corr_entropy.device)
                )
            else:
                ref_corr_norm_entropy = torch.zeros_like(ref_corr_entropy)

            src_corr_proba = torch.softmax(neg_distances_clamped, dim=0)
            src_corr_entropy = -(src_corr_proba * torch.log(src_corr_proba.clamp(min=1e-8))).sum(dim=0)
            n_ref =  neg_distances.shape[0] #n_src_total
            if n_ref > 1:
                src_corr_norm_entropy = src_corr_entropy / torch.log(
                    torch.tensor(n_ref, dtype=torch.float, device=src_corr_entropy.device)
                )
            else:
                src_corr_norm_entropy = torch.zeros_like(src_corr_entropy)

            ref_valid = ref_corr_norm_entropy[~torch.isnan(ref_corr_norm_entropy)]
            src_valid = src_corr_norm_entropy[~torch.isnan(src_corr_norm_entropy)]
            ref_mean = ref_valid.mean() if ref_valid.numel() > 0 else torch.tensor(0.0, device=neg_distances.device)
            src_mean = src_valid.mean() if src_valid.numel() > 0 else torch.tensor(0.0, device=neg_distances.device)
            cor_norm_entropy = (ref_mean + src_mean) / 2
        
        ## beginning otsu
        scores_flat = matching_scores.view(-1)
        scores_np = scores_flat.detach().cpu().numpy()
        try:
            otsu_thresh = threshold_otsu(scores_np)
        except Exception:
            otsu_thresh = float(scores_np.mean())
        above_thresh_mask = scores_flat >= otsu_thresh
        if above_thresh_mask.sum() == 0:
            above_thresh_mask[scores_flat.argmax()] = True
        #print(f"[Otsu] threshold={otsu_thresh:.4f} | correspondences selected={int(above_thresh_mask.sum())}/{len(scores_flat)}")
        ## end otsu

        #num_correspondences = min(self.num_correspondences, matching_scores.numel())
        num_correspondences = min(int(above_thresh_mask.sum()), matching_scores.numel())
        corr_scores, corr_indices = matching_scores.view(-1).topk(k=num_correspondences, largest=True)
        ref_sel_indices = corr_indices // matching_scores.shape[1]
        src_sel_indices = corr_indices % matching_scores.shape[1]
        ref_corr_indices = ref_indices[ref_sel_indices]
        src_corr_indices = src_indices[src_sel_indices]

        if return_entropy :
            return ref_corr_indices, src_corr_indices, corr_scores, cor_norm_entropy
        else : 
            return ref_corr_indices, src_corr_indices, corr_scores, scores_flat
