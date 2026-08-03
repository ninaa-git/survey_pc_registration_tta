import pdb

import numpy as np
import torch
import torch.nn as nn

from pareconv.modules.ops import pairwise_distance
from pareconv.modules.transformer import SinusoidalPositionalEmbedding, RPEConditionalTransformer_LN

from pareconv.modules.geotransformer.geotransformer import GeometricStructureEmbedding

class GeometricTransformer_LN(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim,
        num_heads,
        blocks,
        sigma_d,
        sigma_a,
        angle_k,
        dropout=None,
        activation_fn='ReLU',
        reduction_a='max',
        ctx=None,
    ):
        r"""Geometric Transformer (GeoTransformer).

        Args:
            input_dim: input feature dimension
            output_dim: output feature dimension
            hidden_dim: hidden feature dimension
            num_heads: number of head in transformer
            blocks: list of 'self' or 'cross'
            sigma_d: temperature of distance
            sigma_a: temperature of angles
            angle_k: number of nearest neighbors for angular embedding
            activation_fn: activation function
            reduction_a: reduction mode of angular embedding ['max', 'mean']
        """
        super(GeometricTransformer_LN, self).__init__()

        self.embedding = GeometricStructureEmbedding(hidden_dim, sigma_d, sigma_a, angle_k, reduction_a=reduction_a)

        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.ctx = ctx
        self.transformer = RPEConditionalTransformer_LN(
            blocks, hidden_dim, num_heads, dropout=dropout, activation_fn=activation_fn, return_attention_scores=True, parallel=False, ctx=ctx,
        )
        self.out_proj = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        ref_points,
        src_points,
        ref_feats,
        src_feats,
        ref_masks=None,
        src_masks=None,
    ):
        r"""Geometric Transformer

        Args:
            ref_points (Tensor): (B, N, 3)
            src_points (Tensor): (B, M, 3)
            ref_feats (Tensor): (B, N, C)
            src_feats (Tensor): (B, M, C)
            ref_masks (Optional[BoolTensor]): (B, N)
            src_masks (Optional[BoolTensor]): (B, M)

        Returns:
            ref_feats: torch.Tensor (B, N, C)
            src_feats: torch.Tensor (B, M, C)
        """
        ref_embeddings = self.embedding(ref_points, masks=ref_masks)
        src_embeddings = self.embedding(src_points, masks=src_masks)
        ref_feats = self.in_proj(ref_feats)
        src_feats = self.in_proj(src_feats)

        ref_pad_masks = (~ref_masks) if ref_masks is not None else None
        src_pad_masks = (~src_masks) if src_masks is not None else None


        ref_feats, src_feats, scores_list = self.transformer(
            ref_feats,
            src_feats,
            ref_embeddings,
            src_embeddings,
            masks0=ref_pad_masks,
            masks1=src_pad_masks,
        )

        ref_feats = self.out_proj(ref_feats)
        src_feats = self.out_proj(src_feats)

        return ref_feats, src_feats, scores_list
