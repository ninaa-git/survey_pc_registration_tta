import pdb

import numpy as np
import torch
import torch.nn as nn

from pareconv.modules.ops import pairwise_distance
from pareconv.modules.transformer import SinusoidalPositionalEmbedding, RPEConditionalTransformer


class GeometricStructureEmbedding(nn.Module):
    def __init__(self, hidden_dim, sigma_d, sigma_a, angle_k, reduction_a='max'):
        super(GeometricStructureEmbedding, self).__init__()
        self.sigma_d = sigma_d
        self.sigma_a = sigma_a
        self.factor_a = 180.0 / (self.sigma_a * np.pi)
        self.angle_k = angle_k

        self.embedding = SinusoidalPositionalEmbedding(hidden_dim)
        self.proj_d = nn.Linear(hidden_dim, hidden_dim)
        self.proj_a = nn.Linear(hidden_dim, hidden_dim)

        self.reduction_a = reduction_a
        if self.reduction_a not in ['max', 'mean']:
            raise ValueError(f'Unsupported reduction mode: {self.reduction_a}.')

    @torch.no_grad()
    def get_embedding_indices(self, points, masks=None):
        r"""Compute the indices of pair-wise distance embedding and triplet-wise angular embedding.

        Args:
            points: torch.Tensor (B, N, 3), input point cloud
            masks: Optional[BoolTensor] (B, N), True for valid points and False
                for padded slots. When provided, padded supports are excluded
                from the KNN angular reference so they cannot be selected as
                a real point's neighbor.

        Returns:
            d_indices: torch.FloatTensor (B, N, N), distance embedding indices
            a_indices: torch.FloatTensor (B, N, N, k), angular embedding indices
        """
        batch_size, num_point, _ = points.shape

        dist_map = torch.sqrt(pairwise_distance(points, points))  # (B, N, N)

        # Block padded columns from being chosen as KNN. We use a large finite
        # value rather than +inf so downstream embedding indices remain finite
        # even for padded query rows (whose outputs are masked in attention).
        if masks is not None:
            invalid_col = (~masks).unsqueeze(1)              # (B, 1, N)
            dist_map = dist_map.masked_fill(invalid_col, 1e6)

        d_indices = dist_map / self.sigma_d

        k = min(self.angle_k, num_point - 1)

        knn_indices = dist_map.topk(k=k + 1, dim=2, largest=False)[1][:, :, 1:]  # (B, N, k)
        knn_indices = knn_indices.unsqueeze(3).expand(batch_size, num_point, k, 3)  # (B, N, k, 3)
        expanded_points = points.unsqueeze(1).expand(batch_size, num_point, num_point, 3)  # (B, N, N, 3)
        knn_points = torch.gather(expanded_points, dim=2, index=knn_indices)  # (B, N, k, 3)
        ref_vectors = knn_points - points.unsqueeze(2)  # (B, N, k, 3)

        anc_vectors = points.unsqueeze(1) - points.unsqueeze(2)  # (B, N, N, 3)
        ref_vectors = ref_vectors.unsqueeze(2).expand(batch_size, num_point, num_point, k, 3)  # (B, N, N, k, 3)
        anc_vectors = anc_vectors.unsqueeze(3).expand(batch_size, num_point, num_point, k, 3)  # (B, N, N, k, 3)
        sin_values = torch.linalg.norm(torch.cross(ref_vectors, anc_vectors, dim=-1), dim=-1)  # (B, N, N, k)
        cos_values = torch.sum(ref_vectors * anc_vectors, dim=-1)  # (B, N, N, k)
        angles = torch.atan2(sin_values, cos_values)  # (B, N, N, k)
        a_indices = angles * self.factor_a

        return d_indices, a_indices

    def forward(self, points, masks=None):
        d_indices, a_indices = self.get_embedding_indices(points, masks=masks)

        d_embeddings = self.embedding(d_indices)
        d_embeddings = self.proj_d(d_embeddings)

        if a_indices.shape[3] == 0:
            # No valid neighbours (degenerate cloud). Skip angular embedding.
            a_embeddings = torch.zeros_like(d_embeddings)
        else:
            a_embeddings = self.embedding(a_indices)
            a_embeddings = self.proj_a(a_embeddings)
            if self.reduction_a == 'max':
                a_embeddings = a_embeddings.max(dim=3)[0]
            else:
                a_embeddings = a_embeddings.mean(dim=3)
        embeddings = d_embeddings + a_embeddings

        return embeddings


class GeometricTransformer_PEA_tr(nn.Module):
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
        super(GeometricTransformer_PEA_tr, self).__init__()

        self.embedding = GeometricStructureEmbedding(hidden_dim, sigma_d, sigma_a, angle_k, reduction_a=reduction_a)

        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.transformer = RPEConditionalTransformer(
            blocks, hidden_dim, num_heads, dropout=dropout, activation_fn=activation_fn, return_attention_scores=True, parallel=False
        )
        self.out_proj = nn.Linear(hidden_dim, output_dim)

        # ── PEA alignment state (single stage, applied after in_proj) ──────
        # The statistics live in the hidden_dim space (post linear projection)
        # rather than the input_dim space, which makes them much smaller to
        # estimate and better conditioned (hidden_dim**2 instead of
        # input_dim**2 covariance entries).
        self._align_stats    = None
        self._align_weight   = 1.0
        self._align_momentum = 0.1

    def set_alignment_stats(self, src_stats: dict,
                            weight: float = 1.0, momentum: float = 0.1):
        """Install PEA source statistics for the post-in_proj features.

        Args:
            src_stats: {'mean': [H], 'cov_sqrt': [H, H], ...} computed by
                generate_stats.py on in_proj outputs of clean source ref
                features (H = hidden_dim).
            weight: blending weight w of F' = (1 - w) F + w Y.
            momentum: EMA momentum m; the effective step follows the
                bias-corrected schedule step = max(n_b / n, m) (incremental
                sample mean for the first batches, constant-momentum EMA
                afterwards).
        """
        dev = self.in_proj.weight.device

        def _to_float(x):
            if not isinstance(x, torch.Tensor):
                x = torch.from_numpy(x)
            return x.float().to(dev)

        expected = self.in_proj.out_features
        got = int(np.asarray(src_stats['mean']).shape[0])
        if got != expected:
            raise ValueError(
                f'PEA stats dim mismatch: got {got}, but in_proj outputs '
                f'{expected} channels. These stats were probably computed at '
                f'the old location (backbone output); re-run '
                f'generate_stats.py with --force.')

        self._align_stats = {
            'src_mean':        _to_float(src_stats['mean']),
            'src_cov_sqrt':    _to_float(src_stats['cov_sqrt']),
            'tgt_mean':        None,
            'tgt_cov':         None,
            'tgt_cov_invsqrt': None,
            'n':               0.0,   # clouds seen; Welford step = 1/n, floored at m
            'tgt_2nd':         None,
        }
        self._align_weight   = float(weight)
        self._align_momentum = float(momentum)

    @staticmethod
    def _wct_align(feats: torch.Tensor,
                   src_mean: torch.Tensor,
                   src_cov_sqrt: torch.Tensor,
                   tgt_mean: torch.Tensor,
                   tgt_cov_invsqrt: torch.Tensor,
                   weight: float) -> torch.Tensor:
        """
        Whitening-Coloring Transform for a single feature matrix.

            Y = (F - μ_t) Σ_t^{-1/2} Σ_s^{1/2} + μ_s
            F' = (1 - w) F + w Y

        Args:
            feats           : [N, C]  target features to align
            src_mean        : [C]     source mean  μ_s
            src_cov_sqrt    : [C, C]  Σ_s^{1/2}
            tgt_mean        : [C]     target mean  μ_t  (estimated from feats)
            tgt_cov_invsqrt : [C, C]  Σ_t^{-1/2}       (estimated from feats)
            weight          : scalar  blending weight w

        Returns:
            aligned features [N, C]
        """
        whitened = (feats - tgt_mean.unsqueeze(0)) @ tgt_cov_invsqrt   # [N, C]
        colored  = whitened @ src_cov_sqrt + src_mean.unsqueeze(0)     # [N, C]
        return (1.0 - weight) * feats + weight * colored

    def _pea_align_ref(self, ref_feats, ref_masks):
        """PEA alignment of the ref (intraoperative / corrupted) stream,
        applied to the post-in_proj features. Target statistics are
        estimated from the valid ref tokens of the incoming stream and
        tracked with the bias-corrected EMA; the src stream is never
        touched. Runs in fp32 even under autocast (eigh stability)."""
        s = self._align_stats
        in_dtype = ref_feats.dtype
        with torch.amp.autocast('cuda', enabled=False):
            feats = ref_feats.float()
            B_sz, ref_N, C = feats.shape

            # ── Compute batch statistics from valid ref features ──────────
            if ref_masks is None:
                mask_f = feats.new_ones(B_sz, ref_N, 1)
            else:
                mask_f = ref_masks.float().unsqueeze(-1)   # [B, ref_N, 1]
            n_valid = mask_f.sum()
            if float(n_valid) < 2.0:
                return ref_feats

            flat   = (feats * mask_f).reshape(-1, C)
            mean_b = flat.sum(dim=0) / n_valid              # E[x]    valid tokens, (d,)
            outer  = flat.T @ flat                          # Σ x x^T valid tokens, (d, d)

            # ── EMA update (step = max(n_b / n, m)) ───────────────────────
            n_b     = float(n_valid)
            outer_b = outer / n_valid

            m = self._align_momentum
            s['n'] = s['n'] + n_b
            if s['tgt_mean'] is None:
                s['tgt_mean'] = mean_b
                s['tgt_2nd']  = outer_b
            else:
                step          = max(n_b / s['n'], m)
                s['tgt_mean'] = (1.0 - step) * s['tgt_mean'] + step * mean_b
                s['tgt_2nd']  = (1.0 - step) * s['tgt_2nd']  + step * outer_b

            cov = s['tgt_2nd'] - torch.outer(s['tgt_mean'], s['tgt_mean'])
            s['tgt_cov'] = 0.5 * (cov + cov.t())

            # ── Recompute cov_invsqrt from updated Σ ──────────────────────
            cov = s['tgt_cov']
            cov = (cov + cov.T) / 2.0                            # symmetrise  [C, C]
            eigvals, eigvecs = torch.linalg.eigh(cov)            # [C], [C, C]
            eigvals_pos = eigvals.clamp(min=1e-6)
            tgt_cov_invsqrt = eigvecs * (1.0 / eigvals_pos.sqrt()).unsqueeze(0) @ eigvecs.T
            s['tgt_cov_invsqrt'] = tgt_cov_invsqrt

            # ── Apply WCT alignment to the ref stream ─────────────────────
            aligned = self._wct_align(
                feats.reshape(-1, C),
                src_mean=s['src_mean'],
                src_cov_sqrt=s['src_cov_sqrt'],
                tgt_mean=s['tgt_mean'],
                tgt_cov_invsqrt=tgt_cov_invsqrt,
                weight=self._align_weight,
            ).reshape(B_sz, ref_N, C)

        return aligned.to(in_dtype)

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
            ref_masks (Optional[BoolTensor]): (B, N) — True for valid points
            src_masks (Optional[BoolTensor]): (B, M) — True for valid points

        Returns:
            ref_feats: torch.Tensor (B, N, C)
            src_feats: torch.Tensor (B, M, C)
        """

        ref_embeddings = self.embedding(ref_points, masks=ref_masks)
        src_embeddings = self.embedding(src_points, masks=src_masks)
        ref_feats = self.in_proj(ref_feats)
        src_feats = self.in_proj(src_feats)

        # ── PEA alignment (post linear projection, ref stream only) ────────
        # The intraoperative (ref) features are realigned toward the source
        # statistics right where the attention consumes them; the clean
        # preoperative (src) stream passes through untouched.
        if self._align_stats is not None and not self.training:
            ref_feats = self._pea_align_ref(ref_feats, ref_masks)

        # Convention conversion: our masks are True=valid (validity masks).
        # The underlying RPE/vanilla attention expects True=ignore (padding masks).
        # Invert at the boundary.
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