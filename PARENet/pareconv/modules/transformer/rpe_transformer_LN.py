import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from pareconv.modules.layers import build_dropout_layer, build_act_layer


class TTAContext:
    def __init__(self):
        self._processing_corrupted = False   # False = ref stream, True = src stream


class AdaptiveLayerNorm(nn.LayerNorm):
    '''
    Modified from https://github.com/yaorz97/PARENet  
    '''
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True, ctx=None, momentum=0.01):
        super().__init__(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)
        self.ctx = ctx
        self.momentum = momentum
        # Per-stream source stats — post-affine y on clean source data, shape (d,)
        self.mu_source_ref     = None   # E[y]   ref stream
        self.sigma2_source_ref = None   # Var[y] ref stream
        self.mu_source_src     = None   # E[y]   src stream
        self.sigma2_source_src = None   # Var[y] src stream
        # Per-stream target stats — pre-affine h on corrupted test data, shape (d,)
        self.mu_target_ref     = None   # running E[h]    ref, (d,)
        self.m2_target_ref     = None   # running E[h^2]  ref, (d,)
        self.sigma2_target_ref = None   # derived Var[h]  ref, (d,)
        self.n_target_ref      = 0.0    # cumulative token count, ref
        self.mu_target_src     = None   # running E[h]    src, (d,)
        self.m2_target_src     = None   # running E[h^2]  src, (d,)
        self.sigma2_target_src = None   # derived Var[h]  src, (d,)
        self.n_target_src      = 0.0    # cumulative token count, src

    def forward(self, x):
        ctx = self.ctx
        if ctx is None:
            return super().forward(x)

        # ── Select per-stream stats for the stream currently being processed ──
        if ctx._processing_corrupted:            # ref stream (corrupted cloud)
            mu_source     = self.mu_source_ref
            sigma2_source = self.sigma2_source_ref
            mu_target     = self.mu_target_ref
            m2_target     = self.m2_target_ref
            n_target      = self.n_target_ref
        else:                                     # src stream (clean cloud)
            mu_source     = self.mu_source_src
            sigma2_source = self.sigma2_source_src
            mu_target     = self.mu_target_src
            m2_target     = self.m2_target_src
            n_target      = self.n_target_src

        # ── Online target-stats update ────────────────────────────────────────
        with torch.no_grad():
            x_flat   = x.detach().float().reshape(-1, x.shape[-1])      # (N, d)
            mean_tok = x_flat.mean(dim=-1, keepdim=True)                # (N, 1)
            var_tok  = x_flat.var(dim=-1, keepdim=True,
                                  unbiased=False).clamp(min=self.eps)   # (N, 1)
            h        = (x_flat - mean_tok) / var_tok.sqrt()             # (N, d)

            n_b    = float(h.shape[0])                                  
            mean_b = h.mean(dim=0)                                      # E[h]   batch, (d,)
            sq_b   = h.pow(2).mean(dim=0)                               # E[h^2] batch, (d,)

            alpha    = self.momentum
            n_target = n_target + n_b
            if mu_target is None:
                mu_target = mean_b                                      # first batch: step==1
                m2_target = sq_b
            else:
                step      = max(n_b / n_target, alpha)
                mu_target = (1.0 - step) * mu_target + step * mean_b
                m2_target = (1.0 - step) * m2_target + step * sq_b

            sigma2_target = (m2_target - mu_target.pow(2)).clamp(min=0) # (d,)

            if ctx._processing_corrupted:
                self.n_target_ref      = n_target
                self.mu_target_ref     = mu_target
                self.m2_target_ref     = m2_target
                self.sigma2_target_ref = sigma2_target
            else:
                self.n_target_src      = n_target
                self.mu_target_src     = mu_target
                self.m2_target_src     = m2_target
                self.sigma2_target_src = sigma2_target

        if mu_source is None or mu_target is None:
            return super().forward(x)

        dev   = x.device
        dtype = x.dtype
        delta = 1e-6
        tau   = 1e-8

        y = super().forward(x)                                          

        sigma2_s = torch.as_tensor(sigma2_source, dtype=dtype, device=dev).clamp(min=0)
        sigma2_t = torch.as_tensor(sigma2_target, dtype=dtype, device=dev).clamp(min=0)
        mu_s     = torch.as_tensor(mu_source,     dtype=dtype, device=dev)
        mu_t     = torch.as_tensor(mu_target,     dtype=dtype, device=dev)

        gamma_star = self.weight.sign() * (sigma2_s / (sigma2_t + delta)).sqrt()
        beta_star  = mu_s - gamma_star * mu_t

        safe_weight = torch.where(self.weight.abs() < tau,
                                  torch.ones_like(self.weight), self.weight)
        gamma_ratio = gamma_star / safe_weight
        y_adapt = (y - self.bias) * gamma_ratio + beta_star
        return y_adapt


# ---------------------------------------------------------------------------
# RPEMultiHeadAttention
# ---------------------------------------------------------------------------

class RPEMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=None):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f'`d_model` ({d_model}) must be a multiple of `num_heads` ({num_heads}).')
        self.d_model          = d_model
        self.num_heads        = num_heads
        self.d_model_per_head = d_model // num_heads
        self.proj_q  = nn.Linear(d_model, d_model)
        self.proj_q1 = nn.Linear(d_model, d_model)
        self.proj_k  = nn.Linear(d_model, d_model)
        self.proj_v  = nn.Linear(d_model, d_model)
        self.proj_p  = nn.Linear(d_model, d_model)
        self.dropout = build_dropout_layer(dropout)

    def forward(self, input_q, input_k, input_v, embed_qk,
                key_weights=None, key_masks=None, attention_factors=None):
        q  = rearrange(self.proj_q(input_q),  'b n (h c) -> b h n c', h=self.num_heads)
        q1 = rearrange(self.proj_q1(input_q), 'b n (h c) -> b h n c', h=self.num_heads)
        k  = rearrange(self.proj_k(input_k),  'b m (h c) -> b h m c', h=self.num_heads)
        v  = rearrange(self.proj_v(input_v),  'b m (h c) -> b h m c', h=self.num_heads)
        p  = rearrange(self.proj_p(embed_qk), 'b n m (h c) -> b h n m c', h=self.num_heads)

        scores_p = torch.einsum('bhnc,bhnmc->bhnm', q1, p)
        scores_e = torch.einsum('bhnc,bhmc->bhnm',  q,  k)
        scores   = (scores_e + scores_p) / self.d_model_per_head ** 0.5

        if attention_factors is not None:
            scores = attention_factors.unsqueeze(1) * scores
        if key_weights is not None:
            scores = scores * key_weights.unsqueeze(1).unsqueeze(1)
        if key_masks is not None:
            scores = scores.masked_fill(key_masks.unsqueeze(1).unsqueeze(1), float('-inf'))

        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        hidden = torch.matmul(scores, v)
        hidden = rearrange(hidden, 'b h n c -> b n (h c)')
        return hidden, scores


# ---------------------------------------------------------------------------
# RPEAttentionLayer  — attention norm is AdaptiveLayerNorm
# ---------------------------------------------------------------------------

class RPEAttentionLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=None, ctx=None):
        super().__init__()
        self.attention = RPEMultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.linear    = nn.Linear(d_model, d_model)
        self.dropout   = build_dropout_layer(dropout)
        self.norm      = AdaptiveLayerNorm(d_model, ctx=ctx)

    def forward(self, input_states, memory_states, position_states,
                memory_weights=None, memory_masks=None, attention_factors=None):
        hidden, scores = self.attention(
            input_states, memory_states, memory_states, position_states,
            key_weights=memory_weights, key_masks=memory_masks,
            attention_factors=attention_factors,
        )
        hidden = self.linear(hidden)
        hidden = self.dropout(hidden)
        output = self.norm(hidden + input_states)
        return output, scores


# ---------------------------------------------------------------------------
# RPETransformerLayer_LN  — self-attention with AdaptiveLayerNorm on attn norm
# ---------------------------------------------------------------------------

class RPETransformerLayer_LN(nn.Module):
    def __init__(self, d_model, num_heads, dropout=None, activation_fn='ReLU', ctx=None):
        super().__init__()
        self.attention = RPEAttentionLayer(d_model, num_heads, dropout=dropout, ctx=ctx)
        self.output    = AdaptiveAttentionOutput(d_model, dropout=dropout,
                                                 activation_fn=activation_fn, ctx=None)

    def forward(self, input_states, memory_states, position_states,
                memory_weights=None, memory_masks=None, attention_factors=None):
        hidden, scores = self.attention(
            input_states, memory_states, position_states,
            memory_weights=memory_weights, memory_masks=memory_masks,
            attention_factors=attention_factors,
        )
        output = self.output(hidden)
        return output, scores


# ---------------------------------------------------------------------------
# AdaptiveAttentionOutput
# ---------------------------------------------------------------------------

class AdaptiveAttentionOutput(nn.Module):
    def __init__(self, d_model, dropout=None, activation_fn='ReLU', ctx=None):
        super().__init__()
        self.expand     = nn.Linear(d_model, d_model * 2)
        self.activation = build_act_layer(activation_fn)
        self.squeeze    = nn.Linear(d_model * 2, d_model)
        self.dropout    = build_dropout_layer(dropout)
        self.norm       = nn.LayerNorm(d_model)   

    def forward(self, input_states):
        hidden = self.expand(input_states)
        hidden = self.activation(hidden)
        hidden = self.squeeze(hidden)
        hidden = self.dropout(hidden)
        return self.norm(input_states + hidden)