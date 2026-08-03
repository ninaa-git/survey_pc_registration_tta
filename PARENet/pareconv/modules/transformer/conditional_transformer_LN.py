import torch
import torch.nn as nn

from pareconv.modules.transformer.lrpe_transformer import LRPETransformerLayer
from pareconv.modules.transformer.pe_transformer import PETransformerLayer
from pareconv.modules.transformer.rpe_transformer_LN import TTAContext
from pareconv.modules.transformer.rpe_transformer import RPETransformerLayer
from pareconv.modules.transformer.rpe_transformer_LN import RPETransformerLayer_LN
from pareconv.modules.transformer.vanilla_transformer_LN import TransformerLayer_LN
from pareconv.modules.transformer.bias_transformer import BiasTransformerLayer


def _check_block_type(block):
    if block not in ['self', 'cross']:
        raise ValueError('Unsupported block type "{}".'.format(block))


class RPEConditionalTransformer_LN(nn.Module):
    '''
    Modified from https://github.com/yaorz97/PARENet  
    '''
    def __init__(self, blocks, d_model, num_heads, dropout=None,
                 activation_fn='ReLU', return_attention_scores=False,
                 parallel=False, ctx=None):
        super().__init__()
        self.blocks = blocks
        self.return_attention_scores = return_attention_scores
        self.parallel = parallel
        self.ctx = ctx

        layers = []
        for block in self.blocks:
            _check_block_type(block)
            if block == 'self':
                if ctx is not None:
                    layers.append(RPETransformerLayer_LN(
                        d_model, num_heads, dropout=dropout,
                        activation_fn=activation_fn, ctx=ctx,
                    ))
                else:
                    layers.append(RPETransformerLayer(
                        d_model, num_heads, dropout=dropout,
                        activation_fn=activation_fn
                    ))
            else:
                layers.append(TransformerLayer_LN(
                    d_model, num_heads, dropout=dropout,
                    activation_fn=activation_fn, ctx=ctx,
                ))
        self.layers = nn.ModuleList(layers)

    def forward(self, feats0, feats1, embeddings0, embeddings1,
                masks0=None, masks1=None):
        attention_scores = []
        ctx = self.ctx

        for i, block in enumerate(self.blocks):
            if block == 'self':
                if ctx is not None:
                    ctx._processing_corrupted = True
                    ctx._current_mask = masks0
                feats0, scores0 = self.layers[i](feats0, feats0, embeddings0, memory_masks=masks0)
                if ctx is not None:
                    ctx._processing_corrupted = False
                    ctx._current_mask = masks1
                feats1, scores1 = self.layers[i](feats1, feats1, embeddings1, memory_masks=masks1)
            else:
                if self.parallel:
                    if ctx is not None:
                        ctx._processing_corrupted = True
                        ctx._current_mask = masks0
                    new_feats0, scores0 = self.layers[i](feats0, feats1, memory_masks=masks1)
                    if ctx is not None:
                        ctx._processing_corrupted = False
                        ctx._current_mask = masks1
                    new_feats1, scores1 = self.layers[i](feats1, feats0, memory_masks=masks0)
                    feats0, feats1 = new_feats0, new_feats1
                else:
                    if ctx is not None:
                        ctx._processing_corrupted = True
                        ctx._current_mask = masks0
                    feats0, scores0 = self.layers[i](feats0, feats1, memory_masks=masks1)
                    if ctx is not None:
                        ctx._processing_corrupted = False
                        ctx._current_mask = masks1
                    feats1, scores1 = self.layers[i](feats1, feats0, memory_masks=masks0)

            if self.return_attention_scores:
                attention_scores.append([scores0, scores1])

        if self.return_attention_scores:
            return feats0, feats1, attention_scores
        return feats0, feats1


class VanillaConditionalTransformer_LN(nn.Module):
    def __init__(self, blocks, d_model, num_heads, dropout=None,
                 activation_fn='ReLU', return_attention_scores=False):
        super().__init__()
        self.blocks = blocks
        self.layers = nn.ModuleList([
            TransformerLayer_LN(d_model, num_heads, dropout=dropout,
                             activation_fn=activation_fn)
            for _ in blocks
        ])
        self.return_attention_scores = return_attention_scores

    def forward(self, feats0, feats1, masks0=None, masks1=None):
        scores_list = []
        for i, block in enumerate(self.blocks):
            if block == 'self':
                feats0, s0 = self.layers[i](feats0, feats0, memory_masks=masks0)
                feats1, s1 = self.layers[i](feats1, feats1, memory_masks=masks1)
            else:
                feats0, s0 = self.layers[i](feats0, feats1, memory_masks=masks1)
                feats1, s1 = self.layers[i](feats1, feats0, memory_masks=masks0)
            if self.return_attention_scores:
                scores_list.append([s0, s1])
        if self.return_attention_scores:
            return feats0, feats1, scores_list
        return feats0, feats1


class PEConditionalTransformer(nn.Module):
    def __init__(self, blocks, d_model, num_heads, dropout=None,
                 activation_fn='ReLU', return_attention_scores=False):
        super().__init__()
        self.blocks = blocks
        layers = []
        for block in blocks:
            _check_block_type(block)
            if block == 'self':
                layers.append(PETransformerLayer(d_model, num_heads, dropout=dropout,
                                                  activation_fn=activation_fn))
            else:
                layers.append(TransformerLayer_LN(d_model, num_heads, dropout=dropout,
                                               activation_fn=activation_fn))
        self.layers = nn.ModuleList(layers)
        self.return_attention_scores = return_attention_scores

    def forward(self, feats0, feats1, embeddings0, embeddings1,
                masks0=None, masks1=None):
        scores_list = []
        for i, block in enumerate(self.blocks):
            if block == 'self':
                feats0, s0 = self.layers[i](feats0, feats0, embeddings0, embeddings0,
                                             memory_masks=masks0)
                feats1, s1 = self.layers[i](feats1, feats1, embeddings1, embeddings1,
                                             memory_masks=masks1)
            else:
                feats0, s0 = self.layers[i](feats0, feats1, memory_masks=masks1)
                feats1, s1 = self.layers[i](feats1, feats0, memory_masks=masks0)
            if self.return_attention_scores:
                scores_list.append([s0, s1])
        if self.return_attention_scores:
            return feats0, feats1, scores_list
        return feats0, feats1


class LRPEConditionalTransformer(nn.Module):
    def __init__(self, blocks, d_model, num_heads, num_embeddings, dropout=None,
                 activation_fn='ReLU', return_attention_scores=False):
        super().__init__()
        self.blocks = blocks
        layers = []
        for block in blocks:
            _check_block_type(block)
            if block == 'self':
                layers.append(LRPETransformerLayer(d_model, num_heads, num_embeddings,
                                                    dropout=dropout,
                                                    activation_fn=activation_fn))
            else:
                layers.append(TransformerLayer_LN(d_model, num_heads, dropout=dropout,
                                               activation_fn=activation_fn))
        self.layers = nn.ModuleList(layers)
        self.return_attention_scores = return_attention_scores

    def forward(self, feats0, feats1, emb_indices0, emb_indices1,
                masks0=None, masks1=None):
        scores_list = []
        for i, block in enumerate(self.blocks):
            if block == 'self':
                feats0, s0 = self.layers[i](feats0, feats0, emb_indices0,
                                             memory_masks=masks0)
                feats1, s1 = self.layers[i](feats1, feats1, emb_indices1,
                                             memory_masks=masks1)
            else:
                feats0, s0 = self.layers[i](feats0, feats1, memory_masks=masks1)
                feats1, s1 = self.layers[i](feats1, feats0, memory_masks=masks0)
            if self.return_attention_scores:
                scores_list.append([s0, s1])
        if self.return_attention_scores:
            return feats0, feats1, scores_list
        return feats0, feats1


class BiasConditionalTransformer(nn.Module):
    def __init__(self, blocks, d_model, num_heads, dropout=None,
                 activation_fn='ReLU', return_attention_scores=False,
                 parallel=False):
        super().__init__()
        self.blocks   = blocks
        self.parallel = parallel
        layers = []
        for block in blocks:
            _check_block_type(block)
            if block == 'self':
                layers.append(BiasTransformerLayer(d_model, num_heads, dropout=dropout,
                                                    activation_fn=activation_fn))
            else:
                layers.append(TransformerLayer_LN(d_model, num_heads, dropout=dropout,
                                               activation_fn=activation_fn))
        self.layers = nn.ModuleList(layers)
        self.return_attention_scores = return_attention_scores

    def forward(self, feats0, feats1, embeddings0, embeddings1,
                masks0=None, masks1=None):
        scores_list = []
        for i, block in enumerate(self.blocks):
            if block == 'self':
                feats0, s0 = self.layers[i](feats0, feats0, embeddings0,
                                             memory_masks=masks0)
                feats1, s1 = self.layers[i](feats1, feats1, embeddings1,
                                             memory_masks=masks1)
            else:
                if self.parallel:
                    new0, s0 = self.layers[i](feats0, feats1, memory_masks=masks1)
                    new1, s1 = self.layers[i](feats1, feats0, memory_masks=masks0)
                    feats0, feats1 = new0, new1
                else:
                    feats0, s0 = self.layers[i](feats0, feats1, memory_masks=masks1)
                    feats1, s1 = self.layers[i](feats1, feats0, memory_masks=masks0)
            if self.return_attention_scores:
                scores_list.append([s0, s1])
        if self.return_attention_scores:
            return feats0, feats1, scores_list
        return feats0, feats1