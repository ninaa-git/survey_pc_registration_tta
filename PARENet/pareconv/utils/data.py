import pdb
from functools import partial

import numpy as np
import torch

from pareconv.modules.ops import grid_subsample, radius_search
from pareconv.utils.torch import build_dataloader
from pareconv.modules.ops.radius_search import radius_search_cpu



def precompute_subsample(points, lengths, num_stages, voxel_size, num_neighbors, subsample_ratio):
    assert num_stages == len(num_neighbors)

    points_list = []
    lengths_list = []
    for i in range(num_stages):
        if i > 0:
            points, lengths = grid_subsample(points, lengths, voxel_size=voxel_size)
        points_list.append(points)
        lengths_list.append(lengths)
        voxel_size *= subsample_ratio 
    return {
        'points': points_list,
        'lengths': lengths_list,
    }


def precompute_neibors(points_list, lengths_list, num_stages, num_neighbors):

    neighbors_list = []
    subsampling_list = []
    upsampling_list = []

    # knn search
    for i in range(num_stages):
        cur_points = points_list[i]
        cur_lengths = lengths_list[i]
        if i < num_stages:
            neighbors = radius_search(
                cur_points,
                cur_points,
                cur_lengths,
                cur_lengths,
                num_neighbors[i],
            )
            neighbors_list.append(neighbors)

        if i < num_stages - 1:
            sub_points = points_list[i + 1]
            sub_lengths = lengths_list[i + 1]

            subsampling = radius_search(
                sub_points,
                cur_points,
                sub_lengths,
                cur_lengths,
                num_neighbors[i],
            )
            subsampling_list.append(subsampling)

            if i > 0:
                upsampling = radius_search(
                    cur_points,
                    sub_points,
                    cur_lengths,
                    sub_lengths,
                    1,
                )
                upsampling_list.append(upsampling)
    return {
        'neighbors': neighbors_list,
        'subsampling': subsampling_list,
        'upsampling': upsampling_list,
    }


def precompute_neibors_cpu(points_list, lengths_list, num_stages, num_neighbors):
    neighbors_list, subsampling_list, upsampling_list = [], [], []
    for i in range(num_stages):
        cur_pts, cur_len = points_list[i], lengths_list[i]
        if i == 0:
            neighbors_list.append(None)          # stage-0 self-neighbors unused by backbone
        else:
            neighbors_list.append(
                radius_search_cpu(cur_pts, cur_pts, cur_len, cur_len, num_neighbors[i])
            )
        if i < num_stages - 1:                   
            sub_pts, sub_len = points_list[i + 1], lengths_list[i + 1]
            subsampling_list.append(
                radius_search_cpu(sub_pts, cur_pts, sub_len, cur_len, num_neighbors[i])
            )
            if i > 0:
                upsampling_list.append(
                    radius_search_cpu(cur_pts, sub_pts, cur_len, sub_len, 1)
                )
    return {'neighbors': neighbors_list, 'subsampling': subsampling_list, 'upsampling': upsampling_list}


def _build_padding_indices(lens_a, lens_b, B):
    device = lens_a.device
    a_max = int(lens_a.max())
    b_max = int(lens_b.max())
    sum_a = int(lens_a.sum())
    sum_b = int(lens_b.sum())

    a_off = torch.cat([lens_a.new_zeros(1), lens_a.cumsum(0)[:-1]])           # [B]
    b_off = sum_a + torch.cat([lens_b.new_zeros(1), lens_b.cumsum(0)[:-1]])   # [B]

    ar_a = torch.arange(a_max, device=device)
    ar_b = torch.arange(b_max, device=device)
    rows = torch.arange(B, device=device)

    # masks: [B, max], True where col < len  (contiguous from 0, same as before)
    a_mask = ar_a.unsqueeze(0) < lens_a.unsqueeze(1)
    b_mask = ar_b.unsqueeze(0) < lens_b.unsqueeze(1)

    # split = destination into flat [B*max]: row*max + local col, taken over valid slots.
    # Boolean indexing flattens row-major, so this reproduces the loop's concat order exactly.
    a_split = ((rows * a_max).unsqueeze(1) + ar_a.unsqueeze(0))[a_mask]
    b_split = ((rows * b_max).unsqueeze(1) + ar_b.unsqueeze(0))[b_mask]

    # stk = source indices into the stacked tensor; ref is contiguous front, src follows.
    a_stk = torch.arange(sum_a, device=device)
    b_stk = torch.arange(sum_a, sum_a + sum_b, device=device)

    return {
        'a_mask': a_mask, 'b_mask': b_mask,
        'a_split': a_split, 'b_split': b_split,
        'a_stk': a_stk, 'b_stk': b_stk,
        'a_max': a_max, 'b_max': b_max,
        'a_off': a_off, 'b_off': b_off,
    }


def registration_collate_fn_stack_mode(
    data_dicts, num_stages, voxel_size, num_neighbors, subsample_ratio, precompute_data=True
):
    r"""Collate function for registration in stack mode.

    Points are organized in the following order: [ref_1, ..., ref_B, src_1, ..., src_B].
    The correspondence indices are within each point cloud without accumulation.

    For B>1, this collate additionally builds:
      - 'ref_mask_c', 'src_mask_c': [B, N_max] boolean masks at the coarse level
      - 'ref_ind_c_split', 'src_ind_c_split': destination indices into a flat
        [B * N_max] tensor for scattering
      - 'ref_ind_c', 'src_ind_c': source indices into the stacked [N_total] tensor
      - 'ref_max_c', 'src_max_c': padding sizes at coarse level
      - 'ref_off_f', 'src_off_f', 'ref_off', 'src_off': per-pair offsets at fine
        and full-resolution levels (for the per-pair Python loop in the model)

    Args:
        data_dicts (List[Dict])
        num_stages (int)
        voxel_size (float)
        num_neighbors (List[int])
        precompute_data (bool)
    Returns:
        collated_dict (Dict)
    """
    batch_size = len(data_dicts)
    collated_dict = {}
    for data_dict in data_dicts:
        for key, value in data_dict.items():
            if isinstance(value, np.ndarray):
                value = torch.from_numpy(value)
            if key not in collated_dict:
                collated_dict[key] = []
            collated_dict[key].append(value)

    feats = torch.cat(collated_dict.pop('ref_feats') + collated_dict.pop('src_feats'), dim=0)
    points_list = collated_dict.pop('ref_points') + collated_dict.pop('src_points')
    lengths = torch.LongTensor([points.shape[0] for points in points_list])
    points = torch.cat(points_list, dim=0)

    if batch_size == 1:
        for key, value in collated_dict.items():
            collated_dict[key] = value[0]

    collated_dict['features'] = feats
    if precompute_data:
        input_dict = precompute_subsample(points, lengths, num_stages, voxel_size, num_neighbors, subsample_ratio)
        collated_dict.update(input_dict)
        data = precompute_neibors_cpu(
            input_dict['points'], input_dict['lengths'], num_stages, num_neighbors
        )
        collated_dict.update(data)
    else:
        collated_dict['points'] = points
        collated_dict['lengths'] = lengths
    collated_dict['batch_size'] = batch_size

    if precompute_data:
        B = batch_size
        # Coarse
        lens_c = collated_dict['lengths'][-1]                # [2B]
        ref_lens_c, src_lens_c = lens_c[:B], lens_c[B:]
        coarse = _build_padding_indices(ref_lens_c, src_lens_c, B)
        collated_dict['ref_mask_c']      = coarse['a_mask']
        collated_dict['src_mask_c']      = coarse['b_mask']
        collated_dict['ref_ind_c_split'] = coarse['a_split']
        collated_dict['src_ind_c_split'] = coarse['b_split']
        collated_dict['ref_ind_c']       = coarse['a_stk']
        collated_dict['src_ind_c']       = coarse['b_stk']
        collated_dict['ref_max_c']       = coarse['a_max']
        collated_dict['src_max_c']       = coarse['b_max']
        collated_dict['ref_off_c']       = coarse['a_off']
        collated_dict['src_off_c']       = coarse['b_off']

        # Fine 
        lens_f = collated_dict['lengths'][1]
        ref_lens_f, src_lens_f = lens_f[:B], lens_f[B:]
        sum_ref_f = int(ref_lens_f.sum())
        collated_dict['ref_off_f'] = torch.cat([ref_lens_f.new_zeros(1), ref_lens_f.cumsum(0)[:-1]])
        collated_dict['src_off_f'] = sum_ref_f + torch.cat([src_lens_f.new_zeros(1), src_lens_f.cumsum(0)[:-1]])
        collated_dict['ref_lens_f'] = ref_lens_f
        collated_dict['src_lens_f'] = src_lens_f
        collated_dict['ref_max_f'] = int(ref_lens_f.max())
        collated_dict['src_max_f'] = int(src_lens_f.max())

        # Full-resolution
        lens_0 = collated_dict['lengths'][0]
        ref_lens_0, src_lens_0 = lens_0[:B], lens_0[B:]
        sum_ref_0 = int(ref_lens_0.sum())
        collated_dict['ref_off']  = torch.cat([ref_lens_0.new_zeros(1), ref_lens_0.cumsum(0)[:-1]])
        collated_dict['src_off']  = sum_ref_0 + torch.cat([src_lens_0.new_zeros(1), src_lens_0.cumsum(0)[:-1]])
        collated_dict['ref_lens'] = ref_lens_0
        collated_dict['src_lens'] = src_lens_0

        # Coarse-level per-pair 
        collated_dict['ref_lens_c'] = ref_lens_c
        collated_dict['src_lens_c'] = src_lens_c

    return collated_dict


def calibrate_neighbors_stack_mode(
    dataset, collate_fn, num_stages, voxel_size, search_radius, subsample_ratio=2,
    keep_ratio=0.8, sample_threshold=2000
):
    # Compute higher bound of neighbors number in a neighborhood
    hist_n = int(np.ceil(4 / 3 * np.pi * (search_radius / voxel_size + 1) ** 2))
    neighbor_hists = np.zeros((num_stages, hist_n), dtype=np.int32)
    max_neighbor_limits = [hist_n] * num_stages

    # Get histogram of neighborhood sizes i in 1 epoch max.
    for i in range(len(dataset)):
        data_dict = collate_fn(
            [dataset[i]], num_stages, voxel_size, max_neighbor_limits, subsample_ratio,
            precompute_data=True,
        )

        counts = []
        for stage_i, neighbors in enumerate(data_dict['neighbors']):
            nb = neighbors.numpy()                       
            is_real = np.ones(nb.shape, dtype=bool)
            is_real[:, 1:] = nb[:, 1:] != nb[:, :-1]
            counts.append(is_real.sum(axis=1).astype(np.int32))
        hists = [np.bincount(c, minlength=hist_n)[:hist_n] for c in counts]
        neighbor_hists += np.vstack(hists)

        if np.min(np.sum(neighbor_hists, axis=1)) > sample_threshold:
            break

    cum_sum = np.cumsum(neighbor_hists.T, axis=0)
    neighbor_limits = np.sum(cum_sum < (keep_ratio * cum_sum[hist_n - 1, :]), axis=0)

    return neighbor_limits


def build_dataloader_stack_mode(
    dataset,
    collate_fn,
    num_stages,
    voxel_size,
    num_neighbors,
    subsample_ratio,
    batch_size=1,
    num_workers=12,
    shuffle=False,
    drop_last=False,
    distributed=False,
    precompute_data=True,
    persistent_workers=True,
):
    dataloader = build_dataloader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        collate_fn=partial(
            collate_fn,
            num_stages=num_stages,
            voxel_size=voxel_size,
            num_neighbors=num_neighbors,
            subsample_ratio=subsample_ratio,
            precompute_data=precompute_data,
        ),
        pin_memory=True,
        drop_last=drop_last,
        distributed=distributed,
        persistent_workers=persistent_workers and num_workers > 0,
    )
    return dataloader