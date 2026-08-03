import pdb
import torch
from pareconv.extensions.pointops.functions import pointops

import numpy as np
from scipy.spatial import cKDTree


def radius_search(q_points, s_points, q_lengths, s_lengths, num_neighbors):
    r"""Computes k nearest neighbors for a stack-mode batch of clouds.

    This is the dense analogue of Minkowski's CoordinateManager: every query
    point is only matched against support points that belong to the **same
    cloud**, so different scenes in the batch never leak into each other.

    Implemented on GPU.

    Args:
        q_points (Tensor): stacked query points, shape (N_total, 3).
        s_points (Tensor): stacked support points, shape (M_total, 3).
        q_lengths (Tensor): per-cloud lengths in q_points, shape (C,).
        s_lengths (Tensor): per-cloud lengths in s_points, shape (C,).
            For B-batched ref/src registration, C = 2*B and the layout is
            [ref_1, ..., ref_B, src_1, ..., src_B].
        num_neighbors (int): number of neighbors per query point.

    Returns:
        neighbors (Tensor): k nearest neighbors of q_points in s_points,
            shape (N_total, k), with indices into the global s_points tensor.
    """
    num_clouds = q_lengths.shape[0]
    assert s_lengths.shape[0] == num_clouds, (
        f"q_lengths and s_lengths must describe the same number of clouds, "
        f"got {num_clouds} vs {s_lengths.shape[0]}"
    )

    indices_per_cloud = []
    q_off = 0
    s_off = 0
    for i in range(num_clouds):
        q_len = int(q_lengths[i])
        s_len = int(s_lengths[i])
        q_pcd = q_points[q_off:q_off + q_len].unsqueeze(0)  # [1, N_q_i, 3]
        s_pcd = s_points[s_off:s_off + s_len].unsqueeze(0)  # [1, N_s_i, 3]

        ind_local = pointops.knnquery_heap(num_neighbors, s_pcd, q_pcd)  # [1, N_q_i, k]
        # Defensive: when s_len < num_neighbors, the heap kernel leaves the
        # extra slots at their zero-init value, which is in range but still
        # ambiguous. Clamp explicitly so the contract is the same as the CPU
        # path (idx in [0, s_len)) and torch.compile bounds checks pass.
        if s_len > 0:
            ind_local = ind_local.clamp_(min=0, max=s_len - 1)
        ind_local = ind_local + s_off  # local -> global s_points indexing
        indices_per_cloud.append(ind_local)

        q_off += q_len
        s_off += s_len

    index = torch.cat(indices_per_cloud, dim=1)  # [1, N_total, k]
    return index.squeeze(0)



def radius_search_cpu(q_points, s_points, q_lengths, s_lengths, num_neighbors):
    """
    Modified from https://github.com/yaorz97/PARENet  
    - Use CPU version in order to include in the dataloader
    """
    q = q_points.numpy() if isinstance(q_points, torch.Tensor) else q_points
    s = s_points.numpy() if isinstance(s_points, torch.Tensor) else s_points
    q_lengths_l = q_lengths.tolist() if isinstance(q_lengths, torch.Tensor) else list(q_lengths)
    s_lengths_l = s_lengths.tolist() if isinstance(s_lengths, torch.Tensor) else list(s_lengths)

    num_clouds = len(q_lengths_l)
    assert len(s_lengths_l) == num_clouds, (
        f"q_lengths and s_lengths must describe the same number of clouds, "
        f"got {num_clouds} vs {len(s_lengths_l)}"
    )

    indices_per_cloud = []
    q_off = 0
    s_off = 0
    for i in range(num_clouds):
        q_len = int(q_lengths_l[i])
        s_len = int(s_lengths_l[i])
        q_i = q[q_off:q_off + q_len]
        s_i = s[s_off:s_off + s_len]

        if s_len == 0:
            # Degenerate cloud
            idx = np.zeros((q_len, num_neighbors), dtype=np.int64)
        else:
            tree = cKDTree(s_i)
            _, idx = tree.query(q_i, k=num_neighbors)  # local indices into s_i
            if num_neighbors == 1:
                idx = idx[:, None]
            if s_len < num_neighbors:
                idx = np.minimum(idx, s_len - 1)
        idx = idx + s_off  # local -> global s_points indexing
        indices_per_cloud.append(idx)

        q_off += q_len
        s_off += s_len

    out = np.concatenate(indices_per_cloud, axis=0).astype(np.int32)
    return torch.from_numpy(out)