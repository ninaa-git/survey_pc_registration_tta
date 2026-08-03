import pdb
import warnings

import torch

from pareconv.modules.ops.pairwise_distance import pairwise_distance
from pareconv.modules.ops.index_select import index_select


def get_point_to_node_indices(points: torch.Tensor, nodes: torch.Tensor, return_counts: bool = False):
    r"""Compute Point-to-Node partition indices of the point cloud.

    Distribute points to the nearest node. Each point is distributed to only one node.

    Args:
        points (Tensor): point cloud (N, C)
        nodes (Tensor): node set (M, C)
        return_counts (bool=False): whether return the number of points in each node.

    Returns:
        indices (LongTensor): index of the node that each point belongs to (N,)
        node_sizes (longTensor): the number of points in each node.
    """
    sq_dist_mat = pairwise_distance(points, nodes)
    indices = sq_dist_mat.min(dim=1)[1]
    if return_counts:
        unique_indices, unique_counts = torch.unique(indices, return_counts=True)
        node_sizes = torch.zeros(nodes.shape[0], dtype=torch.long, device=nodes.device)
        node_sizes[unique_indices] = unique_counts
        return indices, node_sizes
    else:
        return indices


@torch.no_grad()
def knn_partition(points: torch.Tensor, nodes: torch.Tensor, k: int, return_distance: bool = False):
    r"""k-NN partition of the point cloud.

    Find the k nearest points for each node.

    Args:
        points: torch.Tensor (num_point, num_channel)
        nodes: torch.Tensor (num_node, num_channel)
        k: int
        return_distance: bool

    Returns:
        knn_indices: torch.Tensor (num_node, k)
        knn_indices: torch.Tensor (num_node, k)
    """
    k = min(k, points.shape[0])
    sq_dist_mat = pairwise_distance(nodes, points)
    knn_sq_distances, knn_indices = sq_dist_mat.topk(dim=1, k=k, largest=False)
    if return_distance:
        knn_distances = torch.sqrt(knn_sq_distances)
        return knn_distances, knn_indices
    else:
        return knn_indices


@torch.no_grad()
def point_to_node_partition(
    points: torch.Tensor,
    nodes: torch.Tensor,
    point_limit: int,
    return_count: bool = False,
):
    r"""Point-to-Node partition to the point cloud.

    Fixed knn bug.

    Args:
        points (Tensor): (N, 3)
        nodes (Tensor): (M, 3)
        point_limit (int): max number of points to each node
        return_count (bool=False): whether to return `node_sizes`

    Returns:
        point_to_node (Tensor): (N,)
        node_sizes (LongTensor): (M,)
        node_masks (BoolTensor): (M,)
        node_knn_indices (LongTensor): (M, K)
        node_knn_masks (BoolTensor) (M, K)
    """

    sq_dist_mat = pairwise_distance(nodes, points)  # (M, N)
    device = nodes.device
    point_to_node = sq_dist_mat.min(dim=0)[1]  # (N,)
    node_masks = torch.zeros(nodes.shape[0], dtype=torch.bool, device=device)  # (M,)
    node_masks.index_fill_(0, point_to_node, True)

    matching_masks = torch.zeros_like(sq_dist_mat, dtype=torch.bool)  # (M, N)
    point_indices = torch.arange(points.shape[0], device=device)  # (N,)
    matching_masks[point_to_node, point_indices] = True  # (M, N)
    sq_dist_mat.masked_fill_(~matching_masks, 1e12)  # (M, N)

    node_knn_indices = sq_dist_mat.topk(k=point_limit, dim=1, largest=False)[1]  # (M, K)
    node_knn_node_indices = index_select(point_to_node, node_knn_indices, dim=0)  # (M, K)
    node_indices = torch.arange(nodes.shape[0], device=device).unsqueeze(1).expand(-1, point_limit)  # (M, K)
    node_knn_masks = torch.eq(node_knn_node_indices, node_indices)  # (M, K)
    node_knn_indices.masked_fill_(~node_knn_masks, points.shape[0])

    if return_count:
        unique_indices, unique_counts = torch.unique(point_to_node, return_counts=True)
        node_sizes = torch.zeros(nodes.shape[0], dtype=torch.long, device=device)  # (M,)
        node_sizes.index_put_([unique_indices], unique_counts)
        return point_to_node, node_sizes, node_masks, node_knn_indices, node_knn_masks
    else:
        return point_to_node, node_masks, node_knn_indices, node_knn_masks

@torch.no_grad()
def batched_point_to_node_partition(
    points,        # [B, N_max, 3]
    nodes,         # [B, M_max, 3]
    point_limit,   # int K
    point_masks,   # [B, N_max] bool, True for valid points
    node_masks_in, # [B, M_max] bool, True for valid (padded-validity) nodes
):
    B, N_max, _ = points.shape
    M_max = nodes.shape[1]
    K = point_limit
    device = points.device
 
    # (B, M_max, N_max)
    sq_dist_mat = pairwise_distance(nodes, points)
 
    # Block padded queries (rows): set their entire row to large; result is
    # ignored downstream because node_masks below will be False for them.
    invalid_q = ~node_masks_in                        # [B, M_max]
    sq_dist_mat = sq_dist_mat.masked_fill(invalid_q.unsqueeze(2), 1e12)
 
    # Block padded supports (cols): they must never be the nearest.
    invalid_k = ~point_masks                          # [B, N_max]
    sq_dist_mat = sq_dist_mat.masked_fill(invalid_k.unsqueeze(1), 1e12)
 
    # For each point, the index of the nearest node (across the M dim).
    point_to_node = sq_dist_mat.min(dim=1)[1]         # [B, N_max]
    # Padded points get a meaningless point_to_node (large distance ties). Mask
    # them out using point_masks so they don't contribute to node_masks below.
 
    # Build node_masks: which nodes have at least one *real* point assigned?
    # We accumulate via scatter_add on a one-hot count, only counting valid points.
    node_assignment_count = torch.zeros(B, M_max, dtype=torch.long, device=device)
    valid_p2n = point_to_node.masked_fill(~point_masks, M_max)  # send invalid to sentinel
    # scatter_add over a sentinel-padded buffer to ignore invalid entries
    buf = torch.zeros(B, M_max + 1, dtype=torch.long, device=device)
    ones = torch.ones_like(valid_p2n, dtype=torch.long)
    buf.scatter_add_(1, valid_p2n, ones)
    node_assignment_count = buf[:, :M_max]
    node_masks = (node_assignment_count > 0) & node_masks_in
 
    # Build node_knn_indices: top-K nearest *assigned* points per node.
    # First filter by "this node owns this point": same trick as 1D version.
    point_idx_grid = torch.arange(N_max, device=device).view(1, 1, N_max)             # [1, 1, N_max]
    # mat[b, m, n] == True iff point n's nearest node is m
    p2n_expanded = point_to_node.unsqueeze(1).expand(B, M_max, N_max)                 # [B, M_max, N_max]
    m_idx_grid = torch.arange(M_max, device=device).view(1, M_max, 1)                 # [1, M_max, 1]
    matching_mat = (p2n_expanded == m_idx_grid)                                        # [B, M_max, N_max]
    matching_mat = matching_mat & point_masks.unsqueeze(1)                             # only real points
 
    sq_dist_mat_for_knn = sq_dist_mat.masked_fill(~matching_mat, 1e12)

    if N_max < K:
        pad = K - N_max
        dist_pad  = sq_dist_mat_for_knn.new_full((B, M_max, pad), 1e12)
        match_pad = matching_mat.new_zeros((B, M_max, pad))
        sq_dist_mat_for_knn = torch.cat([sq_dist_mat_for_knn, dist_pad], dim=2)
        matching_mat        = torch.cat([matching_mat, match_pad], dim=2)
    
    node_knn_indices = sq_dist_mat_for_knn.topk(k=K, dim=2, largest=False)[1]          # [B, M_max, K]
 
    # Validate selected neighbors: a slot is valid iff its underlying matching
    # was True (i.e., not a sentinel pick).
    knn_match = torch.gather(matching_mat, 2, node_knn_indices)                        # [B, M_max, K]
    node_knn_masks = knn_match & node_masks_in.unsqueeze(2)
 
    # Replace invalid neighbor indices with the per-cloud sentinel = N_max.
    node_knn_indices = node_knn_indices.masked_fill(~node_knn_masks, N_max)
 
    return point_to_node, node_masks, node_knn_indices, node_knn_masks
 


@torch.no_grad()
def ball_query_partition(
    points: torch.Tensor,
    nodes: torch.Tensor,
    radius: float,
    point_limit: int,
    return_count: bool = False,
):
    node_knn_distances, node_knn_indices = knn_partition(points, nodes, point_limit, return_distance=True)
    node_knn_masks = torch.lt(node_knn_distances, radius)  # (N, k)
    sentinel_indices = torch.full_like(node_knn_indices, points.shape[0])  # (N, k)
    node_knn_indices = torch.where(node_knn_masks, node_knn_indices, sentinel_indices)  # (N, k)
    node_masks = node_knn_masks.sum(-1) > 0
    if return_count:
        node_sizes = node_knn_masks.sum(1)  # (N,)
        return node_masks, node_knn_indices, node_knn_masks, node_sizes
    else:
        return node_masks, node_knn_indices, node_knn_masks


@torch.no_grad()
def point_to_node_partition_bug(
    points: torch.Tensor,
    nodes: torch.Tensor,
    point_limit: int,
    return_count: bool = False,
):
    r"""Point-to-Node partition to the point cloud.

    BUG: this implementation ignores point_to_node indices when building patches. However, the points that do not
    belong to a superpoint should be masked out.


    Args:
        points (Tensor): (N, 3)
        nodes (Tensor): (M, 3)
        point_limit (int): max number of points to each node
        return_count (bool=False): whether to return `node_sizes`

    Returns:
        point_to_node (Tensor): (N,)
        node_sizes (LongTensor): (M,)
        node_masks (BoolTensor): (M,)
        node_knn_indices (LongTensor): (M, K)
        node_knn_masks (BoolTensor) (M, K)
    """
    warnings.warn('There is a bug in this implementation. Use `point_to_node_partition` instead.')
    sq_dist_mat = pairwise_distance(nodes, points)  # (M, N)
    device = nodes.device
    point_to_node = sq_dist_mat.min(dim=0)[1]  # (N,)
    node_masks = torch.zeros(nodes.shape[0], dtype=torch.bool, device=device)  # (M,)
    node_masks.index_fill_(0, point_to_node, True)

    node_knn_indices = sq_dist_mat.topk(k=point_limit, dim=1, largest=False)[1]  # (M, K)
    node_knn_node_indices = index_select(point_to_node, node_knn_indices, dim=0)  # (M, K)
    node_indices = torch.arange(nodes.shape[0]).to(device).unsqueeze(1).expand(-1, point_limit)  # (M, K)
    node_knn_masks = torch.eq(node_knn_node_indices, node_indices)  # (M, K)
    node_knn_indices.masked_fill_(~node_knn_masks, points.shape[0])

    if return_count:
        unique_indices, unique_counts = torch.unique(point_to_node, return_counts=True)
        node_sizes = torch.zeros(nodes.shape[0], dtype=torch.long, device=device)  # (M,)
        node_sizes.index_put_([unique_indices], unique_counts)
        return point_to_node, node_sizes, node_masks, node_knn_indices, node_knn_masks
    else:
        return point_to_node, node_masks, node_knn_indices, node_knn_masks


# @torch.no_grad()
# def ball_query_partition(
#     points: torch.Tensor,
#     nodes: torch.Tensor,
#     radius: float,
#     point_limit: int,
#     return_count: bool = False,
# ):
#     node_knn_distances, node_knn_indices = knn_partition(points, nodes, point_limit, return_distance=True)
#     node_knn_masks = torch.lt(node_knn_distances, radius)  # (N, k)
#     sentinel_indices = torch.full_like(node_knn_indices, points.shape[0])  # (N, k)
#     node_knn_indices = torch.where(node_knn_masks, node_knn_indices, sentinel_indices)  # (N, k)
#
#     if return_count:
#         node_sizes = node_knn_masks.sum(1)  # (N,)
#         return node_knn_indices, node_knn_masks, node_sizes
#     else:
#         return node_knn_indices, node_knn_masks
