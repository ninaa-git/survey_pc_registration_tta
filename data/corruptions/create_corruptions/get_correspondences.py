from scipy.spatial import cKDTree

def get_correspondences(ref_points, src_points, transform, matching_radius):
    r"""Find the ground truth correspondences within the matching radius between two point clouds.

    Return correspondence indices [indices in ref_points, indices in src_points]
    """
    src_points = apply_transform(src_points, transform)
    src_tree = cKDTree(src_points)
    indices_list = src_tree.query_ball_point(ref_points, matching_radius)
    corr_indices = np.array(
        [(i, j) for i, indices in enumerate(indices_list) for j in indices],
        dtype=np.int64,
    )
    return corr_indices