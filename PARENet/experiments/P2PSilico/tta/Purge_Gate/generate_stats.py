import os
import torch
from tqdm import tqdm

from pareconv.utils.torch import to_cuda
from pareconv.utils.data import precompute_neibors
from config import make_cfg
from tta.Purge_Gate.model import create_model
from dataset import train_valid_data_loader
from common_utils.parser import make_parser

def _welford_update(x, count, mean, M2):
    """Single-sample update — kept for API compatibility."""
    count += 1
    delta = x - mean
    mean  = mean + delta / count
    M2    = M2   + delta * (x - mean)
    return count, mean, M2


def _welford_merge(count_a, mean_a, M2_a, count_b, mean_b, M2_b):
    """Combine two Welford accumulators (Chan et al., 1979)."""
    if count_a == 0:
        return count_b, mean_b, M2_b
    if count_b == 0:
        return count_a, mean_a, M2_a
    n     = count_a + count_b
    delta = mean_b - mean_a
    mean  = mean_a + delta * (count_b / n)
    M2    = M2_a + M2_b + delta.pow(2) * (count_a * count_b / n)
    return n, mean, M2


def _welford_chunk_update(chunk, count, mean, M2):
    """Fold a whole [N, C] chunk into an existing accumulator in one shot."""
    n_b = chunk.shape[0]
    if n_b == 0:
        return count, mean, M2
    mean_b = chunk.mean(dim=0)
    if n_b == 1:
        M2_b = torch.zeros_like(mean_b)
    else:
        # Population sum-of-squares about the chunk's own mean = unbiased_var * (n-1).
        M2_b = chunk.var(dim=0, unbiased=True) * (n_b - 1)
    return _welford_merge(count, mean, M2, n_b, mean_b, M2_b)

def _welford_finalize(count, mean, M2):
    if count < 2:
        return mean, torch.full_like(M2, float('nan'))
    return mean, torch.sqrt(M2 / (count - 1))

def _run_forward(model, data_dict, cfg):
    data_dict = to_cuda(data_dict)
    data_dict.update(precompute_neibors(
        data_dict['points'],
        data_dict['lengths'],
        cfg.backbone.num_stages,
        cfg.backbone.num_neighbors,
    ))
    with torch.no_grad():
        return model.forward_out_intermediate(data_dict)


_FEAT_TO_MASK = {
    'b_ref_feats_c_pad': 'b_ref_masks_c',
}


def collect_embeddings(
    model,
    cfg,
    data_loader,
    feature_keys,
    mode="stats",
    max_batches=None,
):
    """
    Same modes as before:
        "stats"           → {'mean': [C], 'std': [C], 'count': int}
        "stats_per_cloud" → list[ {'mean': [C], 'std': [C]} ], one entry per cloud
        "per_cloud"       → tensor [N_clouds, C]
        "per_point"       → {'features': [N_total_valid, C], 'labels': [N_total_valid]}
    Labels in "per_point" are now per-cloud ids that increase monotonically
    across the dataset (each batch contributes B ids), not per-batch ids.
    """
    model.eval()
    total = min(len(data_loader), max_batches) if max_batches else len(data_loader)

    welford      = {k: {'count': 0, 'mean': None, 'M2': None} for k in feature_keys}
    cloud_stats  = {k: [] for k in feature_keys}
    cloud_vecs   = {k: [] for k in feature_keys}
    point_feats  = {k: [] for k in feature_keys}
    point_labels = {k: [] for k in feature_keys}

    cloud_counter = 0   # global cloud id; each batch contributes B

    for batch_idx, data_dict in tqdm(enumerate(data_loader), total=total, desc="Collecting"):
        if max_batches and batch_idx >= max_batches:
            break

        output_dict = _run_forward(model, data_dict, cfg)
        B = int(output_dict['batch_size'])
        cloud_ids = list(range(cloud_counter, cloud_counter + B))

        for key in feature_keys:
            if key not in output_dict:
                print(f"Warning: '{key}' not found. Available: {list(output_dict.keys())}")
                continue
            mask_key = _FEAT_TO_MASK.get(key)
            if mask_key is None or mask_key not in output_dict:
                print(f"Warning: no mask known for '{key}'. Add it to _FEAT_TO_MASK.")
                continue

            feats = output_dict[key]            # [B, N_max, C]
            mask  = output_dict[mask_key].bool()  # [B, N_max]

            fm = feats.mean(dim=-1, keepdim=True)
            fs = feats.std(dim=-1,  keepdim=True)
            feats_norm = (feats - fm) / (fs + 1e-6)

            if mode == "stats":
                valid_feats = feats_norm[mask].detach().cpu()   # [N_total_valid, C]
                if welford[key]['mean'] is None:
                    C = valid_feats.shape[1]
                    welford[key]['mean'] = torch.zeros(C, dtype=valid_feats.dtype)
                    welford[key]['M2']   = torch.zeros(C, dtype=valid_feats.dtype)
                welford[key]['count'], welford[key]['mean'], welford[key]['M2'] = \
                    _welford_chunk_update(
                        valid_feats,
                        welford[key]['count'], welford[key]['mean'], welford[key]['M2'],
                    )


            elif mode == "stats_per_cloud":
                for b in range(B):
                    f = feats_norm[b][mask[b]].detach().cpu()   # [N_valid_b, C]
                    if f.shape[0] < 2:
                        # std with <2 points is undefined; skip or store nan.
                        cloud_stats[key].append({
                            'mean': f.mean(dim=0) if f.shape[0] else torch.zeros(f.shape[-1]),
                            'std':  torch.full((f.shape[-1],), float('nan')),
                        })
                        continue
                    cloud_stats[key].append({'mean': f.mean(dim=0), 'std': f.std(dim=0)})

            elif mode == "per_cloud":
                for b in range(B):
                    f = feats_norm[b][mask[b]]
                    if f.shape[0] == 0:
                        cloud_vecs[key].append(torch.zeros(f.shape[-1]))
                        continue
                    cloud_vecs[key].append(f.mean(dim=0).detach().cpu())

            elif mode == "per_point":
                for b in range(B):
                    f = feats_norm[b][mask[b]].detach().cpu()
                    if f.shape[0] == 0:
                        continue
                    point_feats[key].append(f)
                    point_labels[key].append(
                        torch.full((f.shape[0],), cloud_ids[b], dtype=torch.long)
                    )

        cloud_counter += B

    # ---- Build results (unchanged) ----
    results = {}
    for key in feature_keys:
        if mode == "stats" and welford[key]['mean'] is not None:
            mean, std = _welford_finalize(welford[key]['count'],
                                          welford[key]['mean'],
                                          welford[key]['M2'])
            results[key] = {'mean': mean, 'std': std, 'count': welford[key]['count']}
            print(f"\n{key}: {welford[key]['count']:,} points | "
                  f"mean [{mean.min():.4f}, {mean.max():.4f}] | "
                  f"std  [{std.min():.4f},  {std.max():.4f}]")

        elif mode == "stats_per_cloud" and cloud_stats[key]:
            results[key] = cloud_stats[key]
            print(f"\n{key}: {len(results[key])} clouds, "
                  f"each with mean/std of dim {results[key][0]['mean'].shape[0]}")

        elif mode == "per_cloud" and cloud_vecs[key]:
            results[key] = torch.stack(cloud_vecs[key])
            print(f"\n{key}: {results[key].shape[0]} clouds × {results[key].shape[1]} dims")

        elif mode == "per_point" and point_feats[key]:
            results[key] = {
                'features': torch.cat(point_feats[key]),
                'labels':   torch.cat(point_labels[key]),
            }
            print(f"\n{key}: {results[key]['features'].shape[0]} points × "
                  f"{results[key]['features'].shape[1]} dims")

    return results


# ---------------------------------------------------------------------------
# Cache-aware loader
# ---------------------------------------------------------------------------

def load_or_collect(
    model,
    cfg,
    cache_path,
    feature_keys,
    data_loader=None,
    mode="stats",
    max_batches=None,
    force_regenerate=False,
    cloud_index=None,       # for "per_point": filter to this cloud index after loading
):
    """
    Load cached results if available, otherwise collect and cache them.

    For mode="stats" and "stats_per_cloud", tensors are moved to CUDA automatically.
    For mode="per_point" with cloud_index set, returns only points for that cloud.
    """
    if os.path.exists(cache_path) and not force_regenerate:
        print(f"Loading cached embeddings from: {cache_path}")
        results = torch.load(cache_path)
    else:
        if data_loader is None:
            data_loader, _, cfg.backbone.num_neighbors = train_valid_data_loader(cfg, False)
        print("Generating embeddings...")
        results = collect_embeddings(model, cfg, data_loader, feature_keys, mode, max_batches)
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        torch.save(results, cache_path)
        print(f"✓ Saved to: {cache_path}")

    # Post-process
    if mode == "stats":
        for key in results:
            results[key]['mean'] = results[key]['mean'].cuda()
            results[key]['std']  = results[key]['std'].cuda()

    elif mode == "stats_per_cloud":
        for key in results:
            for entry in results[key]:
                entry['mean'] = entry['mean'].cuda()
                entry['std']  = entry['std'].cuda()

    elif mode == "per_point" and cloud_index is not None:
        for key in results:
            mask = results[key]['labels'] == cloud_index
            results[key] = results[key]['features'][mask]   # [N_cloud, C]

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from tta_purge import load_tta_dataset

    args = make_parser().parse_args()
    cfg  = make_cfg()

    if args.output is None:
        dataset_name = getattr(cfg.data, 'dataset', 'parenet')
        args.output  = f"intermediate_features/mean_std/{dataset_name}_{'_'.join(args.features)}_intermediates.pth"

    print("=" * 60)
    print("PAREConv Embedding Statistics")
    print("=" * 60)

    model = create_model(cfg).cuda()
    state = torch.load(args.checkpoint)
    model.load_state_dict(state['model'])
    model.eval()
    print(f"✓ Model loaded from {args.checkpoint}")

    data_loader = load_tta_dataset(args, cfg)
    print("✓ Dataset loaded")

    results = load_or_collect(
        model=model,
        cfg=cfg,
        cache_path=args.output,
        feature_keys=args.features,
        data_loader=data_loader,
        mode=getattr(args, 'mode', 'stats'),
        force_regenerate=getattr(args, 'force', False),
    )

    print("\n✓ Done!")