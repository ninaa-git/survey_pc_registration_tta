import types

from pareconv.datasets.registration.p2ilreg.dataset import PoseDataset
from pareconv.datasets.registration.p2ilreg.dataset import CorruptedP2ILRegDataset
from pareconv.utils.data import (
    registration_collate_fn_stack_mode,
    calibrate_neighbors_stack_mode,
    build_dataloader_stack_mode,
)


def _benchmark_to_real_syn(benchmark):
    b = benchmark.lower()
    if b.endswith('real'):
        return 'real'
    if b.endswith('syn') or b.endswith('silico'):
        return 'syn'
    raise ValueError(
        f"Unknown benchmark '{benchmark}'. "
        "Expected one of: 'test_syn', 'test_real' (or 'syn'/'real')."
    )


def train_valid_data_loader(cfg, distributed):
    # Training is always synthetic.
    train_dataset = PoseDataset(
        mode='train',
        config=cfg,
        data_augmentation=True,
    )

    train_loader = build_dataloader_stack_mode(
        train_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.num_neighbors,
        cfg.backbone.subsample_ratio,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        distributed=distributed,
        precompute_data=True,
    )

    valid_dataset = PoseDataset(
        mode='val',
        config=cfg,
        data_augmentation=False,
    )
    valid_loader = build_dataloader_stack_mode(
        valid_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.num_neighbors,
        cfg.backbone.subsample_ratio,
        batch_size=cfg.test.batch_size,
        num_workers=cfg.test.num_workers,
        shuffle=False,
        distributed=distributed,
        precompute_data=True,
    )

    return train_loader, valid_loader, cfg.backbone.num_neighbors


def test_data_loader(cfg, benchmark):
    real_syn = _benchmark_to_real_syn(benchmark)
    test_dataset = PoseDataset(
        mode='test',
        config=cfg,
        data_augmentation=False,
    )
    test_loader = build_dataloader_stack_mode(
        test_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.num_neighbors,
        cfg.backbone.subsample_ratio,
        batch_size=cfg.test.batch_size,
        num_workers=cfg.test.num_workers,
        shuffle=False,
    )

    return test_loader, cfg.backbone.num_neighbors

def corrupted_test_data_loader(args, cfg):
    inference_dataset = CorruptedP2ILRegDataset(args, cfg.data.cor_dataset_root)
    tta_loader = build_dataloader_stack_mode(
        inference_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.num_neighbors,
        cfg.backbone.subsample_ratio,
        batch_size=cfg.test.batch_size,
        num_workers=cfg.test.num_workers,
        shuffle=False,
    )

    if args.corruption == "clean":
        print(f"Loading data from: {cfg.data.dataset_root} (clean)")
    else:
        print(f"Loading data from corruption: {args.corruption}, level: {args.severity}")

    return tta_loader, cfg.backbone.num_neighbors


def viz_data_loaders(args, cfg):
    """Clean + corrupted loaders for t-SNE viz (mirrors test.py).

    real  → source (clean) from test_data_loader; corrupted from corruptions.
    syn   → both from corrupted_test_data_loader (clean / corrupted splits).
    """
    if cfg.data.dataset == "real":
        clean_loader, neighbor_limits = test_data_loader(cfg, cfg.data.dataset)
    else:
        clean_args = types.SimpleNamespace(**vars(args))
        clean_args.corruption = "clean"
        clean_args.severity = None
        clean_loader, neighbor_limits = corrupted_test_data_loader(clean_args, cfg)

    cor_loader, _ = corrupted_test_data_loader(args, cfg)
    return clean_loader, cor_loader, neighbor_limits


def viz_data_loaders(args, cfg):
    """Clean + corrupted loaders for t-SNE viz (mirrors test.py).

    real  → source (clean) from test_data_loader; corrupted from corruptions.
    syn   → both from corrupted_test_data_loader (clean / corrupted splits).
    """
    if cfg.data.dataset == "real":
        clean_loader, neighbor_limits = test_data_loader(cfg, cfg.data.dataset)
    else:
        clean_args = types.SimpleNamespace(**vars(args))
        clean_args.corruption = "clean"
        clean_args.severity = None
        clean_loader, neighbor_limits = corrupted_test_data_loader(clean_args, cfg)

    cor_loader, _ = corrupted_test_data_loader(args, cfg)
    return clean_loader, cor_loader, neighbor_limits