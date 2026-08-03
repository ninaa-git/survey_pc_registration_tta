from pareconv.datasets.registration.p2psilico.dataset import P2PSilicoDataset
from pareconv.datasets.registration.p2psilico.dataset import CorruptedP2PSilicoDataset
from pareconv.utils.data import (
    registration_collate_fn_stack_mode,
    calibrate_neighbors_stack_mode,
    build_dataloader_stack_mode,
)

def train_valid_data_loader(cfg, distributed):
    train_dataset = P2PSilicoDataset(
        cfg.data.dataset_root,
        'train',
        point_limit=cfg.train.point_limit,
        use_augmentation=cfg.train.use_augmentation,
        augmentation_noise=cfg.train.augmentation_noise,
        augmentation_rotation=cfg.train.augmentation_rotation, 
        vox_size=cfg.data.voxel_size,
        min_vis=cfg.train.min_vis,
        max_vis=cfg.train.max_vis,
        overfit=cfg.train.overfit,
        n_val_ratio = cfg.train.n_val_ratio,
        seed = cfg.train.val_seed
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
        precompute_data=True
    )

    valid_dataset = P2PSilicoDataset(
        cfg.data.dataset_root,
        'val',
        point_limit=cfg.test.point_limit,
        vox_size=cfg.data.voxel_size,
        n_val_ratio = cfg.train.n_val_ratio,
        seed = cfg.train.val_seed
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
        precompute_data=True
    )

    return train_loader, valid_loader, cfg.backbone.num_neighbors


def test_data_loader(cfg, benchmark):
    test_dataset = P2PSilicoDataset(
        cfg.data.dataset_root, 
        benchmark,
        point_limit=cfg.test.point_limit,
        use_augmentation=cfg.train.use_augmentation,
        augmentation_noise=cfg.train.augmentation_noise,
        augmentation_rotation=cfg.train.augmentation_rotation,
        vox_size=cfg.data.voxel_size, 
        min_vis=cfg.train.min_vis,
        max_vis=cfg.train.max_vis,
        overfit=cfg.train.overfit,
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
    if args.dataset != "P2PSilico":
        raise NotImplementedError(f"TTA for {args.dataset} is not implemented")

    inference_dataset = CorruptedP2PSilicoDataset(args, cfg.data.cor_dataset_root)
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

def corrupted_ft_trainval_data_loader(args, cfg):
    if args.dataset != "P2PSilico":
        raise NotImplementedError(f"TTA for {args.dataset} is not implemented")

    inference_dataset = CorruptedP2PSilicoDataset(args, cfg.data.cor_dataset_root)
    ft_trainval_loader = build_dataloader_stack_mode(
        inference_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.num_neighbors,
        cfg.backbone.subsample_ratio,
        batch_size=cfg.test.batch_size,
        num_workers=cfg.test.num_workers,
        shuffle=True,
    )

    if args.corruption == "clean":
        print(f"Loading data from: {cfg.data.dataset_root} (clean)")
    else:
        print(f"Loading data from corruption: {args.corruption}, level: {args.severity}")

    return ft_trainval_loader, cfg.backbone.num_neighbors

