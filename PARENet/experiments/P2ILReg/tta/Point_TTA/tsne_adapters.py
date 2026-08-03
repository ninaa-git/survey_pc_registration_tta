import numpy as np
import torch
import torch.optim as optim

from config import make_cfg
from common_utils.parser import make_parser
from pareconv.utils.torch import to_cuda

import tsne_core as core


def _tag(args, cfg):
    return f"{cfg.dataset_name}_{args.corruption}_{args.severity}"


def _load_model_and_ckpt(model, args):
    """Load base weights from --snapshot and/or --meta_snapshot (like test.py)."""
    ckpt = {}
    if getattr(args, "snapshot", None):
        core.load_checkpoint(model, args.snapshot)
        print(f"  loaded base checkpoint: {args.snapshot}")
    if getattr(args, "meta_snapshot", None):
        ckpt = core.load_checkpoint(model, args.meta_snapshot)
        print(f"  loaded meta checkpoint: {args.meta_snapshot}")
    elif not getattr(args, "snapshot", None):
        print("  [warn] no --snapshot or --meta_snapshot — model uses random init")
    return ckpt


def _load_rec_branch(aux, ckpt):
    if "rec_branch" not in ckpt:
        return
    aux_state = {
        k: v for k, v in ckpt["rec_branch"].items()
        if not k.startswith("backbone.")
    }
    aux.load_state_dict(aux_state, strict=False)


def run(args, cfg):
    from tta.Source_Only.model import create_model
    from tta.Point_TTA.rec_aux import RecAux, RecAuxLoss
    from dataset import viz_data_loaders

    n_clouds   = args.viz_max_clouds
    sample_idx = args.viz_sample
    tag        = _tag(args, cfg)
    core.print_viz_window(sample_idx, n_clouds)
    cfg.test.batch_size  = 1
    cfg.test.num_workers = 0

    model = create_model(cfg).cuda()
    ckpt  = _load_model_and_ckpt(model, args)

    sel  = lambda name, m: name == "backbone"
    root = model
    stop = model.backbone
    hook_kw = dict(capture_output_idx=3, early_stop_module=stop, sample_idx=sample_idx)

    clean_loader, cor_loader, cfg.backbone.num_neighbors = viz_data_loaders(args, cfg)

    print(f"[Point_TTA] collecting clean features ({n_clouds} cloud(s))...")
    F_clean = core.collect_with_hook(root, clean_loader, sel, to_cuda, n_clouds, **hook_kw)
    print(f"[Point_TTA] collecting corrupted features (no adapt)...")
    F_cor   = core.collect_with_hook(root, cor_loader,   sel, to_cuda, n_clouds, **hook_kw)

    aux     = RecAux(model, cfg).cuda()
    _load_rec_branch(aux, ckpt)
    aux_loss = RecAuxLoss().cuda()
    inner    = optim.SGD(
        [p for _, p in aux.named_parameters() if p.requires_grad], lr=args.lr
    )
    aux.train()
    adapted_feats = []
    for batch_i, data in core.iter_viz_batches(cor_loader, sample_idx, n_clouds):
        data = to_cuda(data)
        print(f"[Point_TTA] inner adapt cloud {batch_i - sample_idx + 1}/{n_clouds} "
              f"(pair idx {batch_i}, {args.niter} SGD steps)...")
        for step in range(args.niter):
            inner.zero_grad(set_to_none=True)
            aux_loss(aux(data), data).backward()
            inner.step()
        col = core.HookCollector(root, sel, only_last=True, capture_output_idx=3)
        _sh = stop.register_forward_hook(
            lambda *_: (_ for _ in ()).throw(core._EarlyStop())
        )
        model.eval()
        with torch.no_grad():
            try:
                _ = model(data)
            except core._EarlyStop:
                pass
        _sh.remove()
        f = col.get()
        col.remove()
        if f is not None:
            adapted_feats.append(f)
    F_adapt = np.concatenate(adapted_feats, 0) if adapted_feats else None

    core.plot_groups(F_clean, F_cor, F_adapt, tag, "Point_TTA", out_dir=cfg.viz_dir)


def main():
    args = make_parser().parse_args()
    cfg  = make_cfg()
    torch.set_float32_matmul_precision("high")
    run(args, cfg)


if __name__ == "__main__":
    main()
