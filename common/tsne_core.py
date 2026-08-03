import os
import os.path as osp
from collections import defaultdict

import numpy as np
import torch


class _EarlyStop(Exception):
    """Raised by an early-stop hook to abort model.forward() cleanly once
    the features we care about have already been captured.  Caught by the
    collection loops; never escapes to the caller."""
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ===========================================================================
# 1. Per-token feature collection
# ===========================================================================
def iter_viz_batches(loader, sample_idx=0, max_clouds=1):
    """Yield (batch_idx, data) for visualization window [sample_idx, sample_idx+max_clouds)."""
    for i, data in enumerate(loader):
        if i < sample_idx:
            continue
        if i >= sample_idx + max_clouds:
            break
        yield i, data


def print_viz_window(sample_idx, max_clouds=1):
    end = sample_idx + max_clouds - 1
    if max_clouds == 1:
        print(f"[viz] sample_idx={sample_idx}")
    else:
        print(f"[viz] sample_idx={sample_idx}..{end} ({max_clouds} pair(s))")


class HookCollector:
    """
    Capture per-token features by forward-hooking modules selected by a predicate.

    capture_input_idx: if set, captures inp[capture_input_idx] instead of the
    module output. Useful when the relevant tensor is passed as an argument
    (e.g. PEA hooks model.transformer and captures inp[2] = ref_feats_c_pad,
    the aligned coarse features before they enter the transformer).
    capture_output_idx: if set, captures out[capture_output_idx] when the module
    returns a tuple (e.g. Point_TTA hooks backbone and captures out[3] =
    ri_feats_c, the coarse features).
    """

    def __init__(self, root_module, select_fn, only_last=True,
                 seed=0, capture_input_idx=None, capture_output_idx=None,
                 ctx=None, stream=None):
        self._sel    = select_fn
        self._rng    = np.random.default_rng(seed)
        self._hooks  = []
        self._names  = {}
        self._order  = []
        self._feats  = defaultdict(list)
        self._only_last      = only_last
        self._input_idx      = capture_input_idx
        self._output_idx     = capture_output_idx
        # stream gating: keep only the ref (corrupted/target) or src stream.
        # ctx._processing_corrupted == True  -> ref stream; False -> src stream
        # (same convention as _PostAffineCollector). stream=None keeps both.
        self._ctx    = ctx
        self._stream = stream

        for name, m in root_module.named_modules():
            if select_fn(name, m):
                self._names[id(m)] = name
                self._order.append(name)
                self._hooks.append(m.register_forward_hook(self._hook))

        if not self._order:
            raise RuntimeError("HookCollector: select_fn matched no module.")

    def _hook(self, module, inp, out):
        if self._stream is not None and self._ctx is not None:
            corrupted = getattr(self._ctx, "_processing_corrupted", None)
            if self._stream == "ref" and corrupted is False:
                return
            if self._stream == "src" and corrupted is True:
                return
        name = self._names[id(module)]
        if self._only_last and name != self._order[-1]:
            return
        if self._input_idx is not None:
            feat = inp[self._input_idx]
        elif self._output_idx is not None and isinstance(out, (tuple, list)):
            feat = out[self._output_idx]
        else:
            feat = out[0] if isinstance(out, (tuple, list)) else out
        y = feat.detach().float().reshape(-1, feat.shape[-1])   # (N, C)
        self._subsample_store(name, y)

    def _subsample_store(self, name, y):
        n = y.shape[0]
        if n == 0:
            return
        self._feats[name].append(y.cpu())

    def get(self):
        layer = self._order[-1] if self._only_last else self._order[0]
        if not self._feats[layer]:
            return None
        return torch.cat(self._feats[layer], 0).numpy()

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


@torch.no_grad()
def collect_with_hook(hook_root, loader, select_fn, to_cuda,
                      max_clouds=40, sample_idx=0,
                      forward_fn=None, only_last=True,
                      capture_input_idx=None, capture_output_idx=None, seed=0,
                      early_stop_module=None, ctx=None, stream=None):
    """Generic: run `max_clouds` batches, hook-capture per-token features → [N,C].

    hook_root         : module whose named_modules() are searched for hooks.
    forward_fn(data)  : drives the forward pass. If None, calls hook_root(data).
                        Pass a lambda when hook_root is not the callable entry
                        point (e.g. LN_TTA hooks model.transformer.transformer
                        but must call the full PARE_Net).
    capture_input_idx : if set, captures inp[i] instead of the output (PEA).
    capture_output_idx: if set, captures out[i] when the hooked module returns
                        a tuple (Point_TTA: backbone out[3] = ri_feats_c).
    seed              : RNG seed for token subsampling — keep identical across
                        methods so comparisons use the same per-cloud tokens.
    early_stop_module : if set, a post-hook is registered on this module that
                        raises _EarlyStop right after it finishes, aborting the
                        forward before any later stage (e.g. hypothesis
                        generation) can fail.  The capture hooks on hook_root
                        fire first (they are registered earlier), so features
                        are always complete when the early stop triggers.
    """
    col = HookCollector(hook_root, select_fn, only_last,
                        seed=seed, capture_input_idx=capture_input_idx,
                        capture_output_idx=capture_output_idx,
                        ctx=ctx, stream=stream)

    _stop_handle = None
    if early_stop_module is not None:
        _stop_handle = early_stop_module.register_forward_hook(
            lambda *_: (_ for _ in ()).throw(_EarlyStop())
        )

    hook_root.eval()
    try:
        for i, data in iter_viz_batches(loader, sample_idx, max_clouds):
            data = to_cuda(data)
            try:
                if forward_fn is None:
                    _ = hook_root(data)
                else:
                    _ = forward_fn(data)
            except _EarlyStop:
                pass                        # expected clean abort — features captured
            except Exception as e:
                print(f"  [collect_with_hook] unexpected forward error on batch {i}: {e}")
                raise
    finally:
        col.remove()
        if _stop_handle is not None:
            _stop_handle.remove()

    return col.get()


def load_checkpoint(model, path, key="model"):
    """Load a checkpoint, transparently stripping the torch.compile _orig_mod prefix."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt[key] if key in ckpt else ckpt
    # torch.compile saves keys as "module._orig_mod.layer..." — strip the infix
    state = {k.replace("._orig_mod", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    return ckpt   # return full ckpt so callers can also read e.g. "rec_branch"


@torch.no_grad()
def collect_from_intermediate(model, loader, to_cuda, n_clouds, sample_idx=0, seed=0,
                               feat_key="ref_feats_c", mask_key=None):
    """Collect features via forward_out_intermediate with per-token z-score.

    This is the canonical extraction path for features that live at the
    backbone coarse level (ref_feats_c / b_ref_feats_c_pad).  Using the same
    function for both PEA and Purge_Gate guarantees that the 'source (clean)'
    cluster is drawn from exactly the same distribution in both plots.

    feat_key  : key in the forward_out_intermediate output dict.
                "ref_feats_c"       → flat [N_valid, C]  (PEA / Source_Only)
                "b_ref_feats_c_pad" → padded [B, N, C]   (Purge_Gate)
    mask_key  : key for the boolean mask [B, N] used to strip padding tokens.
                Pass None when the tensor is already flat.
    Per-token z-score: each token's channel vector is normalised to zero mean /
    unit std (same as Purge_Gate's collect_embeddings mode="per_point").
    """
    feats_list = []
    model.eval()
    _fwd_warned = False
    for i, data in iter_viz_batches(loader, sample_idx, n_clouds):
        data = to_cuda(data)
        try:
            out = model.forward_out_intermediate(data)
        except Exception as e:
            if not _fwd_warned:
                print(f"  [collect_from_intermediate] forward exceptions suppressed: {e}")
                _fwd_warned = True
            continue
        f    = out[feat_key].float()
        if mask_key and mask_key in out:
            f = f[out[mask_key].bool()]          # [N_valid, C] — strip padding
        # per-token z-score over channels (same as Purge_Gate collect_embeddings)
        fm = f.mean(dim=-1, keepdim=True)
        fs = f.std( dim=-1, keepdim=True)
        f  = (f - fm) / (fs + 1e-6)
        f  = f.cpu().numpy()
        feats_list.append(f)
    return np.concatenate(feats_list, 0) if feats_list else None


@torch.no_grad()
def collect_transformer_input(model, loader, to_cuda, n_clouds, sample_idx=0, seed=0,
                              transformer_attr="transformer",
                              feat_arg_idx=2, mask_kw="ref_masks", zscore=True,
                              early_stop_module=None):
    """Capture the REF-only coarse feature FED INTO the transformer.

    Both PEA and Purge_Gate call (identical signature):
        self.transformer(ref_points_c_pad, src_points_c_pad,
                         ref_feats_c_pad, src_feats_c_pad,
                         ref_masks=ref_mask_c, src_masks=src_mask_c)
    so a forward-pre-hook on `model.<transformer_attr>` gives
        args[feat_arg_idx] = ref_feats_c_pad      (ref-only, padded [B, N, C])
        kwargs[mask_kw]    = ref_mask_c           (bool [B, N])

    This is the SAME feature in both pipelines, so the 'source (clean)' cluster
    matches across the two figures (provided both models load the same backbone
    weights). For PEA the captured tensor is already WCT-aligned when alignment
    is installed, so this one function yields the correct feature in every
    state (clean / no-adapt / adapted) with no model edits.

    Per-token z-score (over channels) matches the other collectors.
    """
    grabbed = {}

    def pre_hook(module, args, kwargs):
        grabbed["feat"] = args[feat_arg_idx].detach()
        m = kwargs.get(mask_kw, None)
        grabbed["mask"] = m.detach() if m is not None else None

    tmod = getattr(model, transformer_attr)
    handle = tmod.register_forward_pre_hook(pre_hook, with_kwargs=True)

    # early-stop: abort forward right after the transformer module completes
    # (pre-hook already fired at that point, so the feature is captured)
    _stop_module = early_stop_module if early_stop_module is not None else tmod
    _stop_handle = _stop_module.register_forward_hook(
        lambda *_: (_ for _ in ()).throw(_EarlyStop())
    )

    feats_list = []
    model.eval()
    try:
        for i, data in iter_viz_batches(loader, sample_idx, n_clouds):
            grabbed.clear()
            data = to_cuda(data)
            try:
                _ = model(data)
            except _EarlyStop:
                pass                        # expected — feature already grabbed
            except Exception as e:
                print(f"  [collect_transformer_input] unexpected forward error on batch {i}: {e}")
                raise
            if "feat" not in grabbed:
                continue
            f = grabbed["feat"].float()                  # [B, N, C]
            m = grabbed["mask"]
            f = f[m.bool()] if m is not None else f.reshape(-1, f.shape[-1])
            if zscore:
                fm = f.mean(-1, keepdim=True)
                fs = f.std(-1, keepdim=True)
                f = (f - fm) / (fs + 1e-6)
            f = f.cpu().numpy()
            feats_list.append(f)
    finally:
        handle.remove()
        _stop_handle.remove()
    return np.concatenate(feats_list, 0) if feats_list else None


def subsample(arr, n, seed=0):
    if arr is None or arr.shape[0] <= n:
        return arr
    rng = np.random.default_rng(seed)
    return arr[rng.choice(arr.shape[0], n, replace=False)]


# ===========================================================================
# 2. Metrics
# ===========================================================================
def collect_rmse_metrics(forward_fn, evaluator_fn, loader, to_cuda, n_clouds,
                         sample_idx=0):
    """Collect mean registration metrics over up to n_clouds pairs.

    forward_fn(data_dict)              → output_dict
    evaluator_fn(output_dict, data)    → result_dict   (must contain RMSE/RRE/RTE/IR)

    Returns a dict {RMSE, RRE, RTE, IR, n} or None if no pair succeeded.
    No torch.no_grad() wrapper here — the caller is responsible (e.g. Point_TTA
    needs gradients for inner SGD, while LN_TTA wraps with no_grad).
    """
    lists = {"RMSE": [], "RRE": [], "RTE": [], "IR": []}
    for i, data in iter_viz_batches(loader, sample_idx, n_clouds):
        data = to_cuda(data)
        try:
            out = forward_fn(data)
            res = evaluator_fn(out, data)
            for k in lists:
                v = res.get(k)
                if v is not None:
                    lists[k].append(float(v))
        except Exception:
            pass
    if not lists["RMSE"]:
        return None
    return {k: float(np.mean(v)) for k, v in lists.items() if v} | {"n": len(lists["RMSE"])}


def _print_rmse(rmse, log=print):
    """Print no-adapt vs adapted registration metrics, one row per metric."""
    na = rmse.get("no_adapt")
    ad = rmse.get("adapted")
    n  = (na or ad or {}).get("n", "?")
    suffix = "pair" if n == 1 else "pairs"
    log(f"  registration metrics  (n={n} {suffix}):")
    for k in ("RMSE", "RRE", "RTE"):
        na_v = f"{na[k]:7.2f}" if na and k in na else "     —"
        ad_v = f"{ad[k]:7.2f}" if ad and k in ad else "     —"
        delta = ""
        if na and ad and k in na and k in ad:
            d    = ad[k] - na[k]
            sign = "✓" if d < 0 else "✗"
            delta = f"   Δ={d:+.2f} {sign}"
        log(f"    {k:<6}  no-adapt={na_v}   adapted={ad_v}{delta}")


def centroid_ratio(clean, no_adapt, adapted, log=print):
    """dist(clean, adapted) / dist(clean, no_adapt) in the original feature
    space. < 1  => adaptation moved corrupted features back toward source."""
    c0, c1, c2 = clean.mean(0), no_adapt.mean(0), adapted.mean(0)
    d_no = float(np.linalg.norm(c0 - c1))
    d_ad = float(np.linalg.norm(c0 - c2))
    r = d_ad / max(d_no, 1e-8)
    log(f"  dist(clean, no-adapt) = {d_no:.4f}")
    log(f"  dist(clean, adapted)  = {d_ad:.4f}")
    log(f"  ratio = {r:.3f}  ->  adaptation "
        f"{'closed the gap' if r < 1 else 'did NOT help (gap same/larger)'}")
    return r


# ===========================================================================
# 3. Projection + plotting
# ===========================================================================
COLOR_CLEAN_STAR   = "#FFD700"   # centroid star for source (clean)
COLOR_CLEAN_POINTS = "#2166AC"   # token markers for source (clean)
COLOR_CORRUPTED    = "#d62728"   # red
COLOR_ADAPTED      = "#2ca02c"   # green


def _is_source_clean(name):
    return name.startswith("source (clean)")


def _project(blocks, method="tsne", seed=42):
    sizes = [b.shape[0] for b in blocks]
    X = np.concatenate(blocks, 0)
    Xs = StandardScaler().fit_transform(X)
    if method == "pca":
        emb = PCA(n_components=2, random_state=seed).fit_transform(Xs)
    else:
        n = X.shape[0]
        perp = max(5, min(30, (n - 1) // 3))
        emb = TSNE(n_components=2, perplexity=perp, init="pca",
                   learning_rate="auto", random_state=seed).fit_transform(Xs)
    return emb, sizes


def _scatter(emb, sizes, names, colors, title, out_path):
    """Scatter plot with centroid markers (★) for each group.

    title is kept as a parameter for call-site compatibility but is no longer
    rendered inside the figure — see _save_legend() for the standalone legend.
    """
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    start = 0
    for nm, c, n in zip(names, colors, sizes):
        sl = slice(start, start + n)
        is_source = _is_source_clean(nm)
        pt_color  = COLOR_CLEAN_POINTS if is_source else c
        pt_marker = "^" if is_source else "o"
        pt_size   = 64 if is_source else 32
        plt.scatter(emb[sl, 0], emb[sl, 1], s=pt_size, alpha=1.0,
                    c=pt_color, marker=pt_marker)
        if n > 0:
            cx, cy = emb[sl, 0].mean(), emb[sl, 1].mean()
            star_color = COLOR_CLEAN_STAR if is_source else c
            plt.scatter([cx], [cy], marker="*", s=1605, c=star_color,
                        edgecolors="black", linewidths=0.8, zorder=6)
        start += n
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_frame_on(False)
    plt.tight_layout(pad=0)
    os.makedirs(osp.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close()
    print(f"  saved {out_path}")


def _save_legend(names, colors, out_path):
    """Standalone legend PNG for a set of scatter plots.

    Shows a filled circle for each group and one entry explaining the ★ centroid
    marker.  Sized automatically to the number of groups.
    """
    from matplotlib.lines import Line2D
    handles = []
    for nm, c in zip(names, colors):
        if _is_source_clean(nm):
            handles.append(Line2D(
                [0], [0], marker="^", color="none", markerfacecolor=COLOR_CLEAN_POINTS,
                markersize=32, alpha=1.0, label=nm,
            ))
        else:
            handles.append(Line2D(
                [0], [0], marker="o", color="none", markerfacecolor=c,
                markersize=36, alpha=1.0, label=nm,
            ))
    handles.append(Line2D(
        [0], [0], marker="*", color="none", markerfacecolor=COLOR_CLEAN_STAR,
        markersize=24, markeredgecolor="black", markeredgewidth=0.8,
        label="★  group centroid",
    ))
    fig = plt.figure(figsize=(3.8, len(handles) * 0.42 + 0.3))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.legend(handles=handles, loc="center", fontsize=9,
              frameon=True, framealpha=0.95)
    os.makedirs(osp.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {out_path}")


def plot_groups(clean, no_adapt, adapted, tag, method_name,
                out_dir="viz", balance=True, seed=42, rmse=None):
    """Produce 3 pairs of figures (pca + tsne) for LN / PEA / Point-TTA.

    _3way : ONE joint fit on all 3 groups — shows relative positions.
    _gap  : independent 2-group fit on {source, no-adapt} — maximises visible
            gap; PEA and Purge_Gate gap plots use the same ref_feats_c so their
            source clusters look the same.
    _adapt: independent 2-group fit on {source, adapted}.

    All figures include centroid markers (✕).

    Saved files per call (6 total):
      {proj}_{method}_{tag}_3way.png    — source / no-adapt / adapted
      {proj}_{method}_{tag}_gap.png     — source + no-adapt   (domain gap)
      {proj}_{method}_{tag}_adapt.png   — source + adapted    (adaptation effect)

    rmse : optional dict {"no_adapt": {...}, "adapted": {...}} as returned by
           collect_rmse_metrics().  When provided, registration metrics (RMSE /
           RRE / RTE) are printed after the centroid-distance summary.
    """
    print(f"\n[{method_name} | {tag}] centroid distances:")
    centroid_ratio(clean, no_adapt, adapted)
    if rmse is not None:
        _print_rmse(rmse)

    if balance:
        n = min(clean.shape[0], no_adapt.shape[0], adapted.shape[0])
        clean    = subsample(clean,    n, seed)
        no_adapt = subsample(no_adapt, n, seed)
        adapted  = subsample(adapted,  n, seed)

    n0, n1, n2 = len(clean), len(no_adapt), len(adapted)
    names  = ["source (clean)", "corrupted (no adapt)", f"corrupted ({method_name})"]
    colors = [COLOR_CLEAN_STAR, COLOR_CORRUPTED, COLOR_ADAPTED]

    for proj in ("pca", "tsne"):
        # ── 3-way: joint fit (shows relative positions of all 3 groups) ──────
        emb3, _ = _project([clean, no_adapt, adapted], proj, seed)
        _scatter(emb3, [n0, n1, n2], names, colors,
                 f"{proj.upper()} — {method_name} — {tag} — 3-way",
                 osp.join(out_dir, f"{proj}_{method_name}_{tag}_3way.png"))

        # ── gap: independent 2-group fit (source vs no-adapt only) ───────────
        # Fitting on only 2 groups maximises the visible separation.
        # Both PEA and Purge_Gate use the same ref_feats_c + same seed, so
        # their _gap plots will have the same source cluster shape.
        emb_g, sg = _project([clean, no_adapt], proj, seed)
        _scatter(emb_g, sg,
                 [names[0], names[1]], [colors[0], colors[1]],
                 f"{proj.upper()} — {method_name} — {tag} — gap",
                 osp.join(out_dir, f"{proj}_{method_name}_{tag}_gap.png"))

        # ── adapt: independent 2-group fit (source vs adapted only) ──────────
        emb_a, sa = _project([clean, adapted], proj, seed)
        _scatter(emb_a, sa,
                 [names[0], names[2]], [colors[0], colors[2]],
                 f"{proj.upper()} — {method_name} — {tag} — adapted",
                 osp.join(out_dir, f"{proj}_{method_name}_{tag}_adapt.png"))

    _save_legend(names, colors, osp.join(out_dir, f"legend_{method_name}.png"))


def plot_two_groups(clean, corrupted, tag, method_name,
                    out_dir="viz", balance=True, seed=42):
    """2-group figure for Source_Only (no adaptation baseline).

    Saved: pca_{method}_{tag}_gap.png  and  tsne_{method}_{tag}_gap.png
    """
    c0, c1 = clean.mean(0), corrupted.mean(0)
    dist = float(np.linalg.norm(c0 - c1))
    print(f"\n[{method_name} | {tag}] dist(clean centroid, corrupted centroid) = {dist:.4f}")
    two_names  = ["source (clean)", "corrupted (no adapt)"]
    two_colors = [COLOR_CLEAN_STAR, COLOR_CORRUPTED]
    if balance:
        n = min(clean.shape[0], corrupted.shape[0])
        clean, corrupted = subsample(clean, n, seed), subsample(corrupted, n, seed)
    for proj in ("pca", "tsne"):
        emb, sizes = _project([clean, corrupted], proj, seed)
        _scatter(emb, sizes, two_names, two_colors,
                 f"{proj.upper()} — {method_name} — {tag}",
                 osp.join(out_dir, f"{proj}_{method_name}_{tag}_gap.png"))
    _save_legend(two_names, two_colors, osp.join(out_dir, f"legend_{method_name}.png"))


def plot_selection(clean, kept, purged, tag, out_dir="viz", target_n=None, seed=42):
    """Purge-Gate: kept vs purged tokens on corrupted, overlaid on clean.

    Fits ONE joint embedding on all 3 groups and produces 3 figures by
    subsetting the SAME coordinates — the source cluster is in the same
    position across all plots.

    Saved files per call (6 total):
      {proj}_PurgeGate_{tag}_3way.png   — source / kept / purged
      {proj}_PurgeGate_{tag}_gap.png    — source + ALL corrupted (kept+purged)
      {proj}_PurgeGate_{tag}_kept.png   — source + kept only     (good tokens)

    target_n : if given, each group is independently subsampled to at most
               target_n points (so groups keep their natural proportions while
               being capped at the same size as PEA/LN/Point-TTA plots).
               If None, groups are used as-is.
    """
    if target_n is not None:
        clean  = subsample(clean,  target_n, seed)
        kept   = subsample(kept,   target_n, seed)
        purged = subsample(purged, target_n, seed)

    c0 = clean.mean(0)
    print(f"\n[Purge-Gate | {tag}] mean dist to clean centroid:")
    print(f"  kept   = {np.linalg.norm(kept   - c0, axis=1).mean():.4f}")
    print(f"  purged = {np.linalg.norm(purged - c0, axis=1).mean():.4f}  "
          f"(should be larger if purge targets shifted tokens)")

    n0, n1, n2 = len(clean), len(kept), len(purged)
    names  = ["source (clean)", "corrupted (kept)", "corrupted (purged)"]
    colors = [COLOR_CLEAN_STAR, COLOR_ADAPTED, COLOR_CORRUPTED]

    for proj in ("pca", "tsne"):
        # ── 3-way: joint fit (all groups together) ───────────────────────────
        emb3, _ = _project([clean, kept, purged], proj, seed)
        _scatter(emb3, [n0, n1, n2], names, colors,
                 f"{proj.upper()} — PurgeGate — {tag} — 3-way",
                 osp.join(out_dir, f"{proj}_PurgeGate_{tag}_3way.png"))

        # ── gap: independent fit on source + ALL corrupted tokens ────────────
        corrupted = np.vstack([kept, purged])
        emb_g, sg = _project([clean, corrupted], proj, seed)
        _scatter(emb_g, sg,
                 ["source (clean)", "corrupted (all tokens)"],
                 ["#1f77b4", "#d62728"],
                 f"{proj.upper()} — PurgeGate — {tag} — gap",
                 osp.join(out_dir, f"{proj}_PurgeGate_{tag}_gap.png"))

        # ── kept: independent fit on source + kept tokens only ───────────────
        emb_k, sk = _project([clean, kept], proj, seed)
        _scatter(emb_k, sk,
                 ["source (clean)", "corrupted (kept — after purge)"],
                 ["#1f77b4", "#d62728"],
                 f"{proj.upper()} — PurgeGate — {tag} — kept",
                 osp.join(out_dir, f"{proj}_PurgeGate_{tag}_kept.png"))

    _save_legend(names, colors, osp.join(out_dir, "legend_PurgeGate.png"))