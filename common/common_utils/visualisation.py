import os
import os.path as osp

import numpy as np
from PIL import Image, ImageChops
import matplotlib
matplotlib.use('Agg')  # headless, must be before pyplot import
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import torch


def to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().float().numpy()
    return np.array(x, dtype=np.float32)


def subsample(pts, n=10000):
    if len(pts) > n:
        idx = np.random.choice(len(pts), n, replace=False)
        return pts[idx]
    return pts


def crop_figure(path, tolerance=10):
    img  = Image.open(path).convert('RGB')
    bg   = Image.new('RGB', img.size, (255, 255, 255))
    diff = ImageChops.difference(img, bg)
    diff = diff.point(lambda x: 0 if x < tolerance else 255)  # ignore near-white
    bbox = diff.getbbox()
    if bbox:
        img.crop(bbox).save(path)
        print(f"Cropped: {path}")
    else:
        print(f"Skipped (blank?): {path}")


def to_opencv_cam(pts, ocv2blender, bbx_center=None, scale=None):
    """Physical tgt-frame points -> OpenCV camera frame (same chain as overlay)."""
    pts = np.asarray(pts, dtype=np.float64)
    R = np.asarray(ocv2blender, dtype=np.float64)
    out = pts @ R.T
    if bbx_center is not None and scale is not None:
        c = np.asarray(bbx_center, dtype=np.float64).reshape(3)
        s = float(np.asarray(scale).reshape(-1)[0])
        out = (out - c) * (1.0 / s) + c
    return out


def cam_to_mpl(pts):
    """OpenCV cam (X right, Y down, Z forward) -> matplotlib axes so that
    view_init(elev=0, azim=-90) looks along +Z with image-up on screen."""
    pts = np.asarray(pts, dtype=np.float64)
    return np.column_stack([pts[:, 0], pts[:, 2], -pts[:, 1]])


def set_mpl_cam_view(ax, pts):
    """Lock a 3D axes to the OpenCV camera viewpoint (after cam_to_mpl)."""
    pts = np.asarray(pts, dtype=np.float64)
    ax.view_init(elev=0, azim=-90)
    mins, maxs = pts.min(0), pts.max(0)
    centers = 0.5 * (mins + maxs)
    radius = 0.5 * float(np.max(maxs - mins) + 1e-8)
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    ax.set_axis_off()

def visualise_registration(output_dict, data_dict, iteration, viz_dir, corruption, severity):
    os.makedirs(viz_dir, exist_ok=True)

    # Retrieve data
    ref_points   = to_np(output_dict['ref_points'])
    src_points   = to_np(output_dict['src_points'])
    est_T        = to_np(output_dict['estimated_transform'])
    ref_corr_pts = to_np(output_dict['ref_corr_points'])
    src_corr_pts = to_np(output_dict['src_corr_points'])
    corr_scores  = to_np(output_dict['corr_scores'])

    s_c = to_np(data_dict['s_c'])
    t_c = to_np(data_dict['t_c'])
    m   = float(data_dict['m'].detach().cpu() if isinstance(data_dict['m'], torch.Tensor) else data_dict['m'])

    R_est = est_T[:3, :3]
    t_est = est_T[:3, 3]

    # Denormalise
    ref_real      = ref_points * m + t_c
    src_reg       = (src_points @ R_est.T + t_est) * m + t_c
    ref_corr_real = ref_corr_pts * m + t_c
    src_corr_reg  = (src_corr_pts @ R_est.T + t_est) * m + t_c

    src_mkrs_norm = to_np(data_dict['src_markers'])
    tgt_mkrs_norm = to_np(data_dict['tgt_markers'])
    tgt_mkrs_real = tgt_mkrs_norm * m + t_c
    src_mkrs_reg  = (src_mkrs_norm @ R_est.T + t_est) * m + t_c

    ref_plot = subsample(ref_real)
    src_plot = subsample(src_reg)

    # Top-200 correspondences by score
    topk    = min(200, len(corr_scores))
    top_idx = np.argsort(corr_scores)[-topk:]
    ref_c   = ref_corr_real[top_idx]
    src_c   = src_corr_reg[top_idx]
    scores  = corr_scores[top_idx]

    src_color = [[0, 150 / 255, 1.0]]

    # Figure 1: Registered point clouds
    fig = plt.figure(figsize=(12, 8))
    ax  = fig.add_subplot(111, projection='3d')
    ax.scatter(*ref_plot.T, s=2, c='red',      alpha=0.3, label='Reference')
    ax.scatter(*src_plot.T, s=2, c=src_color,  alpha=0.7, label='Source (registered)')
    ax.set_axis_off()

    all_pts = np.vstack([ref_plot, src_plot])
    ax.set_xlim(all_pts[:, 0].min(), all_pts[:, 0].max())
    ax.set_ylim(all_pts[:, 1].min(), all_pts[:, 1].max())
    ax.set_zlim(all_pts[:, 2].min(), all_pts[:, 2].max())

    plt.tight_layout()
    fig1_path = osp.join(viz_dir, f'{corruption}_{severity}_iter{iteration:04d}_1_registered.png')
    plt.savefig(fig1_path, dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f"[viz] saved iter{iteration:04d}_1_registered.png")

    # Figure 2: Gap view + match lines
    extent = ref_real.max(0) - ref_real.min(0)
    gap    = np.array([float(extent[0]) * 1.6, 0.0, 0.0])

    fig = plt.figure(figsize=(16, 8))
    ax  = fig.add_subplot(111, projection='3d')
    ax.scatter(*ref_plot.T,          s=2, c='red',      alpha=0.4, label='Reference')
    ax.scatter(*(src_plot + gap).T,  s=2, c=src_color,  alpha=0.4, label='Source (registered)')

    norm_s = (scores - scores.min()) / (scores.ptp() + 1e-8)
    for i in range(len(ref_c)):
        p1, p2 = ref_c[i], src_c[i] + gap
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                color='orange', linewidth=0.4, alpha=0.7)

    all_pts = np.vstack([ref_plot, src_plot + gap])
    ax.set_xlim(all_pts[:, 0].min(), all_pts[:, 0].max())
    ax.set_ylim(all_pts[:, 1].min(), all_pts[:, 1].max())
    ax.set_zlim(all_pts[:, 2].min(), all_pts[:, 2].max())
    ax.set_axis_off()

    plt.tight_layout()
    fig2_path = osp.join(viz_dir, f'{corruption}_{severity}_iter{iteration:04d}_2_matches_with_space.png')
    plt.savefig(fig2_path, dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f"[viz] saved iter{iteration:04d}_2_matches_with_space.png")

    crop_figure(fig1_path)
    crop_figure(fig2_path)

def weighted_sim3(A, B, w=None, eps=1e-8):
    """A,B: [N,3] correspondences (src, ref). w: [N]. Returns (s, R, t): B ~= s*R@A + t."""
    if w is None:
        w = torch.ones(A.shape[0], device=A.device, dtype=A.dtype)
    w = w / (w.sum() + eps)
    Ac = A - (w[:, None] * A).sum(0)            # centered (weighted)
    Bc = B - (w[:, None] * B).sum(0)
    H = (w[:, None] * Ac).t() @ Bc              # [3,3] weighted cross-cov
    U, S, Vt = torch.linalg.svd(H.float())
    d = torch.det((Vt.t() @ U.t()))
    D = torch.diag(torch.tensor([1., 1., d], device=A.device))
    R = (Vt.t() @ D @ U.t()).to(A.dtype)        # [3,3]
    var_A = (w * (Ac ** 2).sum(1)).sum()        # weighted source variance
    s = (S * torch.tensor([1., 1., d], device=A.device)).sum().to(A.dtype) / (var_A + eps)
    A_mean = (w[:, None] * A).sum(0)
    B_mean = (w[:, None] * B).sum(0)
    t = B_mean - s * (R @ A_mean)
    return s, R, t

def visualise_registration_SelfP2IR(output_dict, data_dict, iteration, viz_dir):
    from pareconv.modules.ops.transformation import apply_transform
    from utils.vis import save_mesh_overlay_png, save_registration_html

    batched = output_dict['batched']
    B = output_dict['batch_size']
    device = batched['ref_feats_c_padded'].device

    T_b = batched['transforms']
    est_b = batched['estimated_transforms']

    def pp(key, b):
        v = data_dict[key]
        if isinstance(v, (list, tuple)):
            return v[b]          # B > 1: list of per-sample values
        if B == 1:
            return v             # B == 1: collate unwrapped to the bare value
        return v[b]              # fallback: stacked tensor/ndarray, first dim B

    src_pts_per_pair = batched['src_points_per_pair']  # list of [Ns, 3] (normalized)
    # Full target points per pair for CD. Prefer the full cloud; fall back to
    # the correspondence points if the full cloud is not exposed by the model.
    if 'ref_points_per_pair' in batched:
        tgt_pts_per_pair = batched['ref_points_per_pair']
    else:
        # NOTE: ref_corr_points_per_pair is only the correspondence subset, so
        # this CD is biased toward inliers. Expose 'ref_points_per_pair' from
        # the model for a full-cloud CD that matches the self-p2ir reference.
        tgt_pts_per_pair = batched['ref_corr_points_per_pair']


    for b in range(B):
        est = est_b[b]                       # [4,4] normalized: src_u -> tgt_u
        sp = src_pts_per_pair[b]             # [Ns,3] normalized source
        tgt = tgt_pts_per_pair[b]            # [Nt,3] normalized target

        s_c = torch.as_tensor(pp('s_c', b), dtype=torch.float32, device=device).reshape(3)
        t_c = torch.as_tensor(pp('t_c', b), dtype=torch.float32, device=device).reshape(3)
        m   = float(pp('m', b))

        # Denormalized estimate mapping PHYSICAL src -> PHYSICAL tgt frame.
        #   tgt_u = R src_u + t_norm,  src_u=(src-s_c)/m,  tgt_u=(tgt-t_c)/m
        #   => tgt = R src + (m*t_norm + t_c - R s_c)

        # xp scale
        ref_corr_n = batched['ref_corr_points_per_pair'][b]              # [M,3] normalized (tgt frame)
        src_corr_n = batched['src_corr_points_per_pair'][b]              # [M,3] normalized (src frame)
        #s, R, t_norm = weighted_sim3(src_corr_n, ref_corr_n) #xp scale
        #R = s*R
        #print(f'Scale: {s}')

        R = est[:3, :3] #pas xp scale
        t_norm = est[:3, 3]


        est_real = torch.eye(4, dtype=est.dtype, device=device)
        est_real[:3, :3] = R
        est_real[:3, 3] = m * t_norm + t_c - torch.matmul(R, s_c)

        real_world_src = sp * m + s_c                                 # [Ns,3] physical source
        real_world_tgt = tgt * m + t_c                               # [Nt,3] physical target
        registered_src = apply_transform(real_world_src, est_real)   # [Ns,3] registered source


        which_patient = pp('scene_name', b)   # if KeyError, use pp('which_patient', b)
        # ---------------- silhouette Dice ----------------
        is_test = ('liver_label' in data_dict)
        if is_test:
            ocv2blender = torch.as_tensor(pp('ocv2blender', b), dtype=torch.float32, device=device)        # [3,3]
            bbx_center  = torch.as_tensor(pp('bbx_center', b), dtype=torch.float32, device=device).reshape(3)
            scale       = torch.as_tensor(pp('scale', b), dtype=torch.float32, device=device).reshape(-1)[0]
            cam_k       = torch.as_tensor(pp('cam_k', b), dtype=torch.float32, device=device)              # [3,3]                                # [1,Ns,3] (batched)

            # VISUALISATIONS

            # ---- 2D: project the registered reconstructed mesh onto the image ----
            # transform the mesh's OWN vertices (faces index into these) into the
            # OpenCV camera frame, identical to the rendered-mask chain above.
            verts0 = torch.as_tensor(pp('preope_reconstructed_verts', b),
                                        dtype=torch.float32, device=device)         # [V,3] metres
            verts0 = apply_transform(verts0, est_real)                            # -> physical tgt frame
            vv = torch.matmul(verts0, ocv2blender.t())                            # -> OpenCV camera frame
            vv = (vv - bbx_center) * (1.0 / scale) + bbx_center

            # Blend intraoperative liver-label pixels in red (alpha=0.3), then
            # draw the preoperative mesh in white on top.
            img = np.asarray(pp('imgs', b).cpu().numpy()).astype(np.uint8)        # [H,W,3]
            liver = to_np(pp('liver_label', b))
            if liver.ndim == 3 and liver.shape[-1] in (3, 4):
                liver_mask = liver.sum(axis=-1) > 0
            else:
                liver_mask = liver > 0
            if liver_mask.any():
                alpha_lbl = 0.3
                img = img.copy()
                img[liver_mask, 0] = (
                    (1.0 - alpha_lbl) * img[liver_mask, 0].astype(np.float32)
                    + alpha_lbl * 255.0
                ).astype(np.uint8)
                img[liver_mask, 1] = ((1.0 - alpha_lbl) * img[liver_mask, 1].astype(np.float32)).astype(np.uint8)
                img[liver_mask, 2] = ((1.0 - alpha_lbl) * img[liver_mask, 2].astype(np.float32)).astype(np.uint8)

            save_mesh_overlay_png(
                f"{viz_dir}/overlay_p{which_patient}_b{b}_index{iteration}.png",
                img,
                cam_k.detach().cpu().numpy(),                                     # K [3,3]
                vv.detach().cpu().numpy(),                                        # verts (camera frame)
                np.asarray(pp('preope_reconstructed_faces', b).cpu().numpy()).astype(np.int32),  # faces
                line_color=(255, 255, 255),  # white wireframe
                alpha=0.8,
            )

        # ---- 3D/3D: interactive registration check (standalone HTML) ----
        # GT-registered source (rigid GT, denormalized like est_real) so the
        # HTML shows prediction (green) vs ground truth (orange) vs target (blue).
        Tg = torch.as_tensor(pp('transform_for_gt', b).cpu().numpy(), dtype=torch.float32, device=device)
        Rg, tg = Tg[:3, :3], Tg[:3, 3]
        gt_real = torch.eye(4, dtype=torch.float32, device=device)
        gt_real[:3, :3] = Rg
        gt_real[:3, 3]  = m * tg + t_c - torch.matmul(Rg, s_c)
        gt_registered_src = apply_transform(real_world_src, gt_real)

        save_registration_html(
                f"{viz_dir}/registration_p{which_patient}_b{b}_index{iteration}.html",
                [("target (intra-op)",        real_world_tgt.detach().cpu().numpy(),     "royalblue"),
                    #("registered source (est)",  registered_src.detach().cpu().numpy(),     "limegreen"),
                    #("registered source (GT)",    gt_registered_src.detach().cpu().numpy(), "limegreen")
                    ],
                title=f"Registration p{which_patient} b{b}  index={iteration}",
            )

        # ---- 3D/3D: correspondence (matches) view ----
        # corr points come from the model in NORMALIZED frame; denormalize +
        # register them into the same physical target frame as the clouds.
        ref_corr_n = batched['ref_corr_points_per_pair'][b]              # [M,3] normalized (tgt frame)
        src_corr_n = batched['src_corr_points_per_pair'][b]              # [M,3] normalized (src frame)
        # optional per-correspondence score (expose 'corr_scores_per_pair'
        # in the model's `batched`; None -> uniform lines, first max_matches)
        scores_b = batched.get('corr_scores_per_pair', None)
        scores_np = None if scores_b is None else \
            np.asarray(scores_b[b].detach().cpu()).reshape(-1)

        

        if ref_corr_n.shape[0] > 0:
            
            R_n, t_n = est[:3, :3], est[:3, 3]
            #R_n, t_n = R, t_norm #xp scale
            
            ref_corr_real = (ref_corr_n * m + t_c)                        # [M,3] physical tgt
            src_corr_reg  = (torch.matmul(src_corr_n, R_n.t()) + t_n) * m + t_c  # registered -> tgt frame
            ref_corr_real = ref_corr_real.detach().cpu().numpy()
            src_corr_reg  = src_corr_reg.detach().cpu().numpy()

            tgt_np = real_world_tgt.detach().cpu().numpy()
            src_np = registered_src.detach().cpu().numpy()

            # Align viewpoint with the registered overlay image: same OpenCV
            # camera frame + look along the optical axis.
            html_camera = None
            if is_test:
                R_ocv = ocv2blender.detach().cpu().numpy()
                c_bbx = bbx_center.detach().cpu().numpy()
                s_bbx = scale.detach().cpu().numpy()
                tgt_np = to_opencv_cam(tgt_np, R_ocv, c_bbx, s_bbx)
                src_np = to_opencv_cam(src_np, R_ocv, c_bbx, s_bbx)
                ref_corr_real = to_opencv_cam(ref_corr_real, R_ocv, c_bbx, s_bbx)
                src_corr_reg = to_opencv_cam(src_corr_reg, R_ocv, c_bbx, s_bbx)
                # Plotly: eye behind camera looking toward +Z, Y down (= image up)
                html_camera = {
                    "eye": {"x": 0.0, "y": 0.0, "z": -2.0},
                    "up":  {"x": 0.0, "y": -1.0, "z": 0.0},
                    "center": {"x": 0.0, "y": 0.0, "z": 0.0},
                }

            gap = np.array([(tgt_np[:, 0].max() - tgt_np[:, 0].min()) * 0.0, 0.0, 0.0])
            src_shifted = src_np + gap
            src_corr_shifted = src_corr_reg + gap
            n_corr = ref_corr_real.shape[0]
            match_line_w = 2.0 / 3.0  # 3x thinner than the previous default

            save_registration_html(
                f"{viz_dir}/matches_p{which_patient}_b{b}_index{iteration}.html",
                [("intraoperative", tgt_np, "red"),
                 ("preoperative (registered)", src_shifted, "royalblue")],
                matches=(ref_corr_real, src_corr_shifted, scores_np),
                max_matches=None,          # draw all used correspondences
                line_width=match_line_w,
                camera=html_camera,
                title=f"Matches p{which_patient} b{b}  ({n_corr} corr)",
            )

            # Static PNG of the same matches view (camera-aligned when possible)
            if is_test:
                tgt_plot = cam_to_mpl(subsample(tgt_np))
                src_plot = cam_to_mpl(subsample(src_shifted))
                ref_plot = cam_to_mpl(ref_corr_real)
                src_c_plot = cam_to_mpl(src_corr_shifted)
            else:
                tgt_plot = subsample(tgt_np)
                src_plot = subsample(src_shifted)
                ref_plot = ref_corr_real
                src_c_plot = src_corr_shifted

            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(*tgt_plot.T, s=2, c='red', alpha=0.5, label='intraoperative')
            ax.scatter(*src_plot.T, s=2, c='royalblue', alpha=0.5, label='preoperative')
            for i in range(n_corr):
                p1, p2 = ref_plot[i], src_c_plot[i]
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                        color='orange', linewidth=0.4 / 3.0, alpha=0.6)
            all_pts = np.vstack([tgt_plot, src_plot])
            if is_test:
                set_mpl_cam_view(ax, all_pts)
            else:
                ax.set_xlim(all_pts[:, 0].min(), all_pts[:, 0].max())
                ax.set_ylim(all_pts[:, 1].min(), all_pts[:, 1].max())
                ax.set_zlim(all_pts[:, 2].min(), all_pts[:, 2].max())
                ax.set_axis_off()
            plt.tight_layout()
            png_path = osp.join(viz_dir, f'matches_p{which_patient}_b{b}_index{iteration}.png')
            plt.savefig(png_path, dpi=150, bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            crop_figure(png_path)
            print(f"[viz] saved matches_p{which_patient}_b{b}_index{iteration}.png ({n_corr} corr)")