import numpy as np 
import cv2

""" def project_mesh_to_image(mesh, K, R, t, image, alpha=0.5):
    
    将 3D 网格模型投影到 2D 图像上并以半透明方式绘制
    
    参数:
    mesh (open3d.geometry.TriangleMesh): 3D 网格模型
    K (numpy.ndarray): 相机内参矩阵
    RT (numpy.ndarray): 相机外参矩阵 (R|T)
    image (numpy.ndarray): 2D 图像
    alpha (float): 半透明度 (0~1)
   
    # 获取网格模型的顶点
    vertices = np.asarray(mesh.vertices)

    rvec, _ = cv2.Rodrigues(R)
    tvec = t.reshape(3, 1)
    
    # 将 3D 顶点投影到 2D 图像平面
    pts_2d, _ = cv2.projectPoints(vertices, rvec, tvec, K, None)
    pts_2d = np.int32(pts_2d.reshape(-1, 2))
    
    # 创建一个透明图层
    h, w, _ = image.shape
    mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    # 在透明图层上绘制网格模型
    for face in mesh.triangles:
        pt1 = pts_2d[face[0]]
        pt2 = pts_2d[face[1]]
        pt3 = pts_2d[face[2]]
        triangle = np.array([pt1, pt2, pt3], dtype=np.int32)
        cv2.fillConvexPoly(mask, triangle, (0, 255, 0))
    
    # 合并原始图像和透明图层
    result = cv2.addWeighted(image, 1 - alpha, mask, alpha, 0)
    
    return result """

def project_mesh_to_image(mesh, K, R, t, image, alpha=0.4, fill_color=(0, 255, 0), 
                         wireframe=True, line_color=(255, 255, 255), line_thickness=1):
    
    """ 将 3D 网格模型投影到 2D 图像上并以半透明方式绘制
    
    参数:
    mesh (open3d.geometry.TriangleMesh): 3D 网格模型
    K (numpy.ndarray): 相机内参矩阵
    RT (numpy.ndarray): 相机外参矩阵 (R|T)
    image (numpy.ndarray): 2D 图像
    alpha (float): 半透明度 (0~1)
    fill_color: 填充颜色 (B, G, R)
    wireframe: 是否绘制线框
    line_color: 线框颜色 (B, G, R)
    line_thickness: 线框粗细 """
   
    # 获取网格模型的顶点
    vertices = np.asarray(mesh.vertices)
    
    # Convert rotation matrix to rotation vector for cv2.projectPoints
    rvec, _ = cv2.Rodrigues(R)
    tvec = t.reshape(3, 1)
    
    # 将 3D 顶点投影到 2D 图像平面
    pts_2d, _ = cv2.projectPoints(vertices, rvec, tvec, K, None)
    pts_2d = np.int32(pts_2d.reshape(-1, 2))
    
    # 创建一个透明图层
    h, w, _ = image.shape
    overlay = image.copy()
    
    # 在透明图层上绘制网格模型
    if wireframe:
        # Draw only edges (wireframe)
        for face in mesh.triangles:
            pt1 = pts_2d[face[0]]
            pt2 = pts_2d[face[1]]
            pt3 = pts_2d[face[2]]
            
            # Draw triangle edges
            cv2.line(overlay, tuple(pt1), tuple(pt2), line_color, line_thickness)
            cv2.line(overlay, tuple(pt2), tuple(pt3), line_color, line_thickness)
            cv2.line(overlay, tuple(pt3), tuple(pt1), line_color, line_thickness)
    else:
        # Draw filled triangles
        for face in mesh.triangles:
            pt1 = pts_2d[face[0]]
            pt2 = pts_2d[face[1]]
            pt3 = pts_2d[face[2]]
            triangle = np.array([pt1, pt2, pt3], dtype=np.int32)
            cv2.fillConvexPoly(overlay, triangle, fill_color)
    
    # 合并原始图像和透明图层
    result = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
    
    return result

# ============================================================================
#  Registration visualisation helpers (added)
#  - save_mesh_overlay_png : 2D — project a mesh onto the image (wraps the
#    existing project_mesh_to_image, handles BGR->RGB + dirs)
#  - save_registration_html: 3D/3D — interactive standalone HTML to check that
#    the registered source lands on the target. Uses Plotly from CDN, so NO
#    extra python package is required (only a browser + internet to view).
# ============================================================================
import os as _os
import json as _json


def _subsample(P, n, seed=0):
    P = np.asarray(P, dtype=np.float64)
    if P.ndim != 2 or P.shape[1] != 3:
        P = P.reshape(-1, 3)
    if n is None or P.shape[0] <= n:
        return P
    rng = np.random.default_rng(seed)
    return P[rng.choice(P.shape[0], int(n), replace=False)]


def save_mesh_overlay_png(path, image, K, verts, faces, R=None, t=None, **proj_kwargs):
    """Project a mesh (verts[V,3] + faces[F,3]) onto `image` using intrinsics K
    and (by default) identity extrinsics — i.e. `verts` are assumed to be in the
    OpenCV camera frame already. Saves a PNG. Returns True on success, False if
    there is nothing to draw.

    This just moves the boilerplate out of loss.py: it builds the duck-typed mesh
    project_mesh_to_image expects, fixes the BGR->RGB channel order for matplotlib,
    and creates the output directory.
    """
    import types
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    verts = np.asarray(verts, dtype=np.float64)
    faces = np.asarray(faces).astype(np.int32)
    image = np.asarray(image).astype(np.uint8)
    if faces.shape[0] == 0 or verts.shape[0] == 0:
        return False
    if int(faces.max()) >= verts.shape[0]:
        raise ValueError(f"face index {int(faces.max())} >= #verts {verts.shape[0]}")

    if R is None:
        R = np.eye(3)
    if t is None:
        t = np.zeros(3)

    mesh = types.SimpleNamespace(vertices=verts, triangles=faces)
    result = project_mesh_to_image(mesh, np.asarray(K, np.float64), R, t, image, **proj_kwargs)
    #result = result[:, :, ::-1]  # BGR -> RGB for matplotlib
    _os.makedirs(_os.path.dirname(path) or ".", exist_ok=True)
    plt.imsave(path, result)
    return True


def save_registration_html(path, clouds, matches=None, max_matches=200,
                           line_color="orange", line_width=2.0,
                           title="Registration check", max_points=20000, point_size=1.8,
                           camera=None):
    """Write a standalone interactive 3D HTML overlaying point clouds (and,
    optionally, the correspondences) to check a 3D/3D registration.
 
    clouds : list of (name, points[N,3], color) tuples, e.g.
        [("target", tgt_np, "royalblue"),
         ("registered source (est)", reg_np, "limegreen")]
 
    matches : optional correspondence lines, one of
        (ref_corr[M,3], src_corr[M,3])            -> uniform lines, first `max_matches`
        (ref_corr[M,3], src_corr[M,3], scores[M]) -> top-`max_matches` by score,
                                                     endpoints colored by score (colorbar)
      Both endpoint arrays MUST be in the SAME frame as `clouds` (i.e. already
      denormalized and registered). Each line i connects ref_corr[i] <-> src_corr[i],
      so its length is the residual of that match. For a side-by-side "which matches
      which" view, offset the source cloud AND src_corr by the same vector before
      calling (see the loss.py usage).
      Pass max_matches=None to draw every correspondence.

    camera : optional Plotly scene camera dict, e.g. for OpenCV camera frame
        {"eye": {"x": 0, "y": 0, "z": -2}, "up": {"x": 0, "y": -1, "z": 0}}
 
    Plotly is loaded from a CDN, so no python package is required; the file only
    needs a browser (+ internet) to view. Returns True on success.
    """
    traces = []
    for i, (name, pts, color) in enumerate(clouds):
        pts = _subsample(pts, max_points, seed=i)
        if pts.shape[0] == 0:
            continue
        traces.append({
            "type": "scatter3d", "mode": "markers",
            "name": f"{name} ({pts.shape[0]})",
            "x": np.round(pts[:, 0], 5).tolist(),
            "y": np.round(pts[:, 1], 5).tolist(),
            "z": np.round(pts[:, 2], 5).tolist(),
            "marker": {"size": point_size, "color": color, "opacity": 0.65},
        })
 
    # ---- correspondence lines (optional) ----
    if matches is not None:
        ref_c = np.asarray(matches[0], dtype=np.float64).reshape(-1, 3)
        src_c = np.asarray(matches[1], dtype=np.float64).reshape(-1, 3)
        n = min(ref_c.shape[0], src_c.shape[0])
        ref_c, src_c = ref_c[:n], src_c[:n]
        scores = None
        if len(matches) >= 3 and matches[2] is not None:
            scores = np.asarray(matches[2], dtype=np.float64).reshape(-1)[:n]
 
        if n > 0:
            # keep top-k by score (or the first k if no scores); None = all
            if max_matches is not None and n > max_matches:
                if scores is not None:
                    keep = np.argsort(scores)[-max_matches:]
                    ref_c, src_c, scores = ref_c[keep], src_c[keep], scores[keep]
                else:
                    keep = np.linspace(0, n - 1, max_matches).astype(int)
                    ref_c, src_c = ref_c[keep], src_c[keep]
 
            # one trace for all lines: ref_i -> src_i -> None (break) -> ...
            lx, ly, lz = [], [], []
            for a, c in zip(ref_c, src_c):
                lx += [round(float(a[0]), 5), round(float(c[0]), 5), None]
                ly += [round(float(a[1]), 5), round(float(c[1]), 5), None]
                lz += [round(float(a[2]), 5), round(float(c[2]), 5), None]
            traces.append({
                "type": "scatter3d", "mode": "lines",
                "name": f"matches ({ref_c.shape[0]})",
                "x": lx, "y": ly, "z": lz,
                "line": {"color": line_color, "width": line_width},
                "opacity": 0.6,
            })
 
            # endpoints colored by score, with a colorbar (only if scores given)
            if scores is not None:
                mx = np.concatenate([ref_c[:, 0], src_c[:, 0]])
                my = np.concatenate([ref_c[:, 1], src_c[:, 1]])
                mz = np.concatenate([ref_c[:, 2], src_c[:, 2]])
                sc = np.concatenate([scores, scores])
                traces.append({
                    "type": "scatter3d", "mode": "markers", "name": "match score",
                    "x": np.round(mx, 5).tolist(),
                    "y": np.round(my, 5).tolist(),
                    "z": np.round(mz, 5).tolist(),
                    "marker": {"size": point_size + 1.5,
                               "color": np.round(sc, 5).tolist(),
                               "colorscale": "Viridis", "showscale": True,
                               "opacity": 0.9, "colorbar": {"title": "score"}},
                })
 
    data_json = _json.dumps(traces)
    title_json = _json.dumps(title)
    camera_json = _json.dumps(camera) if camera is not None else "undefined"
    html = (
        '<!doctype html><html><head><meta charset="utf-8">'
        f'<title>{title}</title>'
        '<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>'
        '<style>html,body{height:100%;margin:0}#g{width:100%;height:100%}</style>'
        '</head><body><div id="g"></div><script>'
        f'var data={data_json};'
        f'var cam={camera_json};'
        'var layout={title:' + title_json + ','
        "scene:{aspectmode:'data',"
        "xaxis:{title:'x (m)',showgrid:false,zeroline:false,showbackground:false,showticklabels:false,showline:false},"
        "yaxis:{title:'y (m)',showgrid:false,zeroline:false,showbackground:false,showticklabels:false,showline:false},"
        "zaxis:{title:'z (m)',showgrid:false,zeroline:false,showbackground:false,showticklabels:false,showline:false}},"
        "legend:{itemsizing:'constant'},margin:{l:0,r:0,t:40,b:0}};"
        "if(cam!==undefined){layout.scene.camera=cam;}"
        "Plotly.newPlot('g',data,layout,{responsive:true});"
        '</script></body></html>'
    )
    _os.makedirs(_os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(html)
    return True