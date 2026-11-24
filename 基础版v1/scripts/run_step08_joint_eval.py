"""F-08 Joint Evaluation (Memory-Safe Streaming + Exact Batched SVD):
1) Independent BSDS300 Test Set (100 images) across sigma in {10, 25, 50}
2) TUM RGB-D Sequences: dev (freiburg1_xyz, 792 frames) & test (freiburg3_structure_texture_far, 907 frames)
   across Original + {Noisy, Gaussian 7x7, Self-Trained ResidualCNN} x {sigma=10, 25, 50}.
"""
import gc
import hashlib
import json
import math
import sys
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.image_baseline import (
    add_gaussian,
    check_rgb,
    gaussian_filter,
    gaussian_kernel,
    image_metrics,
    load_reference,
    rgb_to_uint8,
    sha256,
)
from src.feature_frontend import detect
from src.rgbd_geometry import backproject, ransac as ref_ransac, rigid
from src.trajectory_metrics import evaluate as eval_traj, quaternion_pose
from infer_step07 import load_model, restore


def fast_gaussian_filter(rgb: np.ndarray, size: int, sigma_px: float) -> np.ndarray:
    k = gaussian_kernel(size, sigma_px)
    out = cv2.sepFilter2D(
        rgb.astype(np.float64), cv2.CV_64F, k, k, borderType=cv2.BORDER_REFLECT_101
    )
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def fast_image_metrics(ref: np.ndarray, est: np.ndarray) -> dict:
    x = ref.astype(np.float64)
    y = est.astype(np.float64)
    mse = float(np.mean((x - y) ** 2))
    psnr = "+inf" if mse == 0 else -10.0 * math.log10(mse)
    k = gaussian_kernel(11, 1.5)
    valid = np.s_[5:-5, 5:-5, :]

    def loc(im):
        return cv2.sepFilter2D(im, cv2.CV_64F, k, k, borderType=cv2.BORDER_REFLECT_101)[valid]

    mx, my = loc(x), loc(y)
    vx = np.maximum(loc(x * x) - mx * mx, 0.0)
    vy = np.maximum(loc(y * y) - my * my, 0.0)
    cov = loc(x * y) - mx * my
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    score = ((2 * mx * my + c1) * (2 * cov + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))
    return {"mse": mse, "psnr_db": psnr, "ssim": float(score.mean())}


def fast_screen_mutual(
    src_pts: list, src_desc: np.ndarray, tgt_pts: list, tgt_desc: np.ndarray, ratio: float = 0.75
):
    if len(src_desc) < 2 or len(tgt_desc) < 2:
        return []
    a_bits = np.unpackbits(src_desc, axis=1).astype(np.float32)
    b_bits = np.unpackbits(tgt_desc, axis=1).astype(np.float32)
    a_sum = a_bits.sum(axis=1, keepdims=True)
    b_sum = b_bits.sum(axis=1, keepdims=True).T
    dist = np.rint(a_sum + b_sum - 2.0 * (a_bits @ b_bits.T)).astype(np.uint16)

    fwd_order = np.argsort(dist, axis=1, kind="stable")
    fwd_j0, fwd_j1 = fwd_order[:, 0], fwd_order[:, 1]
    r_idx = np.arange(dist.shape[0])
    fwd_d0, fwd_d1 = dist[r_idx, fwd_j0], dist[r_idx, fwd_j1]
    fwd_ok = fwd_d0 < (ratio * fwd_d1)

    dist_t = dist.T
    rev_order = np.argsort(dist_t, axis=1, kind="stable")
    rev_i0, rev_i1 = rev_order[:, 0], rev_order[:, 1]
    c_idx = np.arange(dist_t.shape[0])
    rev_d0, rev_d1 = dist_t[c_idx, rev_i0], dist_t[c_idx, rev_i1]
    rev_ok = rev_d0 < (ratio * rev_d1)

    mutual = []
    for i in np.flatnonzero(fwd_ok):
        ii = int(i)
        jj = int(fwd_j0[ii])
        if rev_ok[jj] and int(rev_i0[jj]) == ii:
            mutual.append(
                {
                    "source_index": ii,
                    "target_index": jj,
                    "source_uv": src_pts[ii],
                    "target_uv": tgt_pts[jj],
                    "distance": int(fwd_d0[ii]),
                    "second_distance": int(fwd_d1[ii]),
                }
            )
    return mutual


def fast_ransac(a: np.ndarray, b: np.ndarray, key: str, config: dict) -> dict:
    """Exact batched LAPACK SVD implementation of src.rgbd_geometry.ransac."""
    seed = int.from_bytes(hashlib.sha256(("F05:" + key + ":7024").encode()).digest()[:8], "big")
    out = dict(
        valid=False,
        reason="insufficient_candidates",
        candidates=len(a),
        seed=str(seed),
        degenerate_samples=0,
    )
    if len(a) < config["min_inliers"]:
        return out

    rng = np.random.default_rng(seed)
    iters = config["iterations"]
    # Draw exact same sequence of 3-index samples as reference loop
    ids_all = np.array([rng.choice(len(a), 3, replace=False) for _ in range(iters)])

    pa = a[ids_all]  # (K, 3, 3)
    pb = b[ids_all]  # (K, 3, 3)
    ma = pa.mean(axis=1, keepdims=True)
    mb = pb.mean(axis=1, keepdims=True)
    ac = pa - ma
    bc = pb - mb

    sa = np.linalg.svd(ac, compute_uv=False)  # (K, 3)
    sb = np.linalg.svd(bc, compute_uv=False)  # (K, 3)
    non_degen = np.minimum(sa[:, 1], sb[:, 1]) >= 1e-6
    out["degenerate_samples"] = int((~non_degen).sum())

    valid_k = np.flatnonzero(non_degen)
    best = None
    score = None
    if len(valid_k) > 0:
        H = np.matmul(np.swapaxes(ac[valid_k], 1, 2), bc[valid_k])  # (V, 3, 3)
        U, _, Vt = np.linalg.svd(H)
        V = np.swapaxes(Vt, 1, 2)
        UT = np.swapaxes(U, 1, 2)
        dets = np.linalg.det(np.matmul(V, UT))
        corr = np.zeros((len(valid_k), 3, 3), dtype=np.float64)
        corr[:, 0, 0] = 1.0
        corr[:, 1, 1] = 1.0
        corr[:, 2, 2] = dets
        R_all = np.matmul(np.matmul(V, corr), UT)  # (V, 3, 3)
        t_all = mb[valid_k, 0, :] - np.einsum("vij,vj->vi", R_all, ma[valid_k, 0, :])  # (V, 3)

        # Evaluate residuals against all points a: (V, N, 3)
        pred = np.einsum("nj,vji->vni", a, np.swapaxes(R_all, 1, 2)) + t_all[:, None, :]
        errs = np.linalg.norm(pred - b[None, :, :], axis=2)  # (V, N)
        masks = errs <= config["threshold_m"]
        counts = masks.sum(axis=1)
        for idx_v in range(len(valid_k)):
            n = int(counts[idx_v])
            cand = (n, -float(errs[idx_v, masks[idx_v]].sum()))
            if score is None or cand > score:
                score = cand
                best = masks[idx_v]

    required = max(config["min_inliers"], int(math.ceil(config["min_fraction"] * len(a))))
    if best is None or best.sum() < required:
        out["reason"] = "insufficient_consensus"
        return out
    try:
        T = rigid(a[best], b[best])
        errors = np.linalg.norm((a @ T[:3, :3].T + T[:3, 3]) - b, axis=1)
        mask = errors <= config["threshold_m"]
        if mask.sum() < required:
            out["reason"] = "refit_consensus_lost"
            return out
        T = rigid(a[mask], b[mask])
        errors = np.linalg.norm((a @ T[:3, :3].T + T[:3, 3]) - b, axis=1)
        mask = errors <= config["threshold_m"]
    except ValueError as e:
        out["reason"] = str(e)
        return out

    out.update(
        valid=bool(mask.sum() >= required),
        reason="ok" if mask.sum() >= required else "final_consensus_lost",
        T_target_source=T.tolist(),
        inliers=int(mask.sum()),
        inlier_indices=np.flatnonzero(mask).tolist(),
        rmse_m=float(np.sqrt(np.mean(errors[mask] ** 2))),
    )
    return out


def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def save_json(p: Path, obj):
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def draw_match_panel(img_a_rgb, img_b_rgb, mutual, inlier_indices, title, out_path: Path):
    a_bgr = rgb_to_uint8(img_a_rgb)[..., ::-1]
    b_bgr = rgb_to_uint8(img_b_rgb)[..., ::-1]
    h, w, _ = a_bgr.shape
    canvas = np.full((h + 50, w * 2 + 20, 3), 255, dtype=np.uint8)
    canvas[45 : 45 + h, :w] = a_bgr
    canvas[45 : 45 + h, w + 20 :] = b_bgr
    inlier_set = set(inlier_indices)
    for idx, m in enumerate(mutual[:80]):
        if idx in inlier_set:
            continue
        p1 = (int(round(m["source_uv"][0])), int(round(m["source_uv"][1])) + 45)
        p2 = (int(round(m["target_uv"][0])) + w + 20, int(round(m["target_uv"][1])) + 45)
        cv2.line(canvas, p1, p2, (60, 90, 230), 1, cv2.LINE_AA)
    for idx in inlier_indices[:80]:
        m = mutual[idx]
        p1 = (int(round(m["source_uv"][0])), int(round(m["source_uv"][1])) + 45)
        p2 = (int(round(m["target_uv"][0])) + w + 20, int(round(m["target_uv"][1])) + 45)
        cv2.line(canvas, p1, p2, (40, 175, 40), 1, cv2.LINE_AA)
        cv2.circle(canvas, p1, 3, (20, 130, 20), -1, cv2.LINE_AA)
        cv2.circle(canvas, p2, 3, (20, 130, 20), -1, cv2.LINE_AA)
    cv2.putText(canvas, title, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (20, 20, 20), 2, cv2.LINE_AA)
    ok, buf = cv2.imencode(".png", canvas)
    if ok:
        buf.tofile(str(out_path))


def run_sequence_vo_streaming(
    seq_key: str,
    rows: list,
    cam: dict,
    gt_lookup: dict,
    assoc_map: dict,
    orb_cfg: dict,
    ratio: float,
    geom_cfg: dict,
    filter_cfg: dict,
    model,
    layout: str,
    base_seed: int,
    out_dir: Path,
):
    """Compact memory footprint (< 600 MB): store raw uint8 RGB + uint16 depth, stream float32 per frame!"""
    cv2.setNumThreads(1)
    cv2.setRNGSeed(7024)
    root_dir = Path(cam["root"]).resolve()

    print(f"[{seq_key}] Loading {len(rows)} compact uint8 RGB + uint16 depth pairs...", flush=True)
    raw_rgb_u8 = []
    raw_depths = []
    for row in rows:
        rp = (root_dir / row["rgb_path"]).resolve()
        dp = (root_dir / row["depth_path"]).resolve()
        bgr = cv2.imdecode(np.frombuffer(rp.read_bytes(), np.uint8), cv2.IMREAD_UNCHANGED)
        dep = cv2.imdecode(np.frombuffer(dp.read_bytes(), np.uint8), cv2.IMREAD_UNCHANGED)
        raw_rgb_u8.append(bgr[..., ::-1].copy())  # uint8 RGB: 792 frames = only 730 MB!
        raw_depths.append(dep)

    conditions = [("original", 0)]
    for s in (10, 25, 50):
        conditions.extend([("noisy", s), ("gaussian", s), ("cnn", s)])

    results_summary = []
    trajectories_saved = {}
    sample_pair_idx = min(15, len(rows) - 2)

    for method, sigma in conditions:
        t_start = time.perf_counter()
        proc_ms_list = []
        psnr_list = []
        ssim_list = []
        feats = []
        sample_pair_imgs = {}

        # Stream frame-by-frame: keep only (pts, desc) (~16 KB/frame) in memory!
        for i in range(len(rows)):
            ref_f32 = raw_rgb_u8[i].astype(np.float32) / np.float32(255.0)
            if sigma == 0:
                cur_f32 = ref_f32
            else:
                noisy_f32, _ = add_gaussian(ref_f32, rows[i]["pair_id"], sigma, base_seed)
                if method == "noisy":
                    cur_f32 = noisy_f32
                elif method == "gaussian":
                    t0 = time.perf_counter()
                    cur_f32 = fast_gaussian_filter(noisy_f32, filter_cfg["size"], filter_cfg["sigma_px"])
                    proc_ms_list.append((time.perf_counter() - t0) * 1000.0)
                elif method == "cnn":
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    cur_f32 = restore(model, noisy_f32, layout)
                    torch.cuda.synchronize()
                    proc_ms_list.append((time.perf_counter() - t0) * 1000.0)

                    if seq_key == "dev" and i == sample_pair_idx and sigma in (25, 50):
                        gau_vis = fast_gaussian_filter(noisy_f32, filter_cfg["size"], filter_cfg["sigma_px"])
                        panel = np.hstack([
                            rgb_to_uint8(x)[..., ::-1] for x in (ref_f32, noisy_f32, gau_vis, cur_f32)
                        ])
                        ok, buf = cv2.imencode(".png", panel)
                        if ok:
                            buf.tofile(str(out_dir / f"tum_4way_s{sigma}.png"))

                if i % 10 == 0:
                    m_im = fast_image_metrics(ref_f32, cur_f32)
                    psnr_list.append(m_im["psnr_db"])
                    ssim_list.append(m_im["ssim"])

            if seq_key == "dev" and i in (sample_pair_idx, sample_pair_idx + 1) and sigma in (0, 25, 50):
                sample_pair_imgs[i] = cur_f32.copy()

            feats.append(detect(cur_f32, orb_cfg))

        # Pairwise matching + batched SVD RANSAC + pose accumulation
        t_vo_start = time.perf_counter()
        poses = []
        diagnostics = []
        segment = 0
        world = np.eye(4)

        for i, row in enumerate(rows):
            origin = i == 0
            reason = None
            if i > 0:
                mutual = fast_screen_mutual(
                    feats[i - 1][0], feats[i - 1][1], feats[i][0], feats[i][1], ratio
                )
                if mutual:
                    a, va = backproject(
                        [x["source_uv"] for x in mutual],
                        raw_depths[i - 1],
                        cam["K"],
                        geom_cfg["depth_range_m"],
                    )
                    b, vb = backproject(
                        [x["target_uv"] for x in mutual],
                        raw_depths[i],
                        cam["K"],
                        geom_cfg["depth_range_m"],
                    )
                    valid_mask = va & vb
                    valid_indices = np.flatnonzero(valid_mask)
                    key_str = rows[i - 1]["pair_id"] + "->" + row["pair_id"]
                    res = fast_ransac(a[valid_mask], b[valid_mask], key_str, geom_cfg)
                    if i == 1 and seq_key == "dev" and method == "original":
                        ref_r = ref_ransac(a[valid_mask], b[valid_mask], key_str, geom_cfg)
                        assert np.allclose(res["T_target_source"], ref_r["T_target_source"], atol=1e-12)
                else:
                    valid_indices = np.array([], dtype=int)
                    res = {
                        "valid": False,
                        "reason": "no_matches",
                        "candidates": 0,
                        "inliers": 0,
                        "inlier_indices": [],
                    }

                res["matches"] = len(mutual)
                res["depth_valid_pairs"] = int(len(valid_indices))
                if res["valid"]:
                    world = world @ np.linalg.inv(res["T_target_source"])
                else:
                    segment += 1
                    world = np.eye(4)
                    origin = True
                    reason = res["reason"]
                diagnostics.append(res)

                if seq_key == "dev" and i == sample_pair_idx + 1 and sigma in (0, 25, 50):
                    inliers_in_mutual = (
                        [int(valid_indices[k]) for k in res.get("inlier_indices", [])]
                        if res["valid"]
                        else []
                    )
                    title = (
                        f"{method.upper()} (sigma={sigma}) | Mutual={len(mutual)} | "
                        f"3D RANSAC Inliers={res.get('inliers', 0)}"
                    )
                    draw_match_panel(
                        sample_pair_imgs[i - 1],
                        sample_pair_imgs[i],
                        mutual,
                        inliers_in_mutual,
                        title,
                        out_dir / f"match_{seq_key}_{method}_s{sigma}.png",
                    )

            poses.append(
                {
                    "pair_id": row["pair_id"],
                    "timestamp": row["rgb_timestamp"],
                    "segment_id": segment,
                    "segment_start": origin,
                    "valid": True,
                    "incoming_valid": None if i == 0 else diagnostics[-1]["valid"],
                    "failure_reason": reason,
                    "T_world_camera": world.tolist(),
                }
            )

        vo_ms_per_frame = ((time.perf_counter() - t_vo_start) * 1000.0) / len(rows)

        entries = []
        for pose in poses:
            pid = pose["pair_id"]
            if pid in assoc_map:
                e = dict(pose)
                e["estimate"] = pose["T_world_camera"]
                e["reference"] = gt_lookup[assoc_map[pid]["gt_timestamp"]].tolist()
                entries.append(e)

        try:
            t_metrics = eval_traj(entries, geom_cfg["rpe_dt_s"], geom_cfg["rpe_tolerance_s"])
            ate_m = t_metrics["ate_rmse_m"]
            rpe_t = t_metrics["rpe_translation_rmse_m"]
            rpe_r = t_metrics["rpe_rotation_rmse_deg"]
            main_assoc = t_metrics["associated"]
            trajectories_saved[f"{method}_s{sigma}"] = {
                "aligned_xyz": np.array(t_metrics["aligned"])[:, :3, 3].tolist(),
                "reference_xyz": np.array(t_metrics["reference"])[:, :3, 3].tolist(),
                "main_segment": t_metrics["main_segment"],
            }
        except Exception:
            ate_m, rpe_t, rpe_r, main_assoc = None, None, None, 0

        successes = sum(1 for d in diagnostics if d["valid"])
        attempts = len(diagnostics)
        mean_matches = float(np.mean([d["matches"] for d in diagnostics]))
        mean_inliers = float(np.mean([d.get("inliers", 0) for d in diagnostics]))
        mean_inlier_ratio = float(
            np.mean(
                [
                    (d.get("inliers", 0) / d["depth_valid_pairs"])
                    if d["depth_valid_pairs"] > 0
                    else 0.0
                    for d in diagnostics
                ]
            )
        )

        rec = {
            "sequence": seq_key,
            "sequence_name": cam["name"],
            "method": method,
            "sigma": sigma,
            "frames": len(rows),
            "transitions": attempts,
            "successful_transitions": successes,
            "failed_transitions": attempts - successes,
            "transition_success_rate": successes / attempts,
            "segments": segment + 1,
            "main_segment_associated": main_assoc,
            "main_segment_coverage": main_assoc / len(entries) if entries else 0.0,
            "seq_psnr_db": float(np.mean(psnr_list)) if psnr_list else None,
            "seq_ssim": float(np.mean(ssim_list)) if ssim_list else None,
            "mean_mutual_matches": mean_matches,
            "mean_ransac_inliers": mean_inliers,
            "mean_inlier_ratio_3d": mean_inlier_ratio,
            "ate_rmse_m": ate_m,
            "rpe_trans_rmse_m": rpe_t,
            "rpe_rot_rmse_deg": rpe_r,
            "preproc_ms_per_frame": float(np.mean(proc_ms_list)) if proc_ms_list else 0.0,
            "vo_ms_per_frame": vo_ms_per_frame,
            "wall_s": time.perf_counter() - t_start,
        }
        results_summary.append(rec)
        print(
            f"  [{seq_key}] {method:8s} s={sigma:2d} | Succ={successes}/{attempts} (seg={segment+1}) | "
            f"Matches={mean_matches:5.1f} Inliers={mean_inliers:5.1f} | "
            f"ATE={ate_m:.4f}m RPE_t={rpe_t:.4f}m ({rec['wall_s']:.1f}s)"
            if ate_m is not None
            else f"  [{seq_key}] {method:8s} s={sigma:2d} | FAILED",
            flush=True,
        )
        del feats, poses, diagnostics, sample_pair_imgs
        gc.collect()

    del raw_rgb_u8, raw_depths
    gc.collect()
    return results_summary, trajectories_saved


def main():
    out_dir = ROOT / "runs" / f"F08_joint_{uuid.uuid4().hex[:8]}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}", flush=True)

    frozen_info = load_json(ROOT / "runs/F07_freeze_6bcca18dff/frozen_model.json")
    best_ckpt = Path(frozen_info["checkpoints"]["best"]["path"])
    assert sha256(best_ckpt.read_bytes()) == frozen_info["checkpoints"]["best"]["sha256"]
    model, train_cfg = load_model(best_ckpt)
    model = model.to("cuda").eval()
    layout = train_cfg["layout"]

    roots_cfg = load_json(ROOT / "data/manifests/step02/roots_and_cameras.json")
    bsds_root = Path(roots_cfg["bsds_root"])
    front_cfg = load_json(ROOT / "configs/frontend_step04.json")
    geom_cfg = load_json(ROOT / "configs/geometry_step05.json")
    base_seed = front_cfg["base_seed"]
    filter_cfg = front_cfg["filter"]

    print("=== Part A: Independent BSDS300 Test Set (100 images) ===", flush=True)
    test_rows = [
        json.loads(line)
        for line in (ROOT / "data/manifests/step02/images_test.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    bsds_summary = []
    for sigma in (10, 25, 50):
        metrics_acc = {k: {"mse": [], "psnr": [], "ssim": []} for k in ("noisy", "gaussian", "cnn")}
        for idx, row in enumerate(test_rows):
            ref = load_reference(bsds_root, row)
            noisy, _ = add_gaussian(ref, row["sample_id"], sigma, base_seed)
            gau = fast_gaussian_filter(noisy, filter_cfg["size"], filter_cfg["sigma_px"])
            cnn = restore(model, noisy, layout)
            for k, est in (("noisy", noisy), ("gaussian", gau), ("cnn", cnn)):
                m = fast_image_metrics(ref, est)
                metrics_acc[k]["mse"].append(m["mse"])
                metrics_acc[k]["psnr"].append(m["psnr_db"])
                metrics_acc[k]["ssim"].append(m["ssim"])
            if idx == 0:
                panel = np.hstack([rgb_to_uint8(x)[..., ::-1] for x in (ref, noisy, gau, cnn)])
                ok, buf = cv2.imencode(".png", panel)
                if ok:
                    buf.tofile(str(out_dir / f"bsds_test0_4way_s{sigma}.png"))
        for k in ("noisy", "gaussian", "cnn"):
            rec = {
                "dataset": "BSDS300_test_100",
                "sigma": sigma,
                "method": k,
                "mean_mse": float(np.mean(metrics_acc[k]["mse"])),
                "mean_psnr_db": float(np.mean(metrics_acc[k]["psnr"])),
                "std_psnr_db": float(np.std(metrics_acc[k]["psnr"])),
                "mean_ssim": float(np.mean(metrics_acc[k]["ssim"])),
                "std_ssim": float(np.std(metrics_acc[k]["ssim"])),
            }
            bsds_summary.append(rec)
            print(
                f"  BSDS300 Test (N=100) | sigma={sigma:2d} | {k:8s} | "
                f"PSNR={rec['mean_psnr_db']:.3f}±{rec['std_psnr_db']:.2f} dB | SSIM={rec['mean_ssim']:.4f}",
                flush=True,
            )

    all_vo_results = []
    all_trajectories = {}
    for seq_key, pair_file, eval_file in (
        ("dev", "dev_pairs.jsonl", "dev_evaluation_index.jsonl"),
        ("test", "test_pairs.jsonl", "test_evaluation_index.jsonl"),
    ):
        print(f"=== Part B: TUM Sequence [{seq_key}] ===", flush=True)
        cam = roots_cfg["sequences"][seq_key]
        rows = [
            json.loads(s)
            for s in (ROOT / "data/manifests/step02" / pair_file).read_text(encoding="utf-8").splitlines()
            if s.strip()
        ]
        assoc_map = {
            r["pair_id"]: r
            for r in (
                json.loads(s)
                for s in (ROOT / "data/manifests/step02" / eval_file)
                .read_text(encoding="utf-8")
                .splitlines()
                if s.strip()
            )
        }
        gt_lines = (Path(cam["root"]) / "groundtruth.txt").read_text(encoding="utf-8").splitlines()
        gt_lookup = {}
        for line in gt_lines:
            if line.strip() and not line.startswith("#"):
                vals = line.split()
                gt_lookup[vals[0]] = quaternion_pose(list(map(float, vals[1:])))

        vo_res, trajs = run_sequence_vo_streaming(
            seq_key=seq_key,
            rows=rows,
            cam=cam,
            gt_lookup=gt_lookup,
            assoc_map=assoc_map,
            orb_cfg=front_cfg["orb"],
            ratio=front_cfg["ratio"],
            geom_cfg=geom_cfg,
            filter_cfg=filter_cfg,
            model=model,
            layout=layout,
            base_seed=base_seed,
            out_dir=out_dir,
        )
        if seq_key == "dev":
            orig_ate = vo_res[0]["ate_rmse_m"]
            assert abs(orig_ate - 0.08274702310687476) < 1e-9, f"F-05 regression mismatch: {orig_ate}"
            print(f"  [VERIFIED] Exact F-05 regression match: ATE={orig_ate:.12f} m", flush=True)

        all_vo_results.extend(vo_res)
        all_trajectories[seq_key] = trajs

    save_json(
        out_dir / "f08_summary.json",
        {
            "phase": "F-08",
            "frozen_checkpoint": str(best_ckpt),
            "checkpoint_step": 6000,
            "bsds_test_100": bsds_summary,
            "tum_vo_results": all_vo_results,
        },
    )
    save_json(out_dir / "f08_trajectories.json", all_trajectories)
    (ROOT / "runs/latest_f08_path.txt").write_text(str(out_dir), encoding="utf-8")
    print("ALL F-08 EXPERIMENTS COMPLETED SUCCESSFULLY!", flush=True)


if __name__ == "__main__":
    main()
