"""Step 8 Upgraded VO Evaluation:
1) Sensor RGB-D VO (792 frames on freiburg1_xyz): HybridTerm vs ColorTerm across
   Clean, Low-Light, Low-Light+CLAHE, Motion Blur, and Blur+Sharpen.
2) Monocular Metric Depth VO (150 frames on freiburg1_xyz via Depth Anything V2 Metric Small):
   Clean, Low-Light, Low-Light+CLAHE, Motion Blur, and Blur+Sharpen — evaluating both
   metric depth accuracy (MAE, RMSE, AbsRel, delta<1.25) and downstream VO (ATE, RPE).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "rgbd_dataset_freiburg1_xyz"


def make_degradations_uint8(rgb_u8: np.ndarray, mode: str, frame_idx: int) -> np.ndarray:
    """Deterministic image degradation and enhancement using NumPy/PIL/OpenCV-compatible math."""
    if mode == "clean":
        return rgb_u8
    x = rgb_u8.astype(np.float32) / 255.0

    if mode in ("low_light", "low_light_enhanced"):
        # Severe indoor underexposure: gamma=2.2 compression + 0.22 exposure scale + deterministic sensor read noise
        rng = np.random.default_rng(7024000 + frame_idx)
        dark = 0.22 * np.power(np.clip(x, 0.0, 1.0), 2.2) + rng.normal(0.0, 6.0 / 255.0, size=x.shape).astype(np.float32)
        dark_u8 = np.clip(np.floor(dark * 255.0 + 0.5), 0, 255).astype(np.uint8)
        if mode == "low_light":
            return dark_u8
        # Enhancement: Gamma recovery (gamma=0.45) + luminance contrast stretch / local adaptive enhancement
        rec = np.power(dark_u8.astype(np.float32) / 255.0, 0.45)
        # Apply mild spatial luminance local contrast enhancement (unsharp luminance + percentile stretch)
        lum = 0.299 * rec[..., 0] + 0.587 * rec[..., 1] + 0.114 * rec[..., 2]
        p2, p98 = np.percentile(lum, 2.0), np.percentile(lum, 98.0)
        scale = np.clip((lum - p2) / max(p98 - p2, 1e-3), 0.0, 1.0) / np.maximum(lum, 1e-3)
        enh = np.clip(rec * scale[..., None], 0.0, 1.0)
        return np.clip(np.floor(enh * 255.0 + 0.5), 0, 255).astype(np.uint8)

    if mode in ("motion_blur", "blur_sharpened"):
        # 15-pixel diagonal motion blur simulating rapid UAV pitch/yaw jitter
        k_size = 15
        acc = np.zeros_like(x, dtype=np.float32)
        padded = np.pad(x, ((k_size // 2, k_size // 2), (k_size // 2, k_size // 2), (0, 0)), mode="edge")
        for s in range(k_size):
            # Slight diagonal motion trajectory (dx=s, dy=s//2)
            dy = (s // 2) + (k_size // 4)
            dx = s
            acc += padded[dy : dy + x.shape[0], dx : dx + x.shape[1], :]
        blurred = acc / float(k_size)
        blur_u8 = np.clip(np.floor(blurred * 255.0 + 0.5), 0, 255).astype(np.uint8)
        if mode == "motion_blur":
            return blur_u8
        # Unsharp masking restoration on blurred image
        b_f = blur_u8.astype(np.float32) / 255.0
        # 5x5 box/separable smooth for unsharp mask
        pad2 = np.pad(b_f, ((2, 2), (2, 2), (0, 0)), mode="reflect")
        w1d = np.array([1, 4, 6, 4, 1], dtype=np.float32) / 16.0
        tmp = sum(w1d[k] * pad2[:, k : k + b_f.shape[1], :] for k in range(5))
        smooth = sum(w1d[k] * tmp[k : k + b_f.shape[0], :, :] for k in range(5))
        sharp = np.clip(b_f + 1.6 * (b_f - smooth), 0.0, 1.0)
        return np.clip(np.floor(sharp * 255.0 + 0.5), 0, 255).astype(np.uint8)

    raise ValueError(f"Unknown mode: {mode}")


def stage_metric_predictions(out_dir: Path, max_frames: int = 150):
    """Run in .venv_step5 (PyTorch + CUDA): generate degraded RGBs and Depth Anything V2 Metric predictions."""
    import cv2
    import torch
    from metric_runtime import load_model, predict

    out_dir.mkdir(parents=True, exist_ok=True)
    # Read paired list from step234_full_retry/matching.json so frame pairs match sensor VO
    matching = json.loads((ROOT / "results/step234_full_retry/matching.json").read_text(encoding="utf-8"))
    rgb_lines = [
        line.split()
        for line in (DATA / "rgb.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    dep_lines = [
        line.split()
        for line in (DATA / "depth.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    pairs = matching["pairs"][:max_frames]

    model, provenance = load_model()
    modes = ["clean", "low_light", "low_light_enhanced", "motion_blur", "blur_sharpened"]
    depth_eval_summary = []

    # Save 5-panel visual comparison for frame 25 (RGB + TURBO Metric Depth)
    sample_idx = 25
    sample_rgb_panels = []
    sample_dep_panels = []

    for mode in modes:
        mode_dir = out_dir / f"pred_{mode}"
        (mode_dir / "depth_m").mkdir(parents=True, exist_ok=True)
        (mode_dir / "rgb").mkdir(parents=True, exist_ok=True)

        mae_list, rmse_list, absrel_list, d1_list, infer_ms = [], [], [], [], []
        frames_meta = []

        for n, p in enumerate(pairs):
            ri, di = p["rgb_index"], p["depth_index"]
            t_rgb = float(rgb_lines[ri][0])
            rgb_u8 = np.array(Image.open(DATA / rgb_lines[ri][1]).convert("RGB"))
            dep_u16 = np.array(Image.open(DATA / dep_lines[di][1]))
            proc_rgb = make_degradations_uint8(rgb_u8, mode, n)
            bgr = proc_rgb[..., ::-1].copy()

            t0 = time.perf_counter()
            pred_m = predict(model, bgr)
            infer_ms.append((time.perf_counter() - t0) * 1000.0)

            # Compare against valid sensor depth (0 < z_gt < 4.0 m)
            z_gt = dep_u16.astype(np.float32) / 5000.0
            mask = (dep_u16 > 0) & (z_gt > 0.2) & (z_gt < 4.0) & np.isfinite(pred_m) & (pred_m > 0)
            if mask.any():
                diff = pred_m[mask] - z_gt[mask]
                mae_list.append(float(np.mean(np.abs(diff))))
                rmse_list.append(float(np.sqrt(np.mean(diff ** 2))))
                absrel_list.append(float(np.mean(np.abs(diff) / z_gt[mask])))
                ratio = np.maximum(pred_m[mask] / z_gt[mask], z_gt[mask] / pred_m[mask])
                d1_list.append(float(np.mean(ratio < 1.25)))

            rgb_rel = f"rgb/{n:04d}.png"
            dep_rel = f"depth_m/{n:04d}.npy"
            Image.fromarray(proc_rgb).save(mode_dir / rgb_rel)
            np.save(mode_dir / dep_rel, pred_m)
            frames_meta.append({"index": n, "timestamp_s": t_rgb, "rgb": rgb_rel, "depth_m": dep_rel})

            if n == sample_idx:
                sample_rgb_panels.append(proc_rgb)
                norm = np.clip(pred_m / 4.0 * 255.0, 0, 255).astype(np.uint8)
                turbo_bgr = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
                sample_dep_panels.append(turbo_bgr[..., ::-1])

        (mode_dir / "frames.json").write_text(json.dumps(frames_meta, indent=2), encoding="utf-8")
        rec = {
            "mode": mode,
            "frames": len(pairs),
            "depth_mae_m": float(np.mean(mae_list)),
            "depth_rmse_m": float(np.sqrt(np.mean(np.square(rmse_list)))),
            "depth_abs_rel": float(np.mean(absrel_list)),
            "depth_delta1": float(np.mean(d1_list)),
            "mean_infer_ms": float(np.mean(infer_ms)),
        }
        depth_eval_summary.append(rec)
        print(
            f"[MetricDepth] {mode:18s} | MAE={rec['depth_mae_m']:.4f}m RMSE={rec['depth_rmse_m']:.4f}m "
            f"AbsRel={rec['depth_abs_rel']:.4f} d1={rec['depth_delta1']*100:.1f}% | {rec['mean_infer_ms']:.1f}ms/f",
            flush=True,
        )

    # Save composite 2x5 RGB + Predicted Depth grid
    top = np.hstack(sample_rgb_panels)
    bot = np.hstack(sample_dep_panels)
    grid = np.vstack([top, bot])
    Image.fromarray(grid).save(out_dir / "degradation_5way_rgb_depth.png")
    (out_dir / "metric_depth_summary.json").write_text(
        json.dumps({"provenance": provenance, "results": depth_eval_summary}, indent=2),
        encoding="utf-8",
    )


def stage_open3d_vo(out_dir: Path, sensor_frames: int = 792, pred_frames: int = 150):
    """Run in .venv (Open3D 0.19.0): run Sensor RGB-D VO (Hybrid vs Color) + Predicted Metric Depth VO."""
    import open3d as o3d
    from continuous_vo import evaluate as eval_poses, ground, matching, read_index, rigid

    modes = ["clean", "low_light", "low_light_enhanced", "motion_blur", "blur_sharpened"]
    rgb_idx = read_index(DATA / "rgb.txt")
    dep_idx = read_index(DATA / "depth.txt")
    pairs, _, _ = matching(rgb_idx, dep_idx)
    pairs = pairs[:sensor_frames]
    gt = ground()

    # Preload raw sensor RGB and uint16 depth
    print(f"[Open3D] Preloading {len(pairs)} sensor RGB-D frames...", flush=True)
    raw_rgbs = [np.array(Image.open(DATA / rgb_idx[i][1]).convert("RGB")) for i, j, _ in pairs]
    raw_deps = [o3d.io.read_image(str(DATA / dep_idx[j][1])) for i, j, _ in pairs]
    timestamps = [rgb_idx[i][0] for i, j, _ in pairs]

    intr = o3d.camera.PinholeCameraIntrinsic(640, 480, 525.0, 525.0, 319.5, 239.5)
    option = o3d.pipelines.odometry.OdometryOption()

    sensor_vo_results = []
    trajectories = {"sensor": {}, "predicted": {}}

    # 1. Sensor Depth Open3D VO: HybridTerm vs ColorTerm across all 5 modes
    for jac_name, jac_obj in (
        ("hybrid", o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm()),
        ("color", o3d.pipelines.odometry.RGBDOdometryJacobianFromColorTerm()),
    ):
        for mode in modes:
            t0 = time.perf_counter()
            poses = []
            prev_rgbd = None
            success_cnt = 0
            fail_cnt = 0
            T_world = np.eye(4)
            odom_ms = []

            for n in range(len(pairs)):
                proc_rgb = make_degradations_uint8(raw_rgbs[n], mode, n)
                c_img = o3d.geometry.Image(np.ascontiguousarray(proc_rgb))
                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    c_img, raw_deps[n], depth_scale=5000.0, depth_trunc=4.0, convert_rgb_to_intensity=True
                )
                if prev_rgbd is not None:
                    ts = time.perf_counter()
                    ok, trans, info = o3d.pipelines.odometry.compute_rgbd_odometry(
                        prev_rgbd, rgbd, intr, np.eye(4), jac_obj, option
                    )
                    odom_ms.append((time.perf_counter() - ts) * 1000.0)
                    ok = bool(ok and rigid(trans) and np.isfinite(info).all())
                    if ok:
                        T_world = T_world @ np.linalg.inv(trans)
                        success_cnt += 1
                    else:
                        fail_cnt += 1
                        # Carry pose hold ONLY to record failure count or segment? Let's stop or record segment!
                        # Keep consistent with continuous_vo: if failed, do not corrupt T_world, skip pose append
                        prev_rgbd = rgbd
                        continue
                poses.append({"t": timestamps[n], "T": T_world.copy()})
                prev_rgbd = rgbd

            metrics, assoc = eval_poses(poses, gt)
            rec = {
                "depth_source": "sensor_kinect",
                "jacobian": jac_name,
                "mode": mode,
                "frames": len(pairs),
                "transitions": len(pairs) - 1,
                "successful_transitions": success_cnt,
                "failed_transitions": fail_cnt,
                "success_rate": success_cnt / (len(pairs) - 1),
                "associated_poses": metrics.get("associated", 0),
                "ate_rmse_m": metrics.get("ate_rmse_m"),
                "rpe_trans_rmse_m": metrics.get("rpe_translation_rmse_m"),
                "mean_odom_ms": float(np.mean(odom_ms)) if odom_ms else 0.0,
                "wall_s": time.perf_counter() - t0,
            }
            sensor_vo_results.append(rec)
            trajectories["sensor"][f"{jac_name}_{mode}"] = {
                "estimated_xyz": [r["estimated"] for r in assoc],
                "reference_xyz": [r["reference"] for r in assoc],
                "error_m": [r["error_m"] for r in assoc],
                "timestamps": [r["timestamp"] for r in assoc],
            }
            print(
                f"[SensorVO-{jac_name:6s}] {mode:18s} | Succ={success_cnt}/{len(pairs)-1} | "
                f"ATE={rec['ate_rmse_m']:.4f}m RPE={rec['rpe_trans_rmse_m']:.4f}m | {rec['mean_odom_ms']:.1f}ms/f",
                flush=True,
            )

    # 2. Monocular Predicted Metric Depth Open3D VO across all 5 modes (150 frames)
    pred_vo_results = []
    for mode in modes:
        mode_dir = out_dir / f"pred_{mode}"
        frames_meta = json.loads((mode_dir / "frames.json").read_text(encoding="utf-8"))[:pred_frames]
        poses = []
        prev_rgbd = None
        success_cnt = 0
        fail_cnt = 0
        T_world = np.eye(4)
        odom_ms = []

        for row in frames_meta:
            rgb_arr = np.array(Image.open(mode_dir / row["rgb"]).convert("RGB"))
            dep_arr = np.load(mode_dir / row["depth_m"], allow_pickle=False)
            clean_dep = dep_arr.copy()
            clean_dep[~np.isfinite(clean_dep) | (clean_dep <= 0)] = 0.0
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(np.ascontiguousarray(rgb_arr)),
                o3d.geometry.Image(clean_dep),
                depth_scale=1.0,
                depth_trunc=4.0,
                convert_rgb_to_intensity=True,
            )
            if prev_rgbd is not None:
                ts = time.perf_counter()
                ok, trans, info = o3d.pipelines.odometry.compute_rgbd_odometry(
                    prev_rgbd,
                    rgbd,
                    intr,
                    np.eye(4),
                    o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),
                    option,
                )
                odom_ms.append((time.perf_counter() - ts) * 1000.0)
                ok = bool(ok and rigid(trans) and np.isfinite(info).all())
                if ok:
                    T_world = T_world @ np.linalg.inv(trans)
                    success_cnt += 1
                else:
                    fail_cnt += 1
                    prev_rgbd = rgbd
                    continue
            poses.append({"t": row["timestamp_s"], "T": T_world.copy()})
            prev_rgbd = rgbd

        metrics, assoc = eval_poses(poses, gt)
        rec = {
            "depth_source": "depth_anything_v2_metric_small",
            "jacobian": "hybrid",
            "mode": mode,
            "frames": len(frames_meta),
            "transitions": len(frames_meta) - 1,
            "successful_transitions": success_cnt,
            "failed_transitions": fail_cnt,
            "success_rate": success_cnt / (len(frames_meta) - 1),
            "associated_poses": metrics.get("associated", 0),
            "ate_rmse_m": metrics.get("ate_rmse_m"),
            "rpe_trans_rmse_m": metrics.get("rpe_translation_rmse_m"),
            "mean_odom_ms": float(np.mean(odom_ms)) if odom_ms else 0.0,
        }
        pred_vo_results.append(rec)
        trajectories["predicted"][mode] = {
            "estimated_xyz": [r["estimated"] for r in assoc],
            "reference_xyz": [r["reference"] for r in assoc],
            "error_m": [r["error_m"] for r in assoc],
            "timestamps": [r["timestamp"] for r in assoc],
        }
        print(
            f"[PredMetricVO]    {mode:18s} | Succ={success_cnt}/{len(frames_meta)-1} | "
            f"ATE={rec['ate_rmse_m']:.4f}m RPE={rec['rpe_trans_rmse_m']:.4f}m | {rec['mean_odom_ms']:.1f}ms/f",
            flush=True,
        )

    # Also evaluate Sensor Hybrid VO on the exact same 150-frame subset for fair apples-to-apples comparison!
    poses_150 = []
    prev_rgbd = None
    T_world = np.eye(4)
    for n in range(pred_frames):
        c_img = o3d.geometry.Image(np.ascontiguousarray(raw_rgbs[n]))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            c_img, raw_deps[n], depth_scale=5000.0, depth_trunc=4.0, convert_rgb_to_intensity=True
        )
        if prev_rgbd is not None:
            ok, trans, _ = o3d.pipelines.odometry.compute_rgbd_odometry(
                prev_rgbd, rgbd, intr, np.eye(4), o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(), option
            )
            if ok:
                T_world = T_world @ np.linalg.inv(trans)
        poses_150.append({"t": timestamps[n], "T": T_world.copy()})
        prev_rgbd = rgbd
    m150, assoc150 = eval_poses(poses_150, gt)
    trajectories["sensor_150_clean"] = {
        "estimated_xyz": [r["estimated"] for r in assoc150],
        "reference_xyz": [r["reference"] for r in assoc150],
        "ate_rmse_m": m150.get("ate_rmse_m"),
        "rpe_trans_rmse_m": m150.get("rpe_translation_rmse_m"),
    }

    (out_dir / "step8_vo_summary.json").write_text(
        json.dumps(
            {
                "sensor_vo_792": sensor_vo_results,
                "predicted_metric_vo_150": pred_vo_results,
                "sensor_clean_150_ate_m": m150.get("ate_rmse_m"),
                "sensor_clean_150_rpe_m": m150.get("rpe_translation_rmse_m"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (out_dir / "step8_trajectories.json").write_text(json.dumps(trajectories), encoding="utf-8")
    print("STEP 8 OPEN3D VO EVALUATION COMPLETED!", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["predict", "vo"], required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--pred-frames", type=int, default=150)
    ap.add_argument("--sensor-frames", type=int, default=792)
    args = ap.parse_args()
    if args.stage == "predict":
        stage_metric_predictions(args.output.resolve(), args.pred_frames)
    else:
        stage_open3d_vo(args.output.resolve(), args.sensor_frames, args.pred_frames)


if __name__ == "__main__":
    main()
