"""Check numeric invariants and compare two full F-03 runs."""
import argparse
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
from statistics import mean
import sys
import time
import traceback
import uuid

import cv2
import numpy as np

from run_step03 import ROOT, baseline, read_rows, write_json


def require(condition, message):
    if not condition:
        raise ValueError(message)


def numeric_checks():
    report = {}
    flat = np.full((12, 13, 3), 0.25, np.float32)
    zero, _ = baseline.add_gaussian(flat, "constant", 0, 7024)
    require(np.array_equal(flat, zero), "Sigma zero must be identical")
    report["sigma_zero_max_abs_diff"] = float(np.max(np.abs(flat - zero)))
    report["noise_scale"] = []
    for sigma in (10, 25, 50):
        noise, _ = baseline.gaussian_noise((512, 512, 3), "synthetic", sigma, 7024)
        expected = sigma / 255
        observed_mean = float(noise.mean(dtype=np.float64))
        observed_std = float(noise.std(dtype=np.float64))
        require(abs(observed_mean) <= 0.005 * expected, "Unexpected noise mean")
        require(abs(observed_std - expected) <= 0.01 * expected, "Unexpected noise scale")
        report["noise_scale"].append({"sigma": sigma, "expected": expected, "mean": observed_mean,
                                      "std": observed_std, "mean_tol": 0.005 * expected,
                                      "std_tol": 0.01 * expected, "element_count": noise.size})
    edge = np.zeros((30, 40, 3), np.float32)
    edge[:, 20:] = 1
    noise, _ = baseline.gaussian_noise(edge.shape, "saturation", 50, 7024)
    noisy, metadata = baseline.add_gaussian(edge, "saturation", 50, 7024)
    raw = edge + noise
    require(raw.min() < 0 and raw.max() > 1, "Synthetic image must exercise clipping")
    require(np.array_equal(noisy, np.clip(raw, 0, 1)), "Wrong clipping")
    require(0 < metadata["clipped_fraction"] < 1, "Expected a partial clipping fraction")
    report["clipping"] = {"before_range": [float(raw.min()), float(raw.max())],
                          "after_range": [float(noisy.min()), float(noisy.max())], **metadata}
    weights = baseline.gaussian_kernel(7, 1)
    require(abs(weights.sum() - 1) <= 1e-15 and np.array_equal(weights, weights[::-1]),
            "Kernel normalization or symmetry failed")
    constant_out = baseline.gaussian_filter(flat, 7, 1)
    require(np.array_equal(flat, constant_out), "Constant image changed")
    impulse = np.zeros((3, 4, 3), np.float32)
    impulse[0, 0, :] = 1
    small = baseline.gaussian_kernel(3, 1)
    impulse_out = baseline.gaussian_filter(impulse, 3, 1)
    require(abs(float(impulse_out[0, 0, 0]) - small[1] ** 2) < 1e-7, "Corner reflection error")
    require(abs(float(impulse_out[0, 1, 0]) - small[0] * small[1]) < 1e-7, "Edge reflection error")
    opencv = cv2.sepFilter2D(noisy.astype(np.float64), cv2.CV_64F, weights, weights,
                             borderType=cv2.BORDER_REFLECT_101).astype(np.float32)
    own = baseline.gaussian_filter(noisy, 7, 1)
    max_diff = float(np.max(np.abs(own - opencv)))
    require(own.shape == noisy.shape and max_diff <= 1e-7, "Filter differs from OpenCV reference")
    report["filter"] = {"kernel": weights.tolist(), "sum": float(weights.sum()), "sum_atol": 1e-15,
                        "symmetric": True, "constant_max_abs_diff": float(np.max(np.abs(flat-constant_out))),
                        "output_shape": list(own.shape), "corner_expected": float(small[1] ** 2),
                        "corner_observed": float(impulse_out[0, 0, 0]), "impulse_atol": 1e-7,
                        "opencv_max_abs_diff": max_diff, "opencv_atol": 1e-7}
    identical = baseline.image_metrics(flat, flat)
    changed = baseline.image_metrics(flat, np.full_like(flat, 0.5))
    expected_ssim = (2 * 0.25 * 0.5 + 0.01**2) / (0.25**2 + 0.5**2 + 0.01**2)
    require(identical["mse"] == 0 and identical["psnr_db"] == "+inf" and abs(identical["ssim"]-1) < 1e-12,
            "Identical image metric is incorrect")
    require(changed["mse"] == 0.0625 and abs(changed["psnr_db"] - 10 * math.log10(16)) < 1e-12
            and abs(changed["ssim"] - expected_ssim) < 1e-12, "Known constant offset metric is incorrect")
    patch_x = noisy[:11, :11].astype(np.float64)
    patch_y = own[:11, :11].astype(np.float64)
    patch_score = baseline.image_metrics(patch_x.astype(np.float32), patch_y.astype(np.float32))["ssim"]
    coords = np.arange(-5, 6)
    square = np.exp(-(coords[:, None] ** 2 + coords[None, :] ** 2) / (2 * 1.5**2))
    square = (square / square.sum())[:, :, None]
    mx = (square * patch_x).sum(axis=(0, 1))
    my = (square * patch_y).sum(axis=(0, 1))
    vx = (square * (patch_x - mx) ** 2).sum(axis=(0, 1))
    vy = (square * (patch_y - my) ** 2).sum(axis=(0, 1))
    cov = (square * (patch_x - mx) * (patch_y - my)).sum(axis=(0, 1))
    direct = float(np.mean((2 * mx * my + .0001) * (2 * cov + .0009) /
                           ((mx**2 + my**2 + .0001) * (vx + vy + .0009))))
    require(abs(patch_score - direct) <= 1e-12, "Direct 2D SSIM window differs")
    report["metrics"] = {"identical": identical, "constant_offset": changed, "constant_expected_ssim": expected_ssim,
                         "patch_ssim": patch_score, "direct_2d_ssim": direct, "atol": 1e-12}
    quantized = baseline.rgb_to_uint8(np.array([[[0, 0.5, 1]]], np.float32))
    require(quantized.tolist() == [[[0, 128, 255]]], "Quantization test failed")
    report["quantized_0_half_1"] = quantized.tolist()
    return report


def compare(first, second):
    results = [json.loads((run / "result.json").read_text(encoding="utf-8")) for run in (first, second)]
    require(all(result["exit_code"] == 0 and result["mode"] == "validation" for result in results)
            and not results[0]["reverse"] and results[1]["reverse"], "Expected forward and reverse validation runs")
    for name in ("config_sha256", "code_sha256", "input_manifest_sha256", "versions"):
        require(results[0][name] == results[1][name], f"Runs differ in {name}")
    for name in ("metrics.json", "summary.json", "inputs.json", "examples.json"):
        require((first / name).read_bytes() == (second / name).read_bytes(), f"Runs differ in {name}")
    metrics = json.loads((first / "metrics.json").read_text(encoding="utf-8"))
    inputs = json.loads((first / "inputs.json").read_text(encoding="utf-8"))
    expected_rows = read_rows(ROOT / "data/manifests/step02/images_val.jsonl")
    require(inputs["rows"] == expected_rows and len(expected_rows) == 26, "Validation inputs differ from frozen split")
    expected_keys = {(row["sample_id"], sigma) for row in expected_rows for sigma in (10, 25, 50)}
    require(len(metrics) == 78 and {(row["sample_id"], row["sigma_8bit"]) for row in metrics} == expected_keys,
            "Incomplete validation coverage")
    require(all(row["noisy"]["array_sha256"] == row["filter_input_sha256"] for row in metrics), "Filter input hash differs")
    summary = json.loads((first / "summary.json").read_text(encoding="utf-8"))
    max_aggregate_diff = 0.0
    for item in summary:
        rows = [row for row in metrics if row["sigma_8bit"] == item["sigma_8bit"]]
        require(len(rows) == item["count"], "Aggregate count differs")
        for method in ("noisy", "filtered"):
            for key in ("mse", "psnr_db", "ssim"):
                computed = mean(row[f"{method}_metrics"][key] for row in rows)
                difference = abs(computed - item[method][f"mean_{key}"])
                max_aggregate_diff = max(max_aggregate_diff, difference)
    require(max_aggregate_diff <= 1e-12, "Aggregates differ from per-image metrics")
    examples = json.loads((first / "examples.json").read_text(encoding="utf-8"))
    saved = []
    for entry in examples:
        arrays = {}
        for name, artifact in entry["arrays"].items():
            a = np.load(first / artifact["path"], allow_pickle=False)
            b = np.load(second / artifact["path"], allow_pickle=False)
            require(np.array_equal(a, b), "Saved arrays differ between runs")
            require(baseline.array_sha(a) == artifact["array_sha256"], "Saved array hash mismatch")
            require(all(baseline.sha256((run / artifact["path"]).read_bytes()) == artifact["file_sha256"]
                        for run in (first, second)), "NPY file hash mismatch")
            arrays[name] = a
            saved.append(artifact["path"])
        prefix = entry["preview_png"].removesuffix("_comparison.png")
        sample_id, sigma_text = prefix.rsplit("_s", 1)
        row = next(r for r in metrics if (r["sample_id"], r["sigma_8bit"]) == (sample_id, int(sigma_text)))
        for method in ("noisy", "filtered"):
            require(baseline.image_metrics(arrays["reference"], arrays[method]) == row[f"{method}_metrics"],
                    "Metrics recomputed from saved arrays differ")
        config = json.loads((first / "config.json").read_text(encoding="utf-8"))
        changed, _ = baseline.add_gaussian(arrays["reference"], sample_id, int(sigma_text), config["base_seed"] + 1)
        require(not np.array_equal(arrays["noisy"], changed), "Different seed produced identical nonzero noise")
        require(all(baseline.sha256((run / entry["preview_png"]).read_bytes()) == entry["preview_sha256"]
                    for run in (first, second)), "Preview hash mismatch")
    return {"all_78_metric_and_array_hash_records_byte_identical": True, "saved_array_files": saved,
            "saved_array_max_abs_diff": 0, "different_seed_changes_saved_noisy_examples": True,
            "metrics_recomputed_from_npy_identical": True, "aggregate_max_abs_diff": max_aggregate_diff,
            "aggregate_atol": 1e-12, "validated_image_count": 26}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    args = parser.parse_args()
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")
    run = ROOT / "runs" / f"F03_verify_{stamp}_{uuid.uuid4().hex[:6]}"
    run.mkdir()
    start = time.perf_counter()
    result = {"command": [sys.executable, *sys.argv], "cwd": str(Path.cwd()), "run_dir": str(run),
              "numpy": np.__version__, "opencv": cv2.__version__}
    try:
        result["numeric"] = numeric_checks()
        result["repeat"] = compare(args.first, args.second)
        result.update(status="passed", exit_code=0)
    except Exception:
        result.update(status="failed", exit_code=1, error=traceback.format_exc())
    result["seconds"] = time.perf_counter() - start
    write_json(run / "result.json", result)
    write_json(run / "console.log", result)
    print(json.dumps(result, ensure_ascii=True, indent=2, allow_nan=False))
    return result["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
