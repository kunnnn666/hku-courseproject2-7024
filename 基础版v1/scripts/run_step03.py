"""Run F-03 on the frozen validation split or on a single training example."""
import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import time
import traceback
import uuid

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import image_baseline as baseline


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def source_rows(config, mode):
    directory = ROOT / config["manifest_directory"]
    split = "val" if mode == "validation" else "train"
    manifest = directory / f"images_{split}.jsonl"
    paths = [manifest, directory / "roots_and_cameras.json"]
    frozen = json.loads((ROOT / "handoffs/step02/handoff.json").read_text(encoding="utf-8"))["manifests"]
    fingerprints = {}
    for path in paths:
        actual = baseline.sha256(path.read_bytes())
        if actual != frozen.get(str(path.resolve())):
            raise ValueError(f"Input differs from F-02 frozen manifest: {path}")
        fingerprints[str(path)] = actual
    root = Path(json.loads(paths[1].read_text(encoding="utf-8"))["bsds_root"])
    rows = read_rows(manifest)
    expected = 26 if split == "val" else 102
    if len(rows) != expected or len({row["sample_id"] for row in rows}) != expected:
        raise ValueError(f"Expected {expected} unique {split} samples")
    if any(row["split"] != split or row["data_root_key"] != "bsds_root" for row in rows):
        raise ValueError(f"Unexpected split or root in {manifest}")
    if mode == "demo":
        demo = ROOT / "data/manifests/F03_demo.jsonl"
        selection, = read_rows(demo)
        if selection["sample_id"] != config["demo"]["sample_id"] or selection["split"] != split:
            raise ValueError("Demo config and example list must select the same training sample")
        rows = [row for row in rows if row["sample_id"] == selection["sample_id"]]
        if len(rows) != 1:
            raise ValueError("Example ID is absent from the frozen training manifest")
        fingerprints[str(demo)] = baseline.sha256(demo.read_bytes())
    return root, rows, fingerprints


def describe(array):
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "color": "RGB",
        "min": float(array.min()),
        "max": float(array.max()),
        "array_sha256": baseline.array_sha(array),
    }


def preview(run, row, sigma, settings, images):
    names = ("reference", "noisy", "filtered")
    prefix = f"{row['sample_id']}_s{sigma}"
    artifacts = {}
    height, width = images[0].shape[:2]
    side = 72
    x, y = (width - side) // 2, (height - side) // 2
    panels = []
    for name, image in zip(names, images):
        path = run / f"{prefix}_{name}.npy"
        np.save(path, image, allow_pickle=False)
        artifacts[name] = {"path": path.name, "file_sha256": baseline.sha256(path.read_bytes()), **describe(image)}
        bgr = baseline.rgb_to_uint8(image)[..., ::-1].copy()
        crop = bgr[y:y + side, x:x + side]
        overview = bgr.copy()
        cv2.rectangle(overview, (x, y), (x + side - 1, y + side - 1), (0, 255, 255), 1)
        enlarged = cv2.resize(crop, (216, 216), interpolation=cv2.INTER_NEAREST)
        panel = np.full((height + 282, width, 3), 248, np.uint8)
        panel[52:52 + height] = overview
        panel[height + 60:height + 276, :216] = enlarged
        cv2.putText(panel, name, (9, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        label = f"noise {sigma}/255 | kernel {settings['size']}px / {settings['sigma_px']}px"
        cv2.putText(panel, label, (9, 41), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
        panels.append(panel)
    canvas = np.vstack([np.full((32, 3 * width, 3), 248, np.uint8), np.hstack(panels)])
    cv2.putText(canvas, f"{row['sample_id']} | same crop x={x}, y={y}, {side}x{side} | RGB 0..1", (9, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    ok, png = cv2.imencode(".png", canvas)
    if not ok:
        raise RuntimeError("Preview PNG encoding failed")
    preview_path = run / f"{prefix}_comparison.png"
    png.tofile(str(preview_path))
    return {"preview_png": preview_path.name, "preview_sha256": baseline.sha256(preview_path.read_bytes()),
            "crop_xywh": [x, y, side, side], "arrays": artifacts}


def evaluate(reference, row, sigma, config):
    noisy, rng = baseline.add_gaussian(reference, row["sample_id"], sigma, config["base_seed"])
    settings = config["filter"]
    filtered = baseline.gaussian_filter(noisy, settings["size"], settings["sigma_px"])
    record = {
        "sample_id": row["sample_id"], "split": row["split"], "group": row["group"],
        "sigma_8bit": sigma, "sigma_unit": sigma / 255, **rng,
        "reference": describe(reference), "noisy": describe(noisy), "filtered": describe(filtered),
        "filter_input_sha256": baseline.array_sha(noisy),
        "noisy_metrics": baseline.image_metrics(reference, noisy),
        "filtered_metrics": baseline.image_metrics(reference, filtered),
    }
    return record, noisy, filtered


def summarize(records, sigmas):
    summary = []
    for sigma in sorted(sigmas):
        rows = [row for row in records if row["sigma_8bit"] == sigma]
        item = {"sigma_8bit": sigma, "count": len(rows)}
        for method in ("noisy", "filtered"):
            scores = [row[f"{method}_metrics"] for row in rows]
            infinite = sum(score["psnr_db"] == "+inf" for score in scores)
            item[method] = {
                "mean_mse": float(np.mean([score["mse"] for score in scores])),
                "mean_psnr_db": "+inf" if infinite else float(np.mean([score["psnr_db"] for score in scores])),
                "psnr_infinite_count": infinite,
                "mean_ssim": float(np.mean([score["ssim"] for score in scores])),
            }
        item["mse_improved_count"] = sum(row["filtered_metrics"]["mse"] < row["noisy_metrics"]["mse"] for row in rows)
        item["mse_worse_count"] = sum(row["filtered_metrics"]["mse"] > row["noisy_metrics"]["mse"] for row in rows)
        summary.append(item)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/baseline_step03.json")
    parser.add_argument("--mode", choices=["validation", "demo"], default="validation")
    parser.add_argument("--sigma", type=int, choices=[0, 10, 25, 50], help="Demo override in 8-bit units")
    parser.add_argument("--reverse", action="store_true", help="Traverse samples and noise levels in reverse")
    args = parser.parse_args()
    if args.sigma is not None and args.mode != "demo":
        parser.error("--sigma is only for --mode demo")
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")
    run = ROOT / "runs" / f"F03_{args.mode}_{stamp}_{uuid.uuid4().hex[:6]}"
    run.mkdir(parents=True)
    started = time.perf_counter()
    report = {"command": [sys.executable, *sys.argv], "cwd": str(Path.cwd()), "mode": args.mode,
              "reverse": args.reverse, "run_dir": str(run), "versions": {"python": sys.version,
              "numpy": np.__version__, "opencv": cv2.__version__}}
    try:
        config = json.loads(args.config.read_text(encoding="utf-8-sig"))
        if config["filter"]["boundary"] != "reflect101":
            raise ValueError("This baseline supports only reflect101")
        if config["sigmas_8bit"] != [10, 25, 50]:
            raise ValueError("Validation requires frozen levels 10, 25, 50")
        report["config_sha256"] = baseline.sha256(args.config.read_bytes())
        report["code_sha256"] = {str(path): baseline.sha256(path.read_bytes())
                                 for path in [Path(__file__), Path(baseline.__file__)]}
        (run / "config.json").write_bytes(args.config.read_bytes())
        root, rows, report["input_manifest_sha256"] = source_rows(config, args.mode)
        sigmas = config["sigmas_8bit"] if args.mode == "validation" else [
            config["demo"]["sigma_8bit"] if args.sigma is None else args.sigma]
        report["sigmas_8bit"] = sigmas
        write_json(run / "inputs.json", {"data_root": str(root), "rows": rows})
        preview_id = rows[0]["sample_id"]
        records = []
        artifacts = []
        traversal = rows[::-1] if args.reverse else rows
        for row in traversal:
            reference = baseline.load_reference(root, row)
            for sigma in (sigmas[::-1] if args.reverse else sigmas):
                record, noisy, filtered = evaluate(reference, row, sigma, config)
                records.append(record)
                if row["sample_id"] == preview_id:
                    artifacts.append(preview(run, row, sigma, config["filter"], (reference, noisy, filtered)))
        records.sort(key=lambda row: (row["sample_id"], row["sigma_8bit"]))
        if args.mode == "validation":
            low_noise = [r for r in records if r["sigma_8bit"] == 10]
            weakest = max(low_noise, key=lambda r: r["filtered_metrics"]["mse"] - r["noisy_metrics"]["mse"])
            report["weakest_sigma10"] = weakest
            if weakest["sample_id"] != preview_id:
                row = next(row for row in rows if row["sample_id"] == weakest["sample_id"])
                reference = baseline.load_reference(root, row)
                repeated, noisy, filtered = evaluate(reference, row, 10, config)
                if repeated != weakest:
                    raise RuntimeError("Preview regeneration differs from recorded evaluation")
                artifacts.append(preview(run, row, 10, config["filter"], (reference, noisy, filtered)))
        write_json(run / "metrics.json", records)
        write_json(run / "examples.json", sorted(artifacts, key=lambda item: item["preview_png"]))
        write_json(run / "summary.json", summarize(records, sigmas))
        report.update(status="passed", exit_code=0, sample_count=len(rows), metric_count=len(records))
    except Exception:
        report.update(status="failed", exit_code=1, error=traceback.format_exc())
        (run / "failure.log").write_text(report["error"], encoding="utf-8")
    report["seconds"] = time.perf_counter() - started
    write_json(run / "result.json", report)
    write_json(run / "console.log", report)
    print(json.dumps(report, ensure_ascii=True, indent=2, allow_nan=False))
    return report["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
