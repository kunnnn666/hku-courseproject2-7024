"""Load local F-07 checkpoints and denoise RGB float32 images."""
import argparse
import sys
import traceback
import uuid
from pathlib import Path

import cv2
import numpy as np
import torch

from train_step07 import ROOT, PatchData, fingerprint, model_for, prepare, write_json
from src.image_baseline import add_gaussian, check_rgb, load_reference, rgb_to_uint8, sha256

ATOL = 1e-6
RTOL = 1e-5


def load_model(checkpoint):
    checkpoint = Path(checkpoint).resolve(strict=True)
    if not checkpoint.is_relative_to(ROOT / "runs"):
        raise ValueError("Select a checkpoint generated under this project's runs directory")
    state = torch.load(checkpoint, weights_only=True, map_location="cpu")
    if state["phase"] != "F-07":
        raise ValueError("Expected F-07 checkpoint")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    model = model_for(state["config"])
    model.load_state_dict(state["model"])
    return model.eval(), state["config"]


def restore(model, rgb, layout):
    check_rgb(rgb)
    tensor = prepare(rgb.transpose(2, 0, 1)[None], "cuda", layout)
    with torch.inference_mode():
        output = (tensor - model(tensor)).clamp(0, 1)
    result = np.ascontiguousarray(output[0].cpu().numpy().transpose(1, 2, 0))
    check_rgb(result)
    return result


def reload_probe(source, role, checkpoint):
    model, config = load_model(checkpoint)
    probe = source / f"{role}_probe.json"
    import json
    metadata = json.loads(probe.read_text(encoding="utf-8"))
    input_path = source / f"{role}_input.npy"
    expected_path = source / f"{role}_expected.npy"
    if (fingerprint(checkpoint) != metadata["checkpoint_sha256"]
            or fingerprint(input_path) != metadata["input_sha256"]
            or fingerprint(expected_path) != metadata["expected_sha256"]):
        raise ValueError("Checkpoint or saved probe fingerprint mismatch")
    tensor = prepare(np.load(input_path, allow_pickle=False), "cuda", config["layout"])
    with torch.inference_mode():
        actual = model(tensor).cpu().numpy()
    expected = np.load(expected_path, allow_pickle=False)
    shape_ok = actual.shape == expected.shape == tuple(metadata["shape"])
    finite = bool(np.isfinite(actual).all() and np.isfinite(expected).all())
    passed = shape_ok and finite and bool(np.allclose(actual, expected, atol=ATOL, rtol=RTOL))
    layout_ok = list(tensor.stride()) == metadata["stride"] and str(tensor.dtype) == metadata["dtype"]
    return dict(step=metadata["step"], checkpoint_sha256=fingerprint(checkpoint),
                input_sha256=fingerprint(input_path), atol=ATOL, rtol=RTOL,
                max_abs_error=float(np.max(abs(actual - expected))) if shape_ok and finite else None,
                allclose=passed, finite=finite, shape_ok=shape_ok, layout_ok=layout_ok,
                dtype=str(tensor.dtype), shape=list(tensor.shape), stride=list(tensor.stride()),
                exit_code=0 if passed and layout_ok else 1)


def demonstrate(checkpoint, output_dir):
    model, config = load_model(checkpoint)
    data = PatchData(ROOT, config)
    row = data.rows["val"][0]
    reference = load_reference(data.image_root, row)
    noisy, noise_meta = add_gaussian(reference, row["sample_id"], config["sigma"], config["seed"])
    restored = restore(model, noisy, config["layout"])
    for name, rgb in (("reference", reference), ("noisy", noisy), ("restored", restored)):
        np.save(output_dir / f"{name}.npy", rgb)
        ok, png = cv2.imencode(".png", rgb_to_uint8(rgb)[..., ::-1])
        if not ok:
            raise IOError(f"PNG encoding failed for {name}")
        png.tofile(str(output_dir / f"{name}.png"))
    return dict(exit_code=0, sample_id=row["sample_id"], file_sha256=row["sha256"],
                scope="one full validation image, illustration only", sigma=config["sigma"], seed=config["seed"],
                noise=noise_meta, shape=list(reference.shape), dtype=str(reference.dtype),
                input_sha256=sha256(noisy.tobytes()), checkpoint_sha256=fingerprint(checkpoint),
                noisy_psnr_db=float(-10*np.log10(np.mean((noisy-reference)**2))),
                restored_psnr_db=float(-10*np.log10(np.mean((restored-reference)**2))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--role", choices=["best", "last"], required=True)
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    run = ROOT / "runs" / f"F07_infer_{uuid.uuid4().hex[:10]}"
    run.mkdir()
    checkpoint = args.run / f"{args.role}.pt"
    report = dict(command=[sys.executable, *sys.argv], cwd=str(Path.cwd()), run_dir=str(run),
                  checkpoint=str(checkpoint.resolve()), role=args.role)
    try:
        report.update(demonstrate(checkpoint, run) if args.demo else reload_probe(args.run, args.role, checkpoint))
    except Exception:
        report.update(exit_code=1, error=traceback.format_exc())
    write_json(run / "result.json", report)
    print(report)
    return report["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
