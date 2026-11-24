"""Explicit F-07 training modes. F-06 remains an unchanged dependency."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time
import traceback
import uuid

import cv2
import numpy as np
import torch

from train_step06 import ROOT, PatchData, model_for, prepare, sha256


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def fingerprint(path):
    return sha256(path.read_bytes())


def validation(model, samples):
    """Evaluate one fixed crop per image, with per-image scores."""
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            scores = []
            for sample, reference, noisy in samples:
                prediction = model(noisy)
                restored = (noisy - prediction).clamp(0, 1)
                restored_mse = (restored - reference).square().mean().item()
                noisy_mse = (noisy - reference).square().mean().item()
                residual_mse = (prediction - (noisy - reference)).square().mean().item()
                scores.append(dict(sample_id=sample, restored_psnr_db=-10 * np.log10(restored_mse),
                                   noisy_psnr_db=-10 * np.log10(noisy_mse), residual_mse=residual_mse))
        return dict(images=scores, **{
            "mean_" + key: float(np.mean([row[key] for row in scores]))
            for key in ("restored_psnr_db", "noisy_psnr_db", "residual_mse")
        })
    finally:
        model.train(was_training)


def save_checkpoint(run, role, model, metadata, probe=None):
    path = run / (role + ".pt")
    temporary = path.with_suffix(".pt.tmp")
    torch.save(dict(model=model.state_dict(), role=role, **metadata), temporary)
    temporary.replace(path)
    if probe is not None:
        array, tensor = probe
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                expected = model(tensor).cpu().numpy()
        finally:
            model.train(was_training)
        np.save(run / f"{role}_input.npy", array)
        np.save(run / f"{role}_expected.npy", expected)
        write_json(run / f"{role}_probe.json", dict(
            step=metadata["step"], checkpoint_sha256=fingerprint(path),
            input_sha256=fingerprint(run / f"{role}_input.npy"),
            expected_sha256=fingerprint(run / f"{role}_expected.npy"),
            dtype=str(tensor.dtype), shape=list(tensor.shape), stride=list(tensor.stride())))


def train(mode):
    config_name = "denoise_step07.json" if mode == "formal" else "denoise_step07_quick.json"
    source = ROOT / "configs" / config_name
    config = json.loads(source.read_text(encoding="utf-8"))
    if config["stage"] != mode or config["amp"] or config["max_seconds"] > 3600:
        raise ValueError("Configuration does not match the requested mode")
    if mode == "formal" and (config["steps"], config["validation_interval"], config["save_interval"]) != (10000, 500, 500):
        raise ValueError("Formal budget must be 10000 updates with intervals of 500")
    run = ROOT / "runs" / f"F07_{mode}_{uuid.uuid4().hex[:10]}"
    run.mkdir()
    start = time.perf_counter()
    report = dict(command=[sys.executable, *sys.argv], cwd=str(ROOT), run_dir=str(run),
                  started=datetime.now().astimezone().isoformat(), completed_steps=0,
                  status="running", config_sha256=fingerprint(source))
    (run / "config.json").write_bytes(source.read_bytes())
    write_json(run / "result.json", report)
    print(f"run_dir={run}", flush=True)
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required")
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed_all(config["seed"])
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
        data = PatchData(ROOT, config)
        model = model_for(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
        initial = [parameter.detach().cpu().clone() for parameter in model.parameters()]
        code = {str(path): fingerprint(path) for path in (
            Path(__file__), ROOT / "scripts/train_step06.py", ROOT / "src/denoiser.py",
            ROOT / "src/denoise_data.py", ROOT / "src/image_baseline.py")}
        report.update(device=torch.cuda.get_device_name(), torch=str(torch.__version__), cuda=torch.version.cuda,
                      numpy=np.__version__, opencv=cv2.__version__, cudnn=torch.backends.cudnn.version(),
                      input_hashes=data.hashes, code_hashes=code, amp=False, tf32=False,
                      initial_parameter_sha256=sha256(b"".join(p.numpy().tobytes() for p in initial)))
        samples = []
        for index, row in enumerate(data.rows["val"]):
            reference, noisy = data.batch("val", 0, [index])
            samples.append((row["sample_id"], prepare(reference, "cuda", config["layout"]),
                            prepare(noisy, "cuda", config["layout"])))
            if index == 0:
                probe = (noisy, samples[0][2])
        write_json(run / "fixed_validation.json", data.log)
        data.log.clear()
        torch.cuda.synchronize()
        report["resident_before_updates"] = dict(allocated=torch.cuda.memory_allocated(),
                                                 reserved=torch.cuda.memory_reserved())
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        best = -float("inf")
        report.update(update_s=0., gpu_event_s=0., validation_save_s=0., peak_update_allocated=0,
                      peak_update_reserved=0, warmup_updates=0)
        with (run / "updates.jsonl").open("w", encoding="utf-8", buffering=1) as updates, (
            run / "validation.jsonl").open("w", encoding="utf-8", buffering=1) as validations:
            for step in range(1, config["steps"] + 1):
                if time.perf_counter() - start >= config["max_seconds"]:
                    raise TimeoutError("F-07 wall-clock budget reached")
                torch.cuda.reset_peak_memory_stats()
                tick = time.perf_counter()
                reference, noisy = data.batch("train", step)
                reference, noisy = [prepare(x, "cuda", config["layout"]) for x in (reference, noisy)]
                target = noisy - reference
                torch.cuda.synchronize()
                begin.record()
                optimizer.zero_grad(set_to_none=True)
                loss = (model(noisy) - target).square().mean()
                loss.backward()
                if not torch.isfinite(loss).item() or not all(torch.isfinite(p.grad).all().item() for p in model.parameters()):
                    raise FloatingPointError(f"Nonfinite loss or gradient at step {step}")
                optimizer.step()
                end.record()
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - tick
                gpu_s = begin.elapsed_time(end) / 1000
                report["update_s"] += elapsed
                report["gpu_event_s"] += gpu_s
                report["completed_steps"] = step
                report["peak_update_allocated"] = max(report["peak_update_allocated"], torch.cuda.max_memory_allocated())
                report["peak_update_reserved"] = max(report["peak_update_reserved"], torch.cuda.max_memory_reserved())
                updates.write(json.dumps(dict(step=step, loss=loss.item(), seconds=elapsed,
                                              gpu_seconds=gpu_s, samples=data.log), allow_nan=False) + "\n")
                data.log.clear()
                if step % config["validation_interval"] == 0 or step == config["steps"]:
                    overhead = time.perf_counter()
                    score = validation(model, samples)
                    validations.write(json.dumps(dict(step=step, **score), allow_nan=False) + "\n")
                    metadata = dict(config=config, step=step, phase="F-07", training_kind=mode,
                                    short_test_only=mode == "teaching", resume_supported=False,
                                    torch_version=report["torch"], config_sha256=report["config_sha256"],
                                    input_hashes=data.hashes, code_hashes=code)
                    if score["mean_restored_psnr_db"] > best:
                        best = score["mean_restored_psnr_db"]
                        report["best_step"] = step
                        save_checkpoint(run, "best", model, metadata, probe)
                    if step % config["save_interval"] == 0 or step == config["steps"]:
                        save_checkpoint(run, f"step_{step}", model, metadata)
                    if step == config["steps"]:
                        save_checkpoint(run, "last", model, metadata, probe)
                    report["validation_save_s"] += time.perf_counter() - overhead
                    report["best_psnr_db"] = best
                    write_json(run / "result.json", report)
                    print(f"step={step} mean_patch_psnr={score['mean_restored_psnr_db']:.6f}", flush=True)
        change = max((p.detach().cpu() - old).abs().max().item() for p, old in zip(model.parameters(), initial))
        if not np.isfinite(change) or change <= 0:
            raise RuntimeError("No finite parameter update")
        if time.perf_counter() - start >= config["max_seconds"]:
            raise TimeoutError("F-07 wall-clock budget reached")
        report.update(status="completed", exit_code=0, parameter_max_change=change)
    except (Exception, KeyboardInterrupt):
        report.update(status="failed", exit_code=1, error=traceback.format_exc())
        print(report["error"], file=sys.stderr)
    report.update(ended=datetime.now().astimezone().isoformat(), seconds=time.perf_counter() - start)
    write_json(run / "result.json", report)
    print(f"status={report['status']} completed_steps={report['completed_steps']}", flush=True)
    return report["exit_code"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["teaching", "formal"], required=True)
    sys.exit(train(parser.parse_args().mode))
