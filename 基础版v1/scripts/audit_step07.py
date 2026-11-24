"""Check completed run evidence independently of the training loop."""
import argparse
import json
import sys
from pathlib import Path
import uuid

import numpy as np
import torch

from train_step07 import ROOT, fingerprint, write_json, PatchData


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def audit(run):
    report = read_json(run / "result.json")
    config = read_json(run / "config.json")
    assert report["status"] == "completed" and report["exit_code"] == 0
    assert report["config_sha256"] == fingerprint(run / "config.json")
    for path, digest in {**report["code_hashes"], **report["input_hashes"]}.items():
        assert fingerprint(Path(path)) == digest, path
    steps = config["steps"]
    assert report["completed_steps"] == steps and report["parameter_max_change"] > 0
    with (run / "updates.jsonl").open(encoding="utf-8") as stream:
        updates = [json.loads(line) for line in stream]
    assert [row["step"] for row in updates] == list(range(1, steps + 1))
    assert all(np.isfinite(row["loss"]) and len(row["samples"]) == config["batch"] for row in updates)
    data = PatchData(ROOT, config)
    frozen_train = {row["sample_id"]: row for row in data.rows["train"]}
    seen = set()
    for update in updates:
        for sample in update["samples"]:
            row = frozen_train[sample["sample_id"]]
            assert sample["split"] == "train" and sample["file_sha"] == row["sha256"]
            assert sample["step"] == update["step"] and sample["group"] == row["group"]
            assert 0 <= sample["x"] <= row["width"] - config["patch"]
            assert 0 <= sample["y"] <= row["height"] - config["patch"]
            seen.add(sample["sample_id"])
    fixed = read_json(run / "fixed_validation.json")
    validation_ids = [row["sample_id"] for row in data.rows["val"]]
    assert [row["sample_id"] for row in fixed] == validation_ids and not seen.intersection(validation_ids)
    for index in range(26):
        data.batch("val", 0, [index])
    assert fixed == data.log
    validations = [json.loads(line) for line in (run / "validation.jsonl").read_text().splitlines()]
    interval = config["validation_interval"]
    expected_steps = sorted(set(range(interval, steps + 1, interval)) | {steps})
    assert [row["step"] for row in validations] == expected_steps
    for row in validations:
        assert [item["sample_id"] for item in row["images"]] == validation_ids
        for key in ("restored_psnr_db", "noisy_psnr_db", "residual_mse"):
            values = [item[key] for item in row["images"]]
            assert np.isfinite(values).all() and float(np.mean(values)) == row["mean_" + key]
    assert len({row["mean_noisy_psnr_db"] for row in validations}) == 1
    best = max(validations, key=lambda row: row["mean_restored_psnr_db"])
    assert best["step"] == report["best_step"] and best["mean_restored_psnr_db"] == report["best_psnr_db"]
    assert {p.name for p in run.glob("step_*.pt")} == {f"step_{step}.pt" for step in expected_steps}
    roles = {"best": best["step"], "last": steps, **{f"step_{step}": step for step in expected_steps}}
    for role, step in roles.items():
        state = torch.load(run / f"{role}.pt", weights_only=True, map_location="cpu")
        assert state["phase"] == "F-07" and state["training_kind"] == config["stage"]
        assert state["role"] == role and state["step"] == step
        assert state["short_test_only"] == (config["stage"] == "teaching")
        assert state["config"] == config and state["config_sha256"] == report["config_sha256"]
        assert state["input_hashes"] == report["input_hashes"] and state["code_hashes"] == report["code_hashes"]
        if role in ("best", "last"):
            at_step = torch.load(run / f"step_{step}.pt", weights_only=True, map_location="cpu")["model"]
            assert state["model"].keys() == at_step.keys()
            assert all(torch.isfinite(value).all() and torch.equal(value, at_step[key])
                       for key, value in state["model"].items())
    return dict(steps=steps, train_unique_images=len(seen), train_patch_count=steps * config["batch"],
                validation_steps=expected_steps, validation_images=26, best=best["step"], last=steps,
                mode=config["stage"], exit_code=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    output = ROOT / "runs" / f"F07_audit_{uuid.uuid4().hex[:10]}"
    output.mkdir()
    result = dict(command=[sys.executable, *sys.argv], source=str(args.run.resolve()), run_dir=str(output))
    try:
        result.update(audit(args.run))
    except Exception:
        import traceback
        result.update(exit_code=1, error=traceback.format_exc())
    write_json(output / "result.json", result)
    print(result)
    sys.exit(result["exit_code"])
