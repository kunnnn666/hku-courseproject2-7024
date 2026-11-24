"""RGB HWC float32 processing shared by the F project."""
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def array_sha(image):
    return sha256(image.tobytes(order="C"))


def check_rgb(image):
    if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 1:
        raise ValueError(f"Expected a nonempty H×W×3 RGB image, got {image.shape}")
    if image.dtype != np.float32:
        raise ValueError(f"Expected float32 RGB, got {image.dtype}")
    if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
        raise ValueError("RGB values must be finite and in [0, 1]")


def load_reference(root, row):
    """Read one F-02 BSDS row and verify file and BGR pixel fingerprints."""
    root = Path(root).resolve()
    path = (root / row["relative_path"]).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Missing image or path outside data root: {path}")
    data = path.read_bytes()
    if sha256(data) != row["sha256"]:
        raise ValueError(f"Image SHA-256 mismatch: {path}")
    bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    expected = (row["height"], row["width"], 3)
    if bgr is None or bgr.shape != expected or bgr.dtype != np.uint8 or row["channels"] != 3:
        raise ValueError(f"Decode failed or expected uint8 shape {expected}: {path}")
    if array_sha(bgr) != row["pixel_sha256"]:
        raise ValueError(f"Decoded BGR fingerprint differs from F-02: {path}")
    rgb = bgr[..., ::-1].astype(np.float32) / np.float32(255)
    check_rgb(rgb)
    return rgb


def gaussian_noise(shape, sample_id, sigma_8bit, base_seed):
    """Return the unclipped, independent Gaussian samples and their seed."""
    if not math.isfinite(sigma_8bit) or sigma_8bit < 0:
        raise ValueError("Noise sigma must be finite and nonnegative")
    seed_data = json.dumps(
        ["F03-PCG64-v1", base_seed, sample_id, float(sigma_8bit)],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii")
    seed = int.from_bytes(hashlib.sha256(seed_data).digest()[:16], "big")
    rng = np.random.Generator(np.random.PCG64(seed))
    noise = rng.standard_normal(shape, dtype=np.float32)
    noise *= np.float32(sigma_8bit / 255)
    return noise, seed


def add_gaussian(reference, sample_id, sigma_8bit, base_seed):
    check_rgb(reference)
    noise, seed = gaussian_noise(reference.shape, sample_id, sigma_8bit, base_seed)
    before_clip = reference + noise
    clipped = (before_clip < 0) | (before_clip > 1)
    noisy = np.clip(before_clip, 0, 1)
    return noisy, {"seed_hex": f"{seed:032x}", "clipped_fraction": float(clipped.mean())}


def rgb_to_uint8(image):
    """Round to nearest byte, with exact half values rounded upward."""
    check_rgb(image)
    return np.floor(image.astype(np.float64) * 255 + 0.5).astype(np.uint8)


def gaussian_kernel(size, sigma_px):
    if type(size) is not int or size < 1 or size % 2 == 0:
        raise ValueError("Gaussian window size must be a positive odd integer")
    if not math.isfinite(sigma_px) or sigma_px <= 0:
        raise ValueError("Kernel sigma must be positive and finite")
    radius = size // 2
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    weights = np.exp(-0.5 * (offsets / sigma_px) ** 2)
    return weights / weights.sum()


def filter_axis(image, kernel, axis):
    radius = len(kernel) // 2
    padding = [(0, 0)] * image.ndim
    padding[axis] = (radius, radius)
    padded = np.pad(image, padding, mode="reflect")
    result = np.zeros(image.shape, dtype=np.float64)
    length = image.shape[axis]
    for offset, weight in enumerate(kernel):
        window = [slice(None)] * image.ndim
        window[axis] = slice(offset, offset + length)
        result += weight * padded[tuple(window)]
    return result


def separable(image, kernel):
    return filter_axis(filter_axis(image, kernel, 1), kernel, 0)


def gaussian_filter(image, size, sigma_px):
    check_rgb(image)
    values = separable(image.astype(np.float64), gaussian_kernel(size, sigma_px))
    return np.clip(values, 0, 1).astype(np.float32)


def image_metrics(reference, estimate):
    """MSE, PSNR and mean of per-channel valid-window SSIM."""
    check_rgb(reference)
    check_rgb(estimate)
    if reference.shape != estimate.shape or min(reference.shape[:2]) < 11:
        raise ValueError("Metrics require equal RGB shapes of at least 11×11")
    x = reference.astype(np.float64)
    y = estimate.astype(np.float64)
    mse = float(np.mean((x - y) ** 2))
    psnr = "+inf" if mse == 0 else -10 * math.log10(mse)
    kernel = gaussian_kernel(11, 1.5)
    valid = np.s_[5:-5, 5:-5, :]

    def local_average(image):
        return separable(image, kernel)[valid]

    mx = local_average(x)
    my = local_average(y)
    vx = np.maximum(local_average(x * x) - mx * mx, 0)
    vy = np.maximum(local_average(y * y) - my * my, 0)
    cov = local_average(x * y) - mx * my
    c1, c2 = 0.01**2, 0.03**2
    score = ((2 * mx * my + c1) * (2 * cov + c2)) / (
        (mx * mx + my * my + c1) * (vx + vy + c2)
    )
    return {"mse": mse, "psnr_db": psnr, "ssim": float(score.mean())}
