"""Official Hypersim metric implementation, isolated from the relative model."""
import hashlib
import inspect
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
METRIC_CODE = ROOT / 'vendor/Depth-Anything-V2/metric_depth'
WEIGHTS = ROOT / 'models/depth_anything_v2_metric/depth_anything_v2_metric_hypersim_vits.pth'
EXPECTED_SHA = 'b782898d8a3e8be1f639de33837ed85e9b4b73e40f8f5e5cd99067588d722545'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_model():
    if any(k == 'depth_anything_v2' or k.startswith('depth_anything_v2.') for k in sys.modules):
        raise RuntimeError('depth_anything_v2 already imported; use a fresh metric-only process')
    sys.path.insert(0, str(METRIC_CODE))
    from depth_anything_v2.dpt import DepthAnythingV2
    implementation = Path(inspect.getfile(DepthAnythingV2)).resolve()
    if not implementation.is_relative_to(METRIC_CODE.resolve()):
        raise RuntimeError('Metric module import resolved to the wrong implementation')
    digest = sha(WEIGHTS)
    if digest != EXPECTED_SHA:
        raise ValueError('Metric weight hash differs from pinned official LFS digest')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DepthAnythingV2(encoder='vits', features=64,
                            out_channels=[48, 96, 192, 384], max_depth=20)
    model.load_state_dict(torch.load(WEIGHTS, map_location='cpu', weights_only=True), strict=True)
    model.to(device).eval()
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    return model, {'model': 'Depth Anything V2 Metric Hypersim Small', 'unit': 'm (model prediction)',
                   'max_depth_parameter_m': 20, 'weights': str(WEIGHTS), 'weights_sha256': digest,
                   'implementation': str(implementation), 'implementation_sha256': sha(implementation),
                   'code_commit': 'a561b849ebae10a6f5ef49e26c83cbbcd36c71bf',
                   'model_revision': '3bc65d4e14a6786a61acec16453c50e12bf5f338',
                   'license': 'Apache-2.0 (Small model card)', 'device': device,
                   'gpu': torch.cuda.get_device_name(0) if device == 'cuda' else None,
                   'torch': torch.__version__, 'numpy': np.__version__, 'python': sys.version}


def predict(model, bgr):
    depth = model.infer_image(bgr, 518)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if depth.shape != bgr.shape[:2] or depth.dtype != np.float32 or not np.isfinite(depth).all():
        raise ValueError('Metric output shape, dtype or finite-value check failed')
    if not (depth > 0).any():
        raise ValueError('Metric output contains no positive depth')
    return depth
