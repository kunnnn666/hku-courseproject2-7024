from pathlib import Path
import argparse, datetime, json, platform, sys, time
import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'rgbd_dataset_freiburg1_xyz'
CONFIG = {'dataset': 'freiburg1_xyz', 'width': 640, 'height': 480,
          'fx': 525.0, 'fy': 525.0, 'cx': 319.5, 'cy': 239.5,
          'depth_scale': 5000.0, 'depth_trunc_m': 4.0, 'max_rgb_depth_difference_s': 0.02,
          'method': 'Open3D hybrid RGB-D odometry', 'pair_count': 2,
          'transform': 'T_target_source: p_target = T_target_source @ p_source'}

def read_index(path):
    entries = []
    for line in path.read_text(encoding='utf-8').splitlines():
        if line.strip() and not line.lstrip().startswith('#'):
            fields = line.split()
            entries.append((float(fields[0]), fields[1]))
    if not entries or any(entries[i][0] >= entries[i+1][0] for i in range(len(entries)-1)):
        raise ValueError(f'Empty or non-increasing timestamps: {path}')
    return entries

def match_frames(rgb, depth, limit):
    # Greedy one-to-one global nearest timestamp matches, then restore RGB order.
    candidates = [(abs(r[0]-d[0]), i, j) for i, r in enumerate(rgb)
                  for j, d in enumerate(depth) if abs(r[0]-d[0]) <= limit]
    used_rgb, used_depth, matches = set(), set(), {}
    for delta, i, j in sorted(candidates):
        if i not in used_rgb and j not in used_depth:
            matches[i] = (rgb[i], depth[j], delta)
            used_rgb.add(i); used_depth.add(j)
    # Require three consecutive RGB index entries to avoid silently skipping frames.
    for i in range(len(rgb)-2):
        if all(j in matches for j in (i, i+1, i+2)):
            return [matches[j] for j in (i, i+1, i+2)], len(matches)
    raise ValueError('No three consecutive RGB frames have valid unique depth matches')

def load_frame(match):
    rgb, depth, delta = match
    color_path, depth_path = DATA / rgb[1], DATA / depth[1]
    if not color_path.is_file() or not depth_path.is_file():
        raise FileNotFoundError(f'Missing image: {color_path} or {depth_path}')
    color = o3d.io.read_image(str(color_path)); raw_depth = o3d.io.read_image(str(depth_path))
    c, d = np.asarray(color), np.asarray(raw_depth)
    if c.shape != (480, 640, 3) or d.shape != (480, 640) or d.dtype != np.uint16:
        raise ValueError(f'Unexpected images: color {c.shape} {c.dtype}; depth {d.shape} {d.dtype}')
    valid = d[d > 0]
    if valid.size == 0: raise ValueError('All depth pixels are missing')
    frame = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, raw_depth, depth_scale=CONFIG['depth_scale'], depth_trunc=CONFIG['depth_trunc_m'],
        convert_rgb_to_intensity=True)
    converted = np.asarray(frame.depth)
    mask = (d > 0) & (d.astype(np.float64) / 5000.0 < CONFIG['depth_trunc_m'])
    if not mask.any() or not np.allclose(converted[mask], d[mask] / 5000.0, atol=1e-6):
        raise ValueError('Open3D depth conversion failed')
    if not np.all(converted[d == 0] == 0): raise ValueError('Missing depth handling failed')
    info = {'rgb_timestamp': rgb[0], 'depth_timestamp': depth[0], 'rgb_file': rgb[1],
            'depth_file': depth[1], 'difference_ms': delta * 1000,
            'depth_raw_dtype': str(d.dtype), 'zero_fraction': float(np.mean(d == 0)),
            'valid_median_m': float(np.median(valid) / 5000), 'raw_min': int(valid.min()), 'raw_max': int(valid.max())}
    return frame, c.copy(), d.copy(), info

def preview(images, out):
    canvas = Image.new('RGB', (640*3, 520*2), 'white'); draw = ImageDraw.Draw(canvas)
    for i, (_, c, d, info) in enumerate(images):
        canvas.paste(Image.fromarray(c), (640*i, 30))
        # Fixed 0-4 m scale. Black means invalid; this visualization is not data.
        z = np.clip(d.astype(np.float32)/5000.0/4.0, 0, 1)
        vis = np.stack([z*255, (1-z)*255, np.full_like(z, 100)], axis=-1).astype(np.uint8)
        vis[d == 0] = 0
        canvas.paste(Image.fromarray(vis), (640*i, 550))
        draw.text((640*i+8, 6), f"frame {i}: RGB {info['rgb_timestamp']:.6f}", fill='black')
        draw.text((640*i+8, 526), 'depth preview: black = missing; green near, red far; clipped 4 m', fill='black')
    canvas.save(out)

def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', required=True, help='New output folder; never overwrites an existing folder')
    args = parser.parse_args(); out = Path(args.output).resolve(); out.mkdir(parents=True, exist_ok=False)
    (out/'config.json').write_text(json.dumps(CONFIG, indent=2), encoding='utf-8')
    rgb, depth = read_index(DATA/'rgb.txt'), read_index(DATA/'depth.txt')
    gt = [line.split() for line in (DATA/'groundtruth.txt').read_text(encoding='utf-8').splitlines() if line.strip() and not line.lstrip().startswith('#')]
    assert gt and all(len(row) == 8 for row in gt), 'Unexpected ground truth format'
    assert 5000 / CONFIG['depth_scale'] == 1.0 and 0 / CONFIG['depth_scale'] == 0
    matches, matched_count = match_frames(rgb, depth, CONFIG['max_rgb_depth_difference_s'])
    images = [load_frame(m) for m in matches]; preview(images, out/'input_preview.png')
    intrinsic = o3d.camera.PinholeCameraIntrinsic(640, 480, 525., 525., 319.5, 239.5)
    results = []
    for i in range(2):
        start = time.perf_counter()
        success, transform, information = o3d.pipelines.odometry.compute_rgbd_odometry(
            images[i][0], images[i+1][0], intrinsic, np.eye(4),
            o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),
            o3d.pipelines.odometry.OdometryOption())
        elapsed = time.perf_counter()-start
        valid = bool(np.isfinite(transform).all() and np.isfinite(information).all() and
                     np.allclose(transform[3], [0,0,0,1]) and
                     np.allclose(transform[:3,:3].T @ transform[:3,:3], np.eye(3), atol=1e-4) and
                     np.isclose(np.linalg.det(transform[:3,:3]), 1, atol=1e-4))
        results.append({'source_frame': i, 'target_frame': i+1, 'success': bool(success),
                        'valid_rigid_transform': valid, 'elapsed_s': elapsed,
                        'translation_norm_m': float(np.linalg.norm(transform[:3,3])),
                        'T_target_source': transform.tolist(), 'information_matrix': information.tolist()})
    report = {'created_local': datetime.datetime.now().astimezone().isoformat(),
              'python': sys.version, 'platform': platform.platform(), 'open3d': o3d.__version__, 'numpy': np.__version__,
              'rgb_entries': len(rgb), 'depth_entries': len(depth), 'matched_unique_entries': matched_count,
              'groundtruth_entries': len(gt), 'groundtruth_format': 'timestamp tx ty tz qx qy qz qw',
              'groundtruth_first': gt[0], 'frames': [x[3] for x in images], 'pairs': results}
    (out/'result.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(report, indent=2, allow_nan=False))
    if not all(x['success'] and x['valid_rigid_transform'] for x in results):
        raise SystemExit('FAIL: one or more pairs failed or were not rigid transforms')
    print('PASS: both adjacent RGB-D pairs returned finite rigid transforms')

if __name__ == '__main__': main()
