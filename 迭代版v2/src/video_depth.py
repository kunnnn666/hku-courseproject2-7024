"""Stream a local video or TUM image sequence through the official metric model."""
import argparse
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

from metric_runtime import ROOT, load_model, predict, sha

DATA = ROOT / 'data/rgbd_dataset_freiburg1_xyz'
FFMPEG = ROOT / '.venv/Lib/site-packages/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe'


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def read_index(path):
    rows = [line.split() for line in path.read_text().splitlines() if line.strip() and not line.startswith('#')]
    return [(float(t), name) for t, name in rows]


def color_depth(depth):
    valid = np.isfinite(depth) & (depth > 0)
    norm = np.zeros(depth.shape, np.uint8)
    norm[valid] = np.clip(depth[valid] / 4 * 255, 0, 255).astype(np.uint8)
    color = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    color[~valid] = 0
    return color


def stream(args):
    if args.tum:
        rows = read_index(DATA / 'rgb.txt')
        for i, (timestamp, name) in enumerate(rows):
            if i % args.stride:
                continue
            path = DATA / name
            bgr = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError(f'Cannot decode {path}')
            yield i, timestamp, bgr, {'path': str(path), 'sha256': sha(path), 'timestamp_source': 'TUM rgb.txt'}
    else:
        cap = cv2.VideoCapture(str(args.video.resolve()))
        if not cap.isOpened():
            raise ValueError('Cannot open local video')
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or not np.isfinite(fps):
            raise ValueError('Video FPS unavailable')
        previous = None
        try:
            i = 0
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                pts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                timestamp = pts if args.timestamp_policy == 'pts' else i / fps
                if not np.isfinite(timestamp) or (previous is not None and timestamp <= previous):
                    raise ValueError('Non-increasing video timestamps; inspect video; explicit --timestamp-policy fps is available for CFR approximation')
                previous = timestamp
                if i % args.stride == 0:
                    yield i, timestamp, bgr, {'reported_pos_msec': pts * 1000, 'timestamp_source': args.timestamp_policy}
                i += 1
        finally:
            cap.release()


def main():
    ap = argparse.ArgumentParser()
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument('--video', type=Path)
    source.add_argument('--tum', action='store_true')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--max-frames', type=int, default=60, help='0 means all frames')
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--timestamp-policy', choices=['pts', 'fps'], default='pts')
    ap.add_argument('--camera', type=Path, help='Optional calibrated raw-camera JSON; undistort RGB before inference')
    ap.add_argument('--resize-width', type=int, default=0, help='Explicit RGB resize before prediction; 0 preserves decoded dimensions')
    args = ap.parse_args()
    if args.stride < 1 or args.max_frames < 0:
        ap.error('stride >= 1 and max-frames >= 0 required')
    if args.resize_width and (args.resize_width < 32 or args.resize_width % 2):
        ap.error('resize-width must be 0 or an even integer >=32')
    if args.camera and args.resize_width:
        ap.error('Combined calibration and resizing is not supported; use one explicit pixel transform')
    if args.tum and args.resize_width:
        ap.error('TUM sensor checking currently requires original resolution; resize-width is supported for video input')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    for name in ('rgb', 'depth_m', 'preview'):
        (out / name).mkdir()
    began = time.perf_counter()
    rows = []
    encoder = None
    encoder_log = None
    config = {'command': sys.argv, 'cwd': str(Path.cwd()), 'source': 'TUM images' if args.tum else str(args.video.resolve()),
              'source_sha256': sha(DATA / 'rgb.txt') if args.tum else sha(args.video),
              'stride': args.stride, 'max_frames': args.max_frames, 'input_size': 518,
              'resize_width': args.resize_width,
              'timestamp_policy': 'TUM original timestamps' if args.tum else args.timestamp_policy,
              'display_depth_m': [0, 4], 'display_note': 'fixed TURBO; above 4 m saturated; black invalid; numeric arrays not clipped',
              'script_sha256': sha(Path(__file__))}
    write(out / 'config.json', config)
    status = 'error'
    try:
        t = time.perf_counter()
        model, provenance = load_model()
        config['model'] = provenance
        config['model_load_s'] = time.perf_counter() - t
        if args.tum:
            fps = 30.0 / args.stride
        else:
            probe = cv2.VideoCapture(str(args.video.resolve()))
            config['video_metadata'] = {k: probe.get(v) for k, v in [('fps', cv2.CAP_PROP_FPS), ('reported_frames', cv2.CAP_PROP_FRAME_COUNT), ('width', cv2.CAP_PROP_FRAME_WIDTH), ('height', cv2.CAP_PROP_FRAME_HEIGHT)]}
            fps = config['video_metadata']['fps'] / args.stride
            probe.release()
        config['preview_fps'] = fps
        config['preview_timing_note'] = 'CFR display from reported FPS, not measured processing speed; numerical frame manifest retains timestamps; TUM preview assumes nominal 30 FPS'
        write(out / 'config.json', config)
        shape = None
        rectify_maps = None
        camera = json.loads(args.camera.read_text(encoding='utf-8')) if args.camera else None
        matches = json.loads((ROOT / 'results/step234_full_retry/matching.json').read_text()) if args.tum else None
        dep_rows = read_index(DATA / 'depth.txt') if args.tum else None
        for source_index, timestamp, bgr, origin in stream(args):
            decoded_hw = list(bgr.shape[:2])
            if args.resize_width:
                target_h = int(round(bgr.shape[0] * args.resize_width / bgr.shape[1] / 2)) * 2
                old_h, old_w = bgr.shape[:2]
                bgr = cv2.resize(bgr, (args.resize_width, target_h), interpolation=cv2.INTER_AREA)
                if shape is None:
                    config['pixel_transform'] = {'decoded_hw':[old_h,old_w], 'processed_hw':[target_h,args.resize_width],
                        'sx':args.resize_width/old_w,'sy':target_h/old_h, 'interpolation':'OpenCV INTER_AREA on RGB BEFORE depth inference; no independent depth stretching'}
            if shape is None:
                shape = bgr.shape
                h, w = shape[:2]
                if w % 2 or h % 2:
                    raise ValueError('Video preview requires even image dimensions; numeric images are never silently resized')
                if camera:
                    if (camera['width'], camera['height']) != (w, h):
                        raise ValueError('Calibration resolution differs from decoded video')
                    K = np.array([[camera['fx'],0,camera['cx']], [0,camera['fy'],camera['cy']], [0,0,1]], float)
                    dist = np.array(camera['distortion'], float)
                    newK, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w,h), 0, (w,h))
                    rectify_maps = cv2.initUndistortRectifyMap(K, dist, None, newK, (w,h), cv2.CV_32FC1)
                    effective = {'width':w,'height':h,'fx':float(newK[0,0]),'fy':float(newK[1,1]),'cx':float(newK[0,2]),'cy':float(newK[1,2]),'distortion':[0,0,0,0,0],
                                 'source':'RGB undistorted before model inference; getOptimalNewCameraMatrix alpha=0; full output dimensions, no ROI crop', 'calibration_file':str(args.camera.resolve()), 'calibration_sha256':sha(args.camera)}
                    write(out/'effective_intrinsics.json', effective)
                    config['rectification'] = effective
                encoder_log = (out / 'encoder.log').open('x', encoding='utf-8')
                cmd = [str(FFMPEG), '-n', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{2*w}x{h+60}', '-r', str(fps), '-i', '-', '-an', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out / 'depth_replay.mp4')]
                config['encoder_command'] = cmd
                encoder = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=encoder_log)
            if bgr.shape != shape:
                raise ValueError('Frame dimensions changed')
            if rectify_maps is not None:
                bgr = cv2.remap(bgr, *rectify_maps, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            t = time.perf_counter()
            depth = predict(model, bgr)
            inference_s = time.perf_counter() - t
            n = len(rows)
            stem = f'{n:06d}'
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            Image.fromarray(rgb).save(out / 'rgb' / (stem + '.png'))
            np.save(out / 'depth_m' / (stem + '.npy'), depth, allow_pickle=False)
            preview = cv2.cvtColor(color_depth(depth), cv2.COLOR_BGR2RGB)
            canvas = Image.new('RGB', (2*w, h+60), 'white')
            canvas.paste(Image.fromarray(rgb), (0, 30))
            canvas.paste(Image.fromarray(preview), (w, 30))
            draw = ImageDraw.Draw(canvas)
            draw.text((8, 8), f'RGB | frame {n} | t={timestamp:.6f}', fill='black')
            draw.text((w+8, 8), 'Metric prediction (m); not sensor measurement', fill='black')
            bar = cv2.cvtColor(cv2.applyColorMap(np.tile(np.arange(256, dtype=np.uint8), (12, 1)), cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
            canvas.paste(Image.fromarray(bar), (w+8, h+36))
            draw.text((w+270, h+35), '0 to 4 m; above 4 saturated; black invalid', fill='black')
            canvas.save(out / 'preview' / (stem + '.png'))
            encoder.stdin.write(canvas.tobytes())
            row = {'index': n, 'source_index': source_index, 'timestamp_s': timestamp, 'origin': origin,
                   'decoded_hw': decoded_hw,
                   'rgb': f'rgb/{stem}.png', 'depth_m': f'depth_m/{stem}.npy', 'preview': f'preview/{stem}.png',
                   'rgb_sha256': sha(out / 'rgb' / (stem+'.png')), 'depth_sha256': sha(out / 'depth_m' / (stem+'.npy')),
                   'shape_hw': list(depth.shape), 'dtype': str(depth.dtype), 'unit': 'm (model predicted)',
                   'finite_count': int(np.isfinite(depth).sum()), 'positive_count': int((depth > 0).sum()),
                   'min_m': float(depth.min()), 'max_m': float(depth.max()), 'median_m': float(np.median(depth)), 'inference_s': inference_s}
            if args.tum:
                pair = next((p for p in matches['pairs'] if p['rgb_index'] == source_index), None)
                if pair:
                    dt, name = dep_rows[pair['depth_index']]
                    ref = cv2.imdecode(np.fromfile(str(DATA/name), dtype=np.uint8), cv2.IMREAD_UNCHANGED).astype(np.float64)/5000
                    valid = (ref > 0) & (ref < 4) & np.isfinite(depth) & (depth > 0)
                    error = depth[valid].astype(float)-ref[valid]
                    row['sensor_check'] = {'reference_path': str(DATA/name), 'sha256': sha(DATA/name), 'time_difference_ms': pair['difference_ms'], 'valid_count': int(valid.sum()), 'coverage_of_all': float(valid.mean()), 'mae_m': float(np.abs(error).mean()), 'rmse_m': float(np.sqrt(np.mean(error**2))), 'rule': 'no fit or scale correction; reference 0<z<4 and prediction finite positive; model values not range-clipped'}
            rows.append(row)
            write(out / 'frames.json', rows)
            print(f'frame={n} source={source_index} timestamp={timestamp:.6f} inference={inference_s:.3f}s', flush=True)
            if args.max_frames and len(rows) >= args.max_frames:
                break
        if not rows:
            raise ValueError('No decoded frames')
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError('Video encoder failed; see encoder.log')
        status = 'completed'
        times = [r['inference_s'] for r in rows]
        summary = {'status': status, 'frames': len(rows), 'first_timestamp': rows[0]['timestamp_s'], 'last_timestamp': rows[-1]['timestamp_s'],
                   'wall_seconds': time.perf_counter()-began, 'inference_mean_s': float(np.mean(times)), 'inference_p95_s': float(np.percentile(times, 95)),
                   'cuda_peak_allocated_bytes': int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None,
                   'limits': 'Framewise metric predictions; no claim of accurate distance, temporal consistency or trajectory. Preview playback is independent of processing speed.'}
        write(out / 'summary.json', summary)
        write(out / 'config.json', config)
        html = '<!doctype html><meta charset="utf-8"><title>米制预测深度回放</title><h1>普通视频／序列 → 预测深度</h1><p>模型预测，单位米；颜色固定0–4米，黑色无效。播放速度不代表计算速度。</p><video controls width="100%" src="depth_replay.mp4"></video><p><a href="frames.json">逐帧数值记录</a> · <a href="summary.json">运行统计</a> · <a href="config.json">配置与来源</a></p>'
        (out / 'index.html').write_text(html, encoding='utf-8')
        print(json.dumps(summary), flush=True)
    except Exception:
        (out / 'error.log').write_text(traceback.format_exc(), encoding='utf-8')
        write(out / 'summary.json', {'status': 'error', 'frames': len(rows), 'wall_seconds': time.perf_counter()-began})
        raise
    finally:
        if encoder and encoder.poll() is None:
            if encoder.stdin and not encoder.stdin.closed:
                encoder.stdin.close()
            encoder.wait()
        if encoder_log:
            encoder_log.close()


if __name__ == '__main__':
    main()
