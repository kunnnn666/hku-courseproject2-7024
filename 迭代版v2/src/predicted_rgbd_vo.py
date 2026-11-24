"""Open3D VO using saved metric float32 arrays; no PyTorch dependency."""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw

from continuous_vo import rigid, quat


def write(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def rgbd(rgb, depth):
    if depth.dtype != np.float32 or rgb.shape[:2] != depth.shape:
        raise ValueError('RGB/depth size or dtype mismatch')
    clean = depth.copy()
    clean[~np.isfinite(clean) | (clean <= 0)] = 0
    frame = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(np.ascontiguousarray(rgb)), o3d.geometry.Image(clean),
        depth_scale=1.0, depth_trunc=4.0, convert_rgb_to_intensity=True)
    mask = (clean > 0) & (clean < 4)
    if mask.sum() < 100:
        raise ValueError('Fewer than 100 usable depth pixels')
    if not np.allclose(np.asarray(frame.depth)[mask], clean[mask], atol=1e-6):
        raise ValueError('Meter input was scaled incorrectly')
    return frame, int(mask.sum())


def render(poses, rows, source, output):
    import subprocess
    points = np.array([p['T'][:3, 3] for p in poses])
    lo, hi = points[:, :2].min(0), points[:, :2].max(0)
    scale = min(500/max(.01, hi[0]-lo[0]), 380/max(.01, hi[1]-lo[1]))
    def xy(p):
        return (680+(p[0]-lo[0])*scale, 440-(p[1]-lo[1])*scale)
    fps = json.loads((source/'config.json').read_text(encoding='utf-8'))['preview_fps']
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), '-n', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1280x540', '-r', str(fps), '-i', '-', '-an', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output/'trajectory_replay.mp4')]
    with (output/'encoder.log').open('x', encoding='utf-8') as log:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=log)
        for i, pose in enumerate(poses):
            im = Image.new('RGB', (1280,540), 'white')
            rgb = Image.open(source/rows[i]['rgb']).convert('RGB')
            rgb.thumbnail((640,480))
            im.paste(rgb, ((640-rgb.width)//2,30+(480-rgb.height)//2))
            draw = ImageDraw.Draw(im)
            draw.text((8,8), f'Metric-depth RGB-D VO | frame {i} | t={pose["t"]:.6f}', fill='black')
            draw.text((660,8), 'Estimated camera XY (m), first camera origin', fill='black')
            if i:
                draw.line([xy(p) for p in points[:i+1]], fill='#2378bb', width=3)
            x,y=xy(points[i]); draw.ellipse((x-4,y-4,x+4,y+4),fill='#123456')
            draw.text((660,480), 'No reference used; success flag does not prove accuracy.', fill='black')
            draw.text((660,505), 'Playback speed is independent of processing speed.', fill='black')
            proc.stdin.write(im.tobytes())
            if i == len(poses)-1:
                im.save(output/'trajectory.png')
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError('Trajectory video encoding failed')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--prediction-run', required=True, type=Path)
    ap.add_argument('--intrinsics', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--max-frames', type=int, default=60)
    ap.add_argument('--tracking-width', type=int, default=0, help='Optional explicit tracking resize; RGB BOX, depth NEAREST; K scaled with pixel centers')
    args=ap.parse_args()
    out=args.output.resolve(); out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter(); logs=[]; poses=[]; status='error'; reason=''
    source=args.prediction_run.resolve()
    try:
        rows=json.loads((source/'frames.json').read_text(encoding='utf-8'))
        if args.max_frames > 0:
            rows=rows[:args.max_frames]
        calib=json.loads(args.intrinsics.read_text(encoding='utf-8'))
        if any(abs(x)>1e-12 for x in calib.get('distortion', [])):
            raise ValueError('Nonzero distortion requires explicit upstream undistortion of RGB before prediction, and matching updated K; not implemented here')
        w,h=calib['width'],calib['height']
        original_wh=(w,h)
        if args.tracking_width:
            if args.tracking_width<32:
                raise ValueError('tracking-width must be >=32')
            tw=args.tracking_width; th=round(h*tw/w); sx,sy=tw/w,th/h
            calib=dict(calib, width=tw,height=th,fx=calib['fx']*sx,fy=calib['fy']*sy,
                       cx=(calib['cx']+.5)*sx-.5,cy=(calib['cy']+.5)*sy-.5)
            w,h=tw,th
        intr=o3d.camera.PinholeCameraIntrinsic(w,h,calib['fx'],calib['fy'],calib['cx'],calib['cy'])
        option=o3d.pipelines.odometry.OdometryOption()
        config={'command':sys.argv,'prediction_run':str(source),'intrinsics_path':str(args.intrinsics.resolve()),'intrinsics':calib,'depth_scale':1.0,'depth_trunc_m':4.0,
                'transform':'p_target=T_target_source*p_source; T_world_target=T_world_source*inverse(T_target_source)',
                'failure_policy':'stop at first failed transform; no copied/reference replacement poses','open3d':o3d.__version__,'numpy':np.__version__,
                'odometry_option':str(option),'depth_source':'raw metric-model arrays; no sensor fit or scale adjustment'}
        config['tracking_resize']={'original_wh':list(original_wh),'tracking_wh':[w,h], 'rgb_interpolation':'Pillow BOX' if args.tracking_width else None,
                                   'depth_interpolation':'Pillow NEAREST' if args.tracking_width else None,'K_rule':'f_new=f_old*s; c_new=(c_old+0.5)*s-0.5'}
        write(out/'config.json',config)
        previous=None
        for n,row in enumerate(rows):
            t=row['timestamp_s']
            if poses and t<=poses[-1]['t']:
                raise ValueError('Timestamps not strictly increasing')
            rgb=np.array(Image.open(source/row['rgb']).convert('RGB'))
            depth=np.load(source/row['depth_m'],allow_pickle=False)
            if args.tracking_width:
                if rgb.shape[:2]!=(original_wh[1],original_wh[0]) or depth.shape!=rgb.shape[:2]:
                    raise ValueError('Raw RGB/depth do not match supplied intrinsics dimensions')
                rgb=np.array(Image.fromarray(rgb).resize((w,h),Image.Resampling.BOX))
                depth=np.array(Image.fromarray(depth).resize((w,h),Image.Resampling.NEAREST),dtype=np.float32)
            if rgb.shape!=(h,w,3):
                raise ValueError('Camera intrinsics resolution does not match decoded RGB')
            frame,count=rgbd(rgb,depth); transform=None; ok=True; tick=time.perf_counter()
            if previous is not None:
                ok,transform,information=o3d.pipelines.odometry.compute_rgbd_odometry(previous,frame,intr,np.eye(4),o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),option)
                ok=bool(ok and rigid(transform) and np.isfinite(information).all())
            logs.append({'index':n,'timestamp_s':t,'status':'initial' if n==0 else ('success' if ok else 'failure'),'usable_depth_pixels':count,'odometry_s':time.perf_counter()-tick,
                         'T_target_source':transform.tolist() if transform is not None and np.isfinite(transform).all() else None})
            if not ok:
                status='interrupted'; reason=f'Odometry failed at frame {n}'; break
            T=np.eye(4) if not poses else poses[-1]['T']@np.linalg.inv(transform)
            poses.append({'t':t,'T':T}); previous=frame
        else:
            status='completed'
    except Exception as e:
        reason=repr(e); (out/'error.log').write_text(traceback.format_exc(),encoding='utf-8'); status='error'
    write(out/'frame_log.json',logs)
    with (out/'trajectory.txt').open('x',encoding='utf-8') as f:
        f.write('# timestamp tx ty tz qx qy qz qw; metric-model depth; unvalidated camera pose\n')
        for p in poses:
            f.write(f'{p["t"]:.9f} '+' '.join(f'{x:.12g}' for x in [*p['T'][:3,3],*quat(p['T'][:3,:3])])+'\n')
    summary={'status':status,'reason':reason,'requested_frames':args.max_frames,'attempted_frames':len(logs),'estimated_poses':len(poses),
             'successful_transitions':sum(r['status']=='success' for r in logs),'failed_transitions':sum(r['status']=='failure' for r in logs),
             'wall_before_render_s':time.perf_counter()-started,'accuracy':'not established; no reference trajectory used in this interface demonstration'}
    write(out/'summary.json',summary)
    if poses:
        render(poses,rows,source,out)
        (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>预测深度驱动的相机轨迹</h1><p>以首帧相机为原点，米；尚未验证轨迹准确度。失败状态请查运行统计。</p><video controls width="100%" src="trajectory_replay.mp4"></video><p><a href="summary.json">运行统计</a> · <a href="frame_log.json">逐帧日志</a> · <a href="trajectory.txt">轨迹</a></p>',encoding='utf-8')
    print(json.dumps(summary))
    if status!='completed':
        return 1
    return 0


if __name__=='__main__':
    sys.exit(main())
