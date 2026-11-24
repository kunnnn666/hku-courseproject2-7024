"""Calibrate from local SAME-MODE video or extracted frames; never uploads data."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from metric_runtime import sha


def main():
    ap=argparse.ArgumentParser()
    group=ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--video',type=Path)
    group.add_argument('--images',type=Path)
    ap.add_argument('--output',required=True,type=Path)
    ap.add_argument('--columns',type=int,default=9,help='Inner corner columns')
    ap.add_argument('--rows',type=int,default=6,help='Inner corner rows')
    ap.add_argument('--square-mm',type=float,default=24)
    args=ap.parse_args(); out=args.output.resolve(); out.mkdir(parents=True,exist_ok=False)
    (out/'detections').mkdir()
    if args.columns<3 or args.rows<3 or args.square_mm<=0:
        raise ValueError('Invalid checkerboard specification')
    cap=None
    if args.video:
        cap=cv2.VideoCapture(str(args.video.resolve()))
        fps=cap.get(cv2.CAP_PROP_FPS)
        if not cap.isOpened() or fps<=0:
            raise ValueError('Cannot open calibration video or FPS missing')
        def images():
            n=0
            while True:
                ok,bgr=cap.read()
                if not ok: break
                if n%max(1,round(fps*.5))==0: yield f'video_{n:06d}',bgr
                n+=1
    else:
        def images():
            for p in sorted(args.images.iterdir()):
                if p.suffix.lower() in ('.png','.jpg','.jpeg'):
                    yield p.name,cv2.imdecode(np.fromfile(str(p),dtype=np.uint8),cv2.IMREAD_COLOR)
    obj=np.zeros((args.rows*args.columns,3),np.float32)
    obj[:,:2]=np.mgrid[0:args.columns,0:args.rows].T.reshape(-1,2)*(args.square_mm/1000)
    object_points=[]; image_points=[]; records=[]; size=None
    try:
        for n,(name,bgr) in enumerate(images()):
            if bgr is None:
                records.append({'source':name,'detected':False,'reason':'cannot decode'}); continue
            h,w=bgr.shape[:2]
            if size is None: size=(w,h)
            if size!=(w,h): raise ValueError('Mixed calibration image dimensions; separate camera settings')
            gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
            ok,corners=cv2.findChessboardCornersSB(gray,(args.columns,args.rows),flags=cv2.CALIB_CB_NORMALIZE_IMAGE)
            records.append({'source':name,'detected':bool(ok)})
            if ok:
                object_points.append(obj.copy()); image_points.append(corners)
                cv2.drawChessboardCorners(bgr,(args.columns,args.rows),corners,True)
                success,encoded=cv2.imencode('.png',bgr)
                if success: encoded.tofile(str(out/'detections'/f'{n:04d}.png'))
            if len(object_points)>=50: break
    finally:
        if cap: cap.release()
    report={'input':str((args.video or args.images).resolve()),'input_kind':'video' if args.video else 'images','board_inner_corners':[args.columns,args.rows],'square_mm':args.square_mm,'detections':records,'usable_views':len(object_points), 'opencv':cv2.__version__, 'script_sha256':sha(__file__)}
    if len(object_points)<12:
        report.update(status='insufficient_views',reason='At least 12 detected views required; 20-30 diverse views recommended')
        (out/'quality.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        raise ValueError(report['reason'])
    rms,K,dist,rvecs,tvecs=cv2.calibrateCamera(object_points,image_points,size,None,None)
    per_view=[]
    for op,ip,rv,tv in zip(object_points,image_points,rvecs,tvecs):
        projected,_=cv2.projectPoints(op,rv,tv,K,dist)
        per_view.append(float(np.sqrt(np.mean(np.sum((ip-projected)**2,axis=2)))))
    report.update(status='computed_requires_review',rms_reprojection_px=float(rms),per_view_rms_px=per_view,warning='Small reprojection error alone does not prove correct video intrinsics; verify lens/mode/crop, board coverage, focus and stability')
    camera={'width':size[0],'height':size[1],'fx':float(K[0,0]),'fy':float(K[1,1]),'cx':float(K[0,2]),'cy':float(K[1,2]),'distortion':dist.ravel().tolist(), 'source':'OpenCV calibration from supplied same-mode chessboard observations', 'calibration_status':'computed_requires_review'}
    (out/'camera_raw.json').write_text(json.dumps(camera,indent=2),encoding='utf-8')
    (out/'quality.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    np.savez_compressed(out/'observations.npz',object_points=np.array(object_points),image_points=np.array(image_points),K=K,distortion=dist)
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
