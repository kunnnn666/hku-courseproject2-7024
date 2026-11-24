import argparse,sys,uuid,traceback
import numpy as np
import cv2
from run_step05 import ROOT,load,save
from src.trajectory_metrics import quaternion_pose,evaluate
from src.image_baseline import sha256
p=argparse.ArgumentParser();p.add_argument('--run',required=True);args=p.parse_args()
source=ROOT/args.run;run=ROOT/'runs'/('F05_eval_'+uuid.uuid4().hex[:8]);run.mkdir()
report=dict(command=[sys.executable,*sys.argv],estimate_run=str(source),run_dir=str(run))
try:
    index=ROOT/'data/manifests/step02/dev_evaluation_index.jsonl'
    frozen=load(ROOT/'handoffs/step02/handoff.json')['manifests'];assert sha256(index.read_bytes())==frozen[str(index)]
    cam=load(source/'config.json')['camera'];gtpath=__import__('pathlib').Path(cam['root'])/'groundtruth.txt'
    lines=gtpath.read_text().splitlines();lookup={}
    for line in lines:
        if line.strip() and not line.startswith('#'):
            values=line.split();lookup[values[0]]=quaternion_pose(list(map(float,values[1:])))
    associations={r['pair_id']:r for r in [__import__('json').loads(s) for s in index.read_text().splitlines()]}
    poses=load(source/'trajectory.json');entries=[]
    for pose in poses:
        if pose['pair_id'] in associations:
            entry=dict(pose);entry.update(estimate=pose['T_world_camera'],reference=lookup[associations[pose['pair_id']]['gt_timestamp']].tolist());entries.append(entry)
    cfg=load(source/'config.json')['geometry'];metrics=evaluate(entries,cfg['rpe_dt_s'],cfg['rpe_tolerance_s'])
    save(run/'metrics.json',metrics);save(run/'associated.json',entries)
    def plot(name,series):
        canvas=np.full((650,900,3),255,np.uint8);points=np.concatenate([v for _,v in series]);lo=points.min(0);span=np.maximum(points.max(0)-lo,.01)
        for (label,values),color in zip(series,[(200,50,20),(20,30,210)]):
            xy=((values-lo)/max(span)*500+70).astype(int)
            for a,b in zip(xy[:-1],xy[1:]):cv2.line(canvas,tuple(a),tuple(b),color,2)
            cv2.circle(canvas,tuple(xy[0]),7,color,-1);cv2.putText(canvas,label,(620,50+30*list(series).index((label,values))) if False else (620,50 if color[0]==200 else 80),0,.5,color,1)
        cv2.putText(canvas,'XY projection (m), start = dot; image y downward',(20,620),0,.6,(0,0,0),1)
        cv2.imencode('.png',canvas)[1].tofile(str(run/name))
    main=metrics['main_segment'];raw=np.array([e['estimate'] for e in entries if e['segment_id']==main])
    plot('raw_xy.png',[('raw local',raw[:,:2,3])]);plot('aligned_xy.png',[('aligned',np.array(metrics['aligned'])[:,:2,3]),('reference',np.array(metrics['reference'])[:,:2,3])])
    diagnostics=load(source/'diagnostics.json');canvas=np.full((400,1000,3),255,np.uint8)
    for i,d in enumerate(diagnostics):
        x=40+int(i/max(1,len(diagnostics)-1)*920);cv2.line(canvas,(x,350),(x,350-int(d.get('inliers',0))), (0,150,0) if d['valid'] else (0,0,255),1)
    cv2.putText(canvas,'Transition index ->; green=inliers, red=failed (no pose carried)',(10,25),0,.5,(0,0,0),1)
    cv2.imencode('.png',canvas)[1].tofile(str(run/'tracking.png'))
    report.update(exit_code=0,associated=metrics['associated'],coverage=metrics['associated']/len(poses),ate_rmse_m=metrics['ate_rmse_m'],rpe_translation_rmse_m=metrics['rpe_translation_rmse_m'],rpe_rotation_rmse_deg=metrics['rpe_rotation_rmse_deg'],rpe_pairs=len(metrics['rpe_pairs']),hashes={str(index):sha256(index.read_bytes()),str(gtpath):sha256(gtpath.read_bytes())})
except Exception:report.update(exit_code=1,error=traceback.format_exc())
save(run/'result.json',report);print(report);sys.exit(report['exit_code'])
