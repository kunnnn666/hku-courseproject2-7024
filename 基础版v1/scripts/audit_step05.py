"""Cross-step interface and independent replay audit; raw source remains read-only."""
import sys,uuid
import cv2
import numpy as np
from run_step05 import ROOT,load,save
from src.rgbd_geometry import backproject,ransac
from src.image_baseline import sha256
run=ROOT/'runs'/('F05_audit_'+uuid.uuid4().hex[:8]);run.mkdir()
first=ROOT/'runs/F05_est_209f3e2ef6';second=ROOT/'runs/F05_est_500be29900'
assert (first/'trajectory.json').read_bytes()==(second/'trajectory.json').read_bytes()
a,b=load(first/'diagnostics.json'),load(second/'diagnostics.json')
for x,y in zip(a,b):x.pop('seconds');y.pop('seconds')
assert a==b
saved=load(ROOT/'runs/F04_20261107_100012_1a22f6/matches.json')
cfg=load(first/'config.json');rows=load(first/'inputs.json');examples=[]
for record in saved:
    if record['method']!='original':continue
    points=[];valids=[]
    for side,i in zip(('source','target'),record['pair']):
        row=rows[i];assert record[side]['pair_id']==row['pair_id'] and record[side]['timestamp']==row['rgb_timestamp']
        assert record[side]['shape']==[480,640,3] and record[side]['sha256']==row['rgb_sha256']
        from pathlib import Path
        p=Path(cfg['camera']['root'])/row['depth_path'];assert sha256(p.read_bytes())==row['depth_sha256']
        depth=cv2.imdecode(np.frombuffer(p.read_bytes(),np.uint8),cv2.IMREAD_UNCHANGED)
        uv=[m[side+'_uv'] for m in record['mutual']]
        assert all(m[side+'_uv']==record[side+'_points'][m[side+'_index']] for m in record['mutual'])
        xyz,valid=backproject(uv,depth,cfg['camera']['K'],cfg['geometry']['depth_range_m']);points.append(xyz);valids.append(valid)
    mask=valids[0]&valids[1];key=record['source']['pair_id']+'->'+record['target']['pair_id']
    result=ransac(points[0][mask],points[1][mask],key,cfg['geometry']);examples.append(dict(pair=record['pair'],result=result))
    if record['pair'][1]==record['pair'][0]+1:assert result['T_target_source']==a[record['pair'][0]]['T_target_source']
save(run/'result.json',dict(exit_code=0,command=[sys.executable,*sys.argv],short_trajectory_byte_identical=True,diagnostics_equal_without_seconds=True,
                          compared=[str(first),str(second)],f04_original_examples=examples,estimator_gt_imports=False,
                          dependency_audit='run_step05 imports feature_frontend, rgbd_geometry, image_baseline; no evaluation module. reads.json enumerates actual observation file reads.'))
print(run)
