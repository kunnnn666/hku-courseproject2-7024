import sys,uuid,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.rgbd_geometry import rigid,transform,ransac,backproject
from src.trajectory_metrics import evaluate
from run_step05 import save,load
run=ROOT/'runs'/('F05_verify_'+uuid.uuid4().hex[:8]);run.mkdir()
cfg=load(ROOT/'configs/geometry_step05.json');rng=np.random.default_rng(7024)
a=rng.normal(size=(100,3));angle=.3
T=np.eye(4);T[:3,:3]=[[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]];T[:3,3]=[.2,-.1,.4]
b=transform(T,a);fit=rigid(a,b)
assert np.max(abs(fit-T))<1e-12 and np.max(abs(rigid(a,a)-np.eye(4)))<1e-12
assert np.max(abs(fit[:3,:3].T@fit[:3,:3]-np.eye(3)))<1e-12 and abs(np.linalg.det(fit[:3,:3])-1)<1e-12
b[70:]=rng.normal(size=(30,3))*3
r=ransac(a,b,'synthetic',cfg);assert r['valid'] and r['inliers']==70
failures=[]
for points in (np.zeros((20,3)),np.column_stack((np.arange(20),np.zeros((20,2))))):
    try:rigid(points,points);raise AssertionError('degenerate accepted')
    except ValueError as e:failures.append(str(e))
depth=np.zeros((480,640),np.uint16);depth[240,320]=10000
xyz,mask=backproject([[320,240],[-.1,240],[639.9,240],[0,0]],depth,[525,525,319.5,239.5])
assert mask.tolist()==[True,False,False,False] and np.allclose(xyz[0],[1/525,1/525,2])
empty=ransac(a[:0],a[:0],'empty',cfg);assert not empty['valid']
U=T.copy();U[:3,3]=[-.3,.2,.1];U[:3,:3]=T[:3,:3].T
world1=np.linalg.inv(T);world2=world1@np.linalg.inv(U)
assert np.allclose(world2@U@T,np.eye(4),atol=1e-12)
entries=[]
for i in range(12):
    pose=np.eye(4);pose[:3,3]=[np.sin(i*.4),np.cos(i*.4),i*.1]
    entries.append(dict(pair_id=str(i),timestamp=str(i),segment_id=0 if i<8 else 1,estimate=(T@pose).tolist(),reference=pose.tolist()))
ev=evaluate(entries);assert ev['ate_rmse_m']<1e-12 and len(ev['rpe_pairs'])==7
for e in entries:e['estimate']=np.array(e['reference']);e['estimate'][:3,3]*=2;e['estimate']=e['estimate'].tolist()
scale=evaluate(entries);assert scale['ate_rmse_m']>.1
save(run/'result.json',dict(exit_code=0,command=[sys.executable,*sys.argv],rigid_max_error=float(np.max(abs(fit-T))),ransac=r,
                           degeneracy=failures,depth_hand_example=xyz[0].tolist(),depth_mask=mask.tolist(),empty=empty,
                           chain_max_error=float(np.max(abs(world2@U@T-np.eye(4)))),coordinate_change_ate=ev['ate_rmse_m'],
                           scale_error_ate=scale['ate_rmse_m'],same_segment_rpe_pairs=len(ev['rpe_pairs']),tolerance=1e-12))
print(run)
