import sys,json
from pathlib import Path
import numpy as np
sys.path.insert(0,'src');import continuous_vo as v
T=np.eye(4);T[:3,3]=[-1,0,0];assert np.allclose((np.eye(4)@np.linalg.inv(T))[:3,3],[1,0,0]);R=np.array([[0.,-1,0],[1,0,0],[0,0,1]]);assert np.allclose(v.rotation(v.quat(R)),R)
r=[(0.0,'a'),(.1,'b'),(.2,'c')];d=[(.001,'x'),(.101,'y')];pairs,ur,ud=v.matching(r,d);assert len(pairs)==2 and ur==[2] and not ud
assert v.rigid(np.eye(4)) and not v.rigid(np.diag([2.,1,1,1])) and not v.rigid(np.full((4,4),np.nan))
gt=np.array([[float(i),float(i),0,0,0,0,0,1] for i in range(5)]);poses=[]
for i in range(5):
 P=np.eye(4);P[0,3]=i;poses.append({'t':float(i),'T':P})
m,a=v.evaluate(poses,gt);assert m['associated']==5 and m['ate_rmse_m']<1e-10 and m['rpe_pairs']==4 and m['rpe_translation_rmse_m']<1e-10
v.o3d.pipelines.odometry.compute_rgbd_odometry=lambda *a:(False,np.eye(4),np.eye(6));sys.argv=['continuous_vo.py','--output','results/validation_forced_failure','--frames','3'];v.main()
s=json.loads(Path('results/validation_forced_failure/state.json').read_text(encoding='utf-8'));assert s['status']=='interrupted' and s['failed_transitions']==1 and s['estimated_poses']==1 and s['processed_frames']==2
print('PASS transform, quaternion, unmatched, invalid matrix, known ATE/RPE, forced tracking failure')
