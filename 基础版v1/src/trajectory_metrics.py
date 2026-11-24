"""No-scale alignment and same-segment relative pose errors."""
import numpy as np
from .rgbd_geometry import rigid,transform
def quaternion_pose(values):
    t=np.array(values[:3]);x,y,z,w=np.array(values[3:])/np.linalg.norm(values[3:])
    R=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
    T=np.eye(4);T[:3,:3]=R;T[:3,3]=t;return T
def evaluate(entries,dt=1.,tolerance=.02):
    segments=sorted({e['segment_id'] for e in entries})
    main=max(segments,key=lambda s:sum(e['segment_id']==s for e in entries))
    selected=[e for e in entries if e['segment_id']==main]
    est=np.array([e['estimate'] for e in selected]);gt=np.array([e['reference'] for e in selected])
    alignment=rigid(est[:,:3,3],gt[:,:3,3])
    aligned=alignment@est;errors=np.linalg.norm(aligned[:,:3,3]-gt[:,:3,3],axis=1)
    relative=[]
    for i,a in enumerate(selected):
        candidates=[(abs(float(b['timestamp'])-float(a['timestamp'])-dt),j) for j,b in enumerate(selected) if j>i]
        if not candidates:continue
        difference,j=min(candidates)
        if difference>tolerance:continue
        delta_est=np.linalg.inv(est[i])@est[j];delta_gt=np.linalg.inv(gt[i])@gt[j]
        error=np.linalg.inv(delta_gt)@delta_est
        relative.append(dict(source=a['pair_id'],target=selected[j]['pair_id'],dt=float(selected[j]['timestamp'])-float(a['timestamp']),
                             translation_m=float(np.linalg.norm(error[:3,3])),rotation_deg=float(np.degrees(np.arccos(np.clip((np.trace(error[:3,:3])-1)/2,-1,1))))))
    return dict(main_segment=main,associated=len(selected),all_associated=len(entries),alignment=alignment.tolist(),ate_rmse_m=float(np.sqrt(np.mean(errors**2))),
                rpe_pairs=relative,rpe_translation_rmse_m=float(np.sqrt(np.mean([r['translation_m']**2 for r in relative]))) if relative else None,
                rpe_rotation_rmse_deg=float(np.sqrt(np.mean([r['rotation_deg']**2 for r in relative]))) if relative else None,
                aligned=aligned.tolist(),reference=gt.tolist(),selected_ids=[e['pair_id'] for e in selected])
