"""Column-vector SE(3); row arrays store points only for vectorized arithmetic."""
import hashlib
import numpy as np

def backproject(uv, depth, K, limits=(0.2,5.0)):
    uv=np.asarray(uv,dtype=float).reshape(-1,2)
    if depth.dtype!=np.uint16 or depth.ndim!=2:
        raise ValueError('Need original single-channel uint16 depth')
    h,w=depth.shape
    valid=np.isfinite(uv).all(axis=1)&(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
    ij=np.floor(np.where(np.isfinite(uv),uv,0)+.5).astype(int)
    valid &= (ij[:,0]<w)&(ij[:,1]<h)
    z=depth[np.clip(ij[:,1],0,h-1),np.clip(ij[:,0],0,w-1)].astype(float)/5000
    valid &= (z>=limits[0])&(z<=limits[1])
    fx,fy,cx,cy=K
    xyz=np.column_stack(((uv[:,0]-cx)*z/fx,(uv[:,1]-cy)*z/fy,z))
    return xyz,valid

def rigid(a,b):
    a,b=np.asarray(a,float),np.asarray(b,float)
    if a.shape!=b.shape or len(a)<3 or not np.isfinite([a,b]).all():
        raise ValueError('insufficient_or_nonfinite')
    ac,bc=a-a.mean(0),b-b.mean(0)
    if min(np.linalg.svd(ac,compute_uv=False)[1],np.linalg.svd(bc,compute_uv=False)[1])<1e-6:
        raise ValueError('collinear_or_duplicate')
    U,s,Vt=np.linalg.svd(ac.T@bc)
    correction=np.diag([1.,1.,np.linalg.det(Vt.T@U.T)])
    R=Vt.T@correction@U.T
    T=np.eye(4);T[:3,:3]=R;T[:3,3]=b.mean(0)-R@a.mean(0)
    return T

def transform(T,a):
    return np.asarray(a)@T[:3,:3].T+T[:3,3]

def ransac(a,b,key,config):
    seed=int.from_bytes(hashlib.sha256(('F05:'+key+':7024').encode()).digest()[:8],'big')
    out=dict(valid=False,reason='insufficient_candidates',candidates=len(a),seed=str(seed),degenerate_samples=0)
    if len(a)<config['min_inliers']:return out
    rng=np.random.default_rng(seed);best=None;score=None
    for iteration in range(config['iterations']):
        ids=rng.choice(len(a),3,replace=False)
        try:T=rigid(a[ids],b[ids])
        except ValueError:
            out['degenerate_samples']+=1;continue
        errors=np.linalg.norm(transform(T,a)-b,axis=1);mask=errors<=config['threshold_m']
        n=int(mask.sum());candidate=(n,-float(errors[mask].sum()))
        if score is None or candidate>score:score,best=candidate,mask
    required=max(config['min_inliers'],int(np.ceil(config['min_fraction']*len(a))))
    if best is None or best.sum()<required:
        out['reason']='insufficient_consensus';return out
    try:
        T=rigid(a[best],b[best])
        errors=np.linalg.norm(transform(T,a)-b,axis=1);mask=errors<=config['threshold_m']
        if mask.sum()<required:out['reason']='refit_consensus_lost';return out
        T=rigid(a[mask],b[mask]);errors=np.linalg.norm(transform(T,a)-b,axis=1);mask=errors<=config['threshold_m']
    except ValueError as e:out['reason']=str(e);return out
    out.update(valid=bool(mask.sum()>=required),reason='ok' if mask.sum()>=required else 'final_consensus_lost',
               T_target_source=T.tolist(),inliers=int(mask.sum()),inlier_indices=np.flatnonzero(mask).tolist(),
               rmse_m=float(np.sqrt(np.mean(errors[mask]**2))),residuals_m=errors.tolist())
    return out
