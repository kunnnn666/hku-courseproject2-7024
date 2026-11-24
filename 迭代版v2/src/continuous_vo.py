import argparse, csv, json, os, sys, time, traceback
from pathlib import Path
import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw
from step1_two_frame import DATA, read_index, load_frame
ROOT=Path(__file__).resolve().parents[1]

def write(path, obj):
    tmp=path.with_suffix('.tmp'); tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    for attempt in range(100):
        try: os.replace(tmp,path); break
        except PermissionError:
            if attempt==99: raise
            time.sleep(.02)
def matching(rgb,dep):
    cand=sorted((abs(r[0]-d[0]),i,j) for i,r in enumerate(rgb) for j,d in enumerate(dep) if abs(r[0]-d[0])<=.02)
    usedr=set(); usedd=set(); pairs=[]
    for delta,i,j in cand:
        if i not in usedr and j not in usedd: usedr.add(i); usedd.add(j); pairs.append((i,j,delta))
    pairs.sort(); assert all(pairs[k][1]<pairs[k+1][1] for k in range(len(pairs)-1))
    return pairs,[i for i in range(len(rgb)) if i not in usedr],[i for i in range(len(dep)) if i not in usedd]
def rigid(T):
    return bool(np.isfinite(T).all() and np.allclose(T[3],[0,0,0,1]) and np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-4) and abs(np.linalg.det(T[:3,:3])-1)<1e-4)
def quat(R):
    # Stable eigenvector conversion, quaternion order x y z w.
    K=np.array([[R[0,0]-R[1,1]-R[2,2],R[1,0]+R[0,1],R[2,0]+R[0,2],R[2,1]-R[1,2]], [R[1,0]+R[0,1],R[1,1]-R[0,0]-R[2,2],R[2,1]+R[1,2],R[0,2]-R[2,0]], [R[2,0]+R[0,2],R[2,1]+R[1,2],R[2,2]-R[0,0]-R[1,1],R[1,0]-R[0,1]], [R[2,1]-R[1,2],R[0,2]-R[2,0],R[1,0]-R[0,1],R.trace()]])/3
    _,v=np.linalg.eigh(K); q=v[:,-1]; return q if q[3]>=0 else -q
def rotation(q):
    x,y,z,w=q/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
def ground():
    rows=np.array([[float(x) for x in line.split()] for line in (DATA/'groundtruth.txt').read_text().splitlines() if line.strip() and not line.startswith('#')]); assert rows.shape[1]==8 and np.all(np.diff(rows[:,0])>0) and np.isfinite(rows).all(); return rows
def evaluate(poses,gt):
    assoc=[]
    for p in poses:
        j=int(np.searchsorted(gt[:,0],p['t'])); candidates=[k for k in (j-1,j) if 0<=k<len(gt)]; k=min(candidates,key=lambda k:abs(gt[k,0]-p['t'])); dt=abs(gt[k,0]-p['t'])
        if dt<=.02: assoc.append((p,k,dt))
    if len(assoc)<3:return {'associated':len(assoc),'reason':'fewer than three associations'},[]
    a=np.array([p['T'][:3,3] for p,k,d in assoc]); b=np.array([gt[k,1:4] for p,k,d in assoc]); ac=a.mean(0); bc=b.mean(0); U,s,Vt=np.linalg.svd((a-ac).T@(b-bc)); D=np.eye(3); D[-1,-1]=np.linalg.det(Vt.T@U.T); R=Vt.T@D@U.T; t=bc-R@ac; aligned=(R@a.T).T+t
    errs=np.linalg.norm(aligned-b,axis=1); rpe=[]
    for i,(p,k,d) in enumerate(assoc):
        target=p['t']+1; j=min(range(i+1,len(assoc)),key=lambda z:abs(assoc[z][0]['t']-target),default=None)
        if j is None or abs(assoc[j][0]['t']-target)>.05:continue
        p2,k2,_=assoc[j]; G=np.eye(4); G[:3,:3]=rotation(gt[k,4:8]); G[:3,3]=gt[k,1:4]; G2=np.eye(4); G2[:3,:3]=rotation(gt[k2,4:8]); G2[:3,3]=gt[k2,1:4]
        E=np.linalg.inv(np.linalg.inv(G)@G2)@(np.linalg.inv(p['T'])@p2['T']); rpe.append(float(np.linalg.norm(E[:3,3])))
    rows=[{'timestamp':p['t'],'gt_timestamp':float(gt[k,0]),'difference_ms':d*1000,'estimated':aligned[i].tolist(),'reference':b[i].tolist(),'error_m':float(errs[i])} for i,(p,k,d) in enumerate(assoc)]
    return {'associated':len(assoc),'coverage_of_estimated':len(assoc)/len(poses),'gt_range':[float(gt[0,0]),float(gt[-1,0])],'association_max_s':.02,'association_max_actual_ms':max(x[2] for x in assoc)*1000,'ate_rmse_m':float(np.sqrt(np.mean(errs**2))),'rpe_interval_s':1,'rpe_tolerance_s':.05,'rpe_pairs':len(rpe),'rpe_translation_rmse_m':float(np.sqrt(np.mean(np.square(rpe)))) if rpe else None,'alignment':'SE(3) rigid least-squares; no scale','alignment_rotation':R.tolist(),'alignment_translation_m':t.tolist()},rows
def plot(rows,path):
    im=Image.new('RGB',(1000,700),'white'); d=ImageDraw.Draw(im); d.text((25,15),'Aligned trajectories: blue estimated, orange reference; XY projection (m)',fill='black')
    if rows:
        arr=np.array([r['estimated'][:2] for r in rows]+[r['reference'][:2] for r in rows]); lo=arr.min(0); hi=arr.max(0); scale=min(880/max(hi[0]-lo[0],.01),580/max(hi[1]-lo[1],.01))
        def coords(key):return [(50+(r[key][0]-lo[0])*scale,650-(r[key][1]-lo[1])*scale) for r in rows]
        if len(rows)>1:d.line(coords('reference'),fill='#ef8b33',width=3); d.line(coords('estimated'),fill='#2670c7',width=3)
    im.save(path)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',required=True); ap.add_argument('--frames',type=int,default=100); args=ap.parse_args(); out=Path(args.output).resolve(); out.mkdir(parents=True,exist_ok=False); (out/'frames').mkdir()
    config={'frames_requested':args.frames,'rgb_depth_max_s':.02,'reference_max_s':.02,'depth_scale':5000,'depth_trunc_m':4,'intrinsics':[525,525,319.5,239.5],'transform':'T_target_source; T_world_target = T_world_source @ inverse(T_target_source)','failure_policy':'record unmatched RGB gaps; stop on first odometry failure; no reference substitution','open3d':o3d.__version__,'numpy':np.__version__,'python':sys.version,'threads':os.environ.get('OMP_NUM_THREADS')}; write(out/'config.json',config)
    # Known world camera moves +1 x; source coordinates transformed into target by -1 x.
    M=np.eye(4); M[0,3]=-1; W=np.eye(4)@np.linalg.inv(M); assert np.allclose(W[:3,3],[1,0,0]) and np.allclose(M@np.array([1,0,0,1]),[0,0,0,1]); assert 5000/config['depth_scale']==1
    rgb,dep=read_index(DATA/'rgb.txt'),read_index(DATA/'depth.txt'); pairs,ur,ud=matching(rgb,dep); gt=ground(); write(out/'matching.json',{'rule':'global candidate time difference ascending; greedy one-to-one; then RGB order; not globally optimal','limit_s':.02,'rgb_count':len(rgb),'depth_count':len(dep),'matched':len(pairs),'unmatched_rgb':[{'index':i,'timestamp':rgb[i][0],'file':rgb[i][1]} for i in ur],'unmatched_depth':[{'index':i,'timestamp':dep[i][0],'file':dep[i][1]} for i in ud],'pairs':[{'rgb_index':i,'depth_index':j,'difference_ms':d*1000} for i,j,d in pairs]})
    config['requested_available']=min(args.frames,len(pairs)); write(out/'config.json',config)
    intr=o3d.camera.PinholeCameraIntrinsic(640,480,525,525,319.5,239.5); poses=[]; logs=[]; start=time.perf_counter(); previous=None; lastidx=None; status='completed'; reason=''; success=0; failure=0
    try:
      for n,(i,j,delta) in enumerate(pairs[:args.frames]):
        skipped_rgb_indices=list(range(lastidx+1,i)) if lastidx is not None else []
        frame,c,d,info=load_frame((rgb[i],dep[j],delta)); t0=time.perf_counter(); T=np.eye(4) if not poses else poses[-1]['T'].copy(); ok=True; transform=None
        if previous is not None:
            ok,transform,information=o3d.pipelines.odometry.compute_rgbd_odometry(previous,frame,intr,np.eye(4),o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),o3d.pipelines.odometry.OdometryOption()); ok=bool(ok and rigid(transform) and np.isfinite(information).all())
            if ok:T=T@np.linalg.inv(transform); success+=1
            else:failure+=1
        entry={'frame':n,'skipped_rgb_indices':skipped_rgb_indices,'rgb_index':i,'depth_index':j,'rgb_timestamp':rgb[i][0],'depth_timestamp':dep[j][0],'difference_ms':delta*1000,'status':'initial' if n==0 else ('success' if ok else 'failure'),'elapsed_odometry_s':time.perf_counter()-t0,'T_target_source':transform.tolist() if transform is not None and np.isfinite(transform).all() else None,'input':info}
        logs.append(entry)
        if not ok:status='interrupted'; reason=f'odometry failed at frame {n}'; break
        poses.append({'t':rgb[i][0],'T':T}); previous=frame; lastidx=i
        Image.fromarray(c).resize((320,240)).save(out/'frames'/f'{n:04d}_rgb.jpg',quality=78)
        z=np.clip(d.astype(np.float32)/5000/4,0,1); vis=np.stack([z*255,(1-z)*255,np.full_like(z,100)],-1).astype(np.uint8); vis[d==0]=0; Image.fromarray(vis).resize((320,240)).save(out/'frames'/f'{n:04d}_depth.jpg',quality=78)
        metrics,assoc=evaluate(poses,gt); state={'status':'running','processed_frames':len(logs),'estimated_poses':len(poses),'successful_transitions':success,'failed_transitions':failure,'unmatched_rgb_total':len(ur),'unmatched_depth_total':len(ud),'elapsed_s':time.perf_counter()-start,'requested_frames':args.frames,'first_timestamp':poses[0]['t'],'last_timestamp':poses[-1]['t'],'metrics':metrics,'points':assoc,'frames':[{'index':k,'timestamp':p['t']} for k,p in enumerate(poses)],'reason':reason}; write(out/'state.json',state)
        if (out/'STOP').exists():status='stopped'; reason='user requested stop'; break
    except Exception as e:
      status='error'; reason=repr(e); (out/'error.log').write_text(traceback.format_exc(),encoding='utf-8')
    metrics,assoc=evaluate(poses,gt); state={'status':status,'processed_frames':len(logs),'estimated_poses':len(poses),'successful_transitions':success,'failed_transitions':failure,'unmatched_rgb_total':len(ur),'unmatched_depth_total':len(ud),'elapsed_s':time.perf_counter()-start,'requested_frames':args.frames,'first_timestamp':poses[0]['t'] if poses else None,'last_timestamp':poses[-1]['t'] if poses else None,'metrics':metrics,'points':assoc,'frames':[{'index':k,'timestamp':p['t']} for k,p in enumerate(poses)],'reason':reason}; write(out/'state.json',state); write(out/'frame_log.json',logs); write(out/'associations.json',assoc); write(out/'metrics.json',metrics); plot(assoc,out/'trajectory.png')
    with (out/'trajectory.txt').open('w',encoding='utf-8') as f:
        f.write('# timestamp tx ty tz qx qy qz qw; camera pose relative to first RGB camera\n')
        for p in poses: f.write(f"{p['t']:.6f} "+' '.join(f'{x:.9f}' for x in [*p['T'][:3,3],*quat(p['T'][:3,:3])])+'\n')
    print(json.dumps(state,ensure_ascii=False,default=str)[:2000]); return 1 if status=='error' else 0
if __name__=='__main__':sys.exit(main())
