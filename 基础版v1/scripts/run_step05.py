"""Observation-only estimation. No evaluation index or groundtruth access."""
import argparse,json,sys,time,uuid,traceback
from pathlib import Path
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.feature_frontend import match_images
from src.rgbd_geometry import backproject,ransac
from src.image_baseline import sha256
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--frames',type=int,default=32);parser.add_argument('--gap',type=int,default=1)
    args=parser.parse_args();run=ROOT/'runs'/('F05_est_'+uuid.uuid4().hex[:10]);run.mkdir()
    start=time.perf_counter();report=dict(command=[sys.executable,*sys.argv],cwd=str(Path.cwd()),run_dir=str(run))
    try:
        cv2.setNumThreads(1);cv2.setRNGSeed(7024)
        cfg=load(ROOT/'configs/geometry_step05.json');front=load(ROOT/'configs/frontend_step04.json')
        files=[ROOT/'data/manifests/step02'/n for n in ('dev_pairs.jsonl','roots_and_cameras.json')]
        frozen=load(ROOT/'handoffs/step02/handoff.json')['manifests']
        for p in files:assert sha256(p.read_bytes())==frozen[str(p)]
        cam=load(files[1])['sequences']['dev'];assert cam['K']==[525.,525.,319.5,239.5] and cam['depth_units_per_m']==5000
        rows=[json.loads(s) for s in files[0].read_text().splitlines()][:args.frames:args.gap]
        assert len(rows)>=2
        save(run/'inputs.json',rows);save(run/'config.json',dict(geometry=cfg,frontend=front,camera=cam))
        reads=[]
        def read(row):
            arrays=[]
            for kind in ('rgb','depth'):
                p=(Path(cam['root'])/row[kind+'_path']).resolve();assert p.is_relative_to(Path(cam['root']).resolve())
                data=p.read_bytes();assert sha256(data)==row[kind+'_sha256']
                im=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_UNCHANGED)
                assert im is not None and im.shape==((480,640,3) if kind=='rgb' else (480,640))
                assert im.dtype==(np.uint8 if kind=='rgb' else np.uint16)
                reads.append(dict(path=str(p),sha256=sha256(data)));arrays.append(im)
            return arrays[0][...,::-1].astype(np.float32)/255,arrays[1]
        poses=[];diagnostics=[];segment=0;world=np.eye(4);previous=None
        for i,row in enumerate(rows):
            current=read(row);reason=None;origin=i==0
            if previous is not None:
                tick=time.perf_counter();assert float(row['rgb_timestamp'])>float(rows[i-1]['rgb_timestamp'])
                m=match_images(previous[0],current[0],front['orb'],front['ratio'])
                a,va=backproject([x['source_uv'] for x in m['mutual']],previous[1],cam['K'],cfg['depth_range_m'])
                b,vb=backproject([x['target_uv'] for x in m['mutual']],current[1],cam['K'],cfg['depth_range_m'])
                valid=va&vb;result=ransac(a[valid],b[valid],rows[i-1]['pair_id']+'->'+row['pair_id'],cfg)
                result.update(source=rows[i-1]['pair_id'],target=row['pair_id'],source_row=(i-1)*args.gap,target_row=i*args.gap,
                              matches=len(m['mutual']),source_depth_valid=int(va.sum()),target_depth_valid=int(vb.sum()),
                              seconds=time.perf_counter()-tick)
                if result['valid']:world=world@np.linalg.inv(result['T_target_source'])
                else:segment+=1;world=np.eye(4);origin=True;reason=result['reason']
                diagnostics.append(result)
            poses.append(dict(pair_id=row['pair_id'],timestamp=row['rgb_timestamp'],segment_id=segment,segment_start=origin,
                              valid=True,incoming_valid=None if i==0 else diagnostics[-1]['valid'],failure_reason=reason,T_world_camera=world.tolist()))
            previous=current
            if i%100==0:print('frame',i,flush=True)
        save(run/'trajectory.json',poses);save(run/'diagnostics.json',diagnostics);save(run/'reads.json',reads)
        report.update(exit_code=0,status='completed',frames=len(rows),attempts=len(diagnostics),successes=sum(d['valid'] for d in diagnostics),
                      failures=sum(not d['valid'] for d in diagnostics),segments=segment+1,numpy=np.__version__,opencv=cv2.__version__,
                      hashes={str(p):sha256(p.read_bytes()) for p in files+[Path(__file__),ROOT/'src/rgbd_geometry.py',ROOT/'src/feature_frontend.py',ROOT/'configs/geometry_step05.json',ROOT/'configs/frontend_step04.json']})
    except Exception:report.update(exit_code=1,status='failed',error=traceback.format_exc())
    report['seconds']=time.perf_counter()-start;save(run/'result.json',report);print(json.dumps(report));return report['exit_code']
if __name__=='__main__':sys.exit(main())
