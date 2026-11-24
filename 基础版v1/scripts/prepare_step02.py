"""只读完整检查原始数据，生成独立运行目录；不训练、不定位、不下载。"""
import argparse, bisect, hashlib, json, math, sys, time, traceback, uuid
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True)+'\n', encoding='utf-8')

def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()

def rootpath(value):
    path = Path(value)
    return (path if path.is_absolute() else ROOT/path).resolve()

def inside(root, relative):
    path = (root/relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError('Missing or escaping path: '+str(path))
    return path

def decode(path):
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError('Cannot decode '+str(path))
    return image

def jsonl(path, rows):
    path.write_text(''.join(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n' for row in rows),encoding='utf-8')

def phash(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    low = cv2.dct(cv2.resize(gray,(32,32)).astype(np.float32))[:8,:8].flatten()[1:]
    return sum(int(v > np.median(low)) << i for i,v in enumerate(low))

def photos(cfg, out, run):
    root = rootpath(cfg['bsds_root'])
    rows, rejected, hashes = [], [], []
    for path in sorted(root.glob('*/*.jpg')):
        try:
            image = decode(path)
            if image.ndim!=3 or image.shape[2]!=3 or image.dtype!=np.uint8:
                raise ValueError('Expected three-channel uint8')
            rows.append(dict(sample_id='BSDS300_'+path.stem, dataset='BSDS300', version='official-images-tgz', official_split=path.parent.name, relative_path=path.relative_to(root).as_posix(), data_root_key='bsds_root', width=image.shape[1], height=image.shape[0], channels=3, sha256=sha(path), pixel_sha256=hashlib.sha256(image.tobytes()).hexdigest()))
            hashes.append(phash(image))
        except Exception as error:
            rejected.append(dict(path=str(path),reason=str(error)))
    dump(run/'image_rejections.json',rejected)
    if len(rows)!=300 or rejected:
        raise ValueError('BSDS300 expected 300 valid color images; inspect rejections')
    parents=list(range(len(rows)))
    def find(i):
        while parents[i]!=i:
            i=parents[i]
        return i
    near=[]
    for i in range(len(rows)):
        for j in range(i):
            distance=(hashes[i]^hashes[j]).bit_count()
            same_prefix=int(rows[i]['sample_id'].split('_')[1])//1000==int(rows[j]['sample_id'].split('_')[1])//1000
            if same_prefix or rows[i]['sha256']==rows[j]['sha256'] or rows[i]['pixel_sha256']==rows[j]['pixel_sha256'] or distance<=cfg['phash_hamming_threshold']:
                parents[find(i)]=find(j)
                near.append(dict(a=rows[j]['sample_id'],b=rows[i]['sample_id'],phash_distance=distance,same_prefix=same_prefix))
    groups={}
    for i,row in enumerate(rows):
        groups.setdefault(find(i),[]).append(row)
    train_groups=[]
    quarantine=[]
    for group in groups.values():
        gid=min(row['sample_id'] for row in group)
        has_test=any(row['official_split']=='test' for row in group)
        for row in group:
            row['group']=gid
            row['split']='test' if row['official_split']=='test' else ('quarantine' if has_test else 'pending')
            if row['split']=='quarantine':
                quarantine.append(dict(**row,reason='Near/exact duplicate group touches official test'))
        if not has_test:
            train_groups.append(gid)
    train_groups.sort(key=lambda g:hashlib.sha256((str(cfg['seed'])+':'+g).encode()).hexdigest())
    validation=set(train_groups[:math.ceil(len(train_groups)*cfg['validation_group_fraction'])])
    for row in rows:
        if row['split']=='pending':
            row['split']='val' if row['group'] in validation else 'train'
    for field in ['sample_id','group','sha256','pixel_sha256']:
        sets=[{r[field] for r in rows if r['split']==s} for s in ['train','val','test']]
        if any(sets[i]&sets[j] for i in range(3) for j in range(i)):
            raise ValueError('Split leakage: '+field)
    for split in ['train','val','test']:
        jsonl(out/('images_'+split+'.jsonl'), sorted([r for r in rows if r['split']==split],key=lambda r:r['sample_id']))
    dump(out/'image_quarantine.json',quarantine)
    dump(out/'near_duplicate_candidates.json',near)
    selected=[r for r in rows if r['split']=='train'][:6]
    tiles=[]
    for row in selected:
        tile=cv2.resize(decode(root/row['relative_path']),(240,160))
        tile=cv2.copyMakeBorder(tile,30,0,0,0,cv2.BORDER_CONSTANT,value=(255,255,255))
        cv2.putText(tile,row['sample_id'],(5,20),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,0),1)
        tiles.append(tile)
    cv2.imencode('.png',np.vstack([np.hstack(tiles[:3]),np.hstack(tiles[3:])]))[1].tofile(str(run/'training_samples.png'))
    return dict(decoded=len(rows),excluded=len(quarantine),decode_rejections=len(rejected),counts={s:sum(r['split']==s for r in rows) for s in ['train','val','test']},group_links=len(near),phash_candidates=sum(r['phash_distance']<=cfg['phash_hamming_threshold'] for r in near),group_count=len(groups),cross_split_id_sha_pixel_group_intersections=0,scope='Conservative numeric image ID // 1000 grouping plus 63-bit DCT pHash; prefix is a heuristic, not official scene metadata; not a guarantee of semantic scene independence')

def index(root, filename, fields):
    rows=[]
    for line_number,line in enumerate((root/filename).read_text(encoding='utf-8-sig').splitlines(),1):
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        parts=line.split()
        if len(parts)!=fields:
            raise ValueError(f'{filename}:{line_number} wrong fields')
        stamp=Decimal(parts[0])
        if not stamp.is_finite() or stamp*1000000!=int(stamp*1000000):
            raise ValueError('Invalid timestamp precision')
        rows.append(dict(timestamp=parts[0],us=int(stamp*1000000),values=parts[1:],line_number=line_number))
    if not rows or any(a['us']>=b['us'] for a,b in zip(rows,rows[1:])):
        raise ValueError(filename+' empty, unordered or duplicate timestamps')
    return rows

def associate(left,right,threshold):
    times=[r['us'] for r in right]
    candidates=[]
    for i,row in enumerate(left):
        for j in range(bisect.bisect_left(times,row['us']-threshold),bisect.bisect_right(times,row['us']+threshold)):
            candidates.append((abs(row['us']-times[j]),row['us'],times[j],i,j))
    used_left,used_right=set(),set()
    pairs=[]
    for delta,lt,rt,i,j in sorted(candidates):
        if i not in used_left and j not in used_right:
            used_left.add(i)
            used_right.add(j)
            pairs.append((i,j,delta))
    return sorted(pairs), sorted(set(range(len(left)))-used_left), sorted(set(range(len(right)))-used_right)

def sequence(role, seq, cfg, out, run):
    root=rootpath(seq['root'])
    rgb=index(root,'rgb.txt',2)
    depth=index(root,'depth.txt',2)
    gt=index(root,'groundtruth.txt',8)
    frames={}
    for kind,records in [('rgb',rgb),('depth',depth)]:
        if len({r['values'][0] for r in records})!=len(records):
            raise ValueError('Repeated file reference')
        frames[kind]=[]
        for i,row in enumerate(records):
            relative=row['values'][0]
            path=inside(root,relative)
            image=decode(path)
            expected=(480,640,3) if kind=='rgb' else (480,640)
            dtype=np.uint8 if kind=='rgb' else np.uint16
            if image.shape!=expected or image.dtype!=dtype:
                raise ValueError(f'Unexpected {kind} shape/dtype: {path}')
            item=dict(index=i,timestamp=row['timestamp'],relative_path=relative,sha256=sha(path),shape=list(image.shape),dtype=str(image.dtype))
            if kind=='depth':
                item.update(zero_pixels=int((image==0).sum()),valid_pixels=int((image>0).sum()),min_positive_raw=int(image[image>0].min()) if (image>0).any() else None,max_raw=int(image.max()))
            frames[kind].append(item)
        jsonl(out/(role+'_'+kind+'_inventory.jsonl'),frames[kind])
    qerrors=[]
    for row in gt:
        pose=np.array(row['values'],dtype=float)
        if not np.isfinite(pose).all():
            raise ValueError('Nonfinite groundtruth')
        qerrors.append(abs(np.linalg.norm(pose[3:])-1))
    if max(qerrors)>0.001:
        raise ValueError('Invalid groundtruth quaternion norm')
    pairs,ur,ud=associate(rgb,depth,cfg['max_pair_delta_us'])
    paired=[]
    for i,j,delta in pairs:
        paired.append(dict(pair_id=f'{seq["name"]}_{i:06d}',sequence=seq['name'],data_root_key=role,rgb_index=i,depth_index=j,rgb_timestamp=rgb[i]['timestamp'],depth_timestamp=depth[j]['timestamp'],delta_us=delta,rgb_path=rgb[i]['values'][0],depth_path=depth[j]['values'][0],rgb_sha256=frames['rgb'][i]['sha256'],depth_sha256=frames['depth'][j]['sha256']))
    if not pairs or len({i for i,j,d in pairs})!=len(pairs) or len({j for i,j,d in pairs})!=len(pairs):
        raise ValueError('Invalid pairing')
    jsonl(out/(role+'_pairs.jsonl'),paired)
    dump(out/(role+'_unmatched.json'),dict(rgb=[rgb[i] for i in ur],depth=[depth[j] for j in ud]))
    left=[rgb[i] for i,j,d in pairs]
    gp,ug,unused_gt=associate(left,gt,cfg['max_pair_delta_us'])
    jsonl(out/(role+'_evaluation_index.jsonl'),[dict(pair_id=paired[i]['pair_id'],rgb_timestamp=left[i]['timestamp'],gt_timestamp=gt[j]['timestamp'],gt_line_number=gt[j]['line_number'],delta_us=d) for i,j,d in gp])
    dump(out/(role+'_evaluation_unmatched.json'),[paired[i]['pair_id'] for i in ug])
    samples=[]
    for k in sorted(set([0,len(pairs)//2,len(pairs)-1,max(range(len(pairs)),key=lambda k:frames['depth'][pairs[k][1]]['zero_pixels'])])):
        row=paired[k]
        color=decode(root/row['rgb_path'])
        raw=decode(root/row['depth_path'])
        meters=raw.astype(np.float32)/cfg['depth_units_per_m']
        scaled=np.round(np.clip(meters/cfg['preview_max_m'],0,1)*255).astype(np.uint8)
        preview=cv2.applyColorMap(scaled,cv2.COLORMAP_TURBO)
        preview[raw==0]=0
        y,x=np.argwhere(raw>0)[len(np.argwhere(raw>0))//2]
        sample=dict(**row,u=int(x),v=int(y),raw_value=int(raw[y,x]),meters=float(raw[y,x])/cfg['depth_units_per_m'],zero_pixels=int((raw==0).sum()),preview_range_m=[0,cfg['preview_max_m']],preview_file=f'{role}_pair_{k}.png')
        samples.append(sample)
        canvas=np.full((550,1280,3),255,np.uint8)
        canvas[:480,:640]=color
        canvas[:480,640:]=preview
        cv2.putText(canvas,f'RGB {row["rgb_timestamp"]} | DEPTH {row["depth_timestamp"]} dt={row["delta_us"]} us',(8,503),cv2.FONT_HERSHEY_SIMPLEX,.58,(0,0,0),1)
        cv2.putText(canvas,f'Depth 0-{cfg["preview_max_m"]} m; black=missing; values > max clipped for display only',(8,529),cv2.FONT_HERSHEY_SIMPLEX,.56,(0,0,0),1)
        bar=cv2.applyColorMap(np.arange(256,dtype=np.uint8).reshape(1,256),cv2.COLORMAP_TURBO)
        canvas[535:550,640:1280]=cv2.resize(bar,(640,15))
        cv2.imencode('.png',canvas)[1].tofile(str(run/sample['preview_file']))
    dump(out/(role+'_samples.json'),samples)
    zero=sum(r['zero_pixels'] for r in frames['depth'])
    return dict(sequence=seq['name'],rgb=len(rgb),depth=len(depth),groundtruth=len(gt),pairs=len(pairs),unmatched_rgb=len(ur),unmatched_depth=len(ud),pair_delta_us=dict(min=min(d for i,j,d in pairs),max=max(d for i,j,d in pairs),mean=float(np.mean([d for i,j,d in pairs]))),depth_zero_pixels=zero,depth_total_pixels=len(depth)*480*640,depth_missing_fraction=zero/(len(depth)*480*640),all_zero_depth_frames=sum(r['valid_pixels']==0 for r in frames['depth']),gt_associated=len(gp),gt_unmatched=len(ug),gt_quaternion_max_norm_error=max(qerrors),rgb_range=[rgb[0]['timestamp'],rgb[-1]['timestamp']],gt_range=[gt[0]['timestamp'],gt[-1]['timestamp']],files_decoded=len(rgb)+len(depth),index_sha256={n:sha(root/n) for n in ['rgb.txt','depth.txt','groundtruth.txt']},bytes_on_disk=sum(p.stat().st_size for p in root.rglob('*') if p.is_file()))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/data_step02.json')
    parser.add_argument('--baseline',type=Path,default=ROOT/'data/manifests/step02')
    args=parser.parse_args()
    stamp=datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
    run=ROOT/'runs'/('F02_check_'+stamp+'_'+uuid.uuid4().hex[:6])
    out=run/'manifests'
    out.mkdir(parents=True)
    started=time.perf_counter()
    report=dict(command=[sys.executable,*sys.argv],cwd=str(Path.cwd()),run_dir=str(run),status='running')
    try:
        cfg=json.loads(args.config.read_text(encoding='utf-8-sig'))
        dump(run/'config.json',cfg)
        (run/'prepare_step02.py').write_bytes(Path(__file__).read_bytes())
        dump(out/'roots_and_cameras.json',dict(bsds_root=str(rootpath(cfg['bsds_root'])),sequences={k:dict(v,root=str(rootpath(v['root'])),width=640,height=480,depth_units_per_m=cfg['depth_units_per_m'],invalid_depth_raw=0,source=cfg['format_source']) for k,v in cfg['sequences'].items()}))
        report['images']=photos(cfg,out,run)
        report['sequences']={role:sequence(role,seq,cfg,out,run) for role,seq in cfg['sequences'].items()}
        if cfg['sequences']['dev']['name']==cfg['sequences']['test']['name']:
            raise ValueError('Development and test must be distinct')
        report['manifest_hashes']={p.name:sha(p) for p in sorted(out.iterdir())}
        if args.baseline.is_dir():
            expected={p.name:sha(p) for p in sorted(args.baseline.iterdir()) if p.is_file()}
            if expected!=report['manifest_hashes']:
                raise ValueError('Frozen baseline differs; do not overwrite it')
            report['baseline_comparison']='all files byte-identical'
        else:
            report['baseline_comparison']='not present; first generation only'
        report.update(status='passed',exit_code=0)
    except Exception:
        report.update(status='failed',exit_code=1,error=traceback.format_exc())
        (run/'failure.log').write_text(report['error'],encoding='utf-8')
    report['seconds']=time.perf_counter()-started
    dump(run/'result.json',report)
    (run/'console.log').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=True,indent=2),flush=True)
    return report['exit_code']

if __name__=='__main__':
    sys.exit(main())
