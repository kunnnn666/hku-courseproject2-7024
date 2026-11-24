"""独立核验两个运行的清单、配对、预览及时间关联边界。"""
import argparse, json, sys, time, uuid
from pathlib import Path
from decimal import Decimal
import cv2
import numpy as np
from prepare_step02 import ROOT, sha, dump, decode, associate, inside

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--first',type=Path,required=True)
    parser.add_argument('--second',type=Path,required=True)
    args=parser.parse_args()
    run=ROOT/'runs'/('F02_verify_'+uuid.uuid4().hex[:10])
    run.mkdir()
    started=time.perf_counter()
    result=dict(command=[sys.executable,*sys.argv],cwd=str(Path.cwd()))
    try:
        first=json.loads((args.first/'result.json').read_text(encoding='utf-8'))
        second=json.loads((args.second/'result.json').read_text(encoding='utf-8'))
        assert first['exit_code']==second['exit_code']==0
        assert first['manifest_hashes']==second['manifest_hashes']
        for name,value in first['manifest_hashes'].items():
            for folder in [args.first/'manifests',args.second/'manifests',ROOT/'data/manifests/step02']:
                assert sha(folder/name)==value
        roots=json.loads((args.first/'manifests/roots_and_cameras.json').read_text(encoding='utf-8'))
        sample_checks=[]
        all_tum_hashes=set()
        for role in ['dev','test']:
            folder=args.first/'manifests'
            pairs=[json.loads(line) for line in (folder/(role+'_pairs.jsonl')).read_text(encoding='utf-8').splitlines()]
            assert len({p['rgb_index'] for p in pairs})==len(pairs)==len({p['depth_index'] for p in pairs})
            assert all('tx' not in p and 'pose' not in p and 'gt_timestamp' not in p for p in pairs)
            assert all(abs(Decimal(p['rgb_timestamp'])-Decimal(p['depth_timestamp']))*1000000==p['delta_us']<=20000 for p in pairs)
            for p in pairs:
                all_tum_hashes.add(p['rgb_sha256'])
            counts=first['sequences'][role]
            assert counts['pairs']+counts['unmatched_rgb']==counts['rgb']
            assert counts['pairs']+counts['unmatched_depth']==counts['depth']
            sample=json.loads((folder/(role+'_samples.json')).read_text(encoding='utf-8'))[0]
            root=Path(roots['sequences'][role]['root'])
            raw=decode(root/sample['depth_path'])
            color=decode(root/sample['rgb_path'])
            preview=decode(args.first/sample['preview_file'])
            assert raw.dtype==np.uint16
            assert int(raw[sample['v'],sample['u']])==sample['raw_value']
            assert sample['raw_value']/5000==sample['meters']
            scaled=np.round(np.clip(raw.astype(np.float32)/5000/5,0,1)*255).astype(np.uint8)
            expected=cv2.applyColorMap(scaled,cv2.COLORMAP_TURBO)
            expected[raw==0]=0
            assert np.array_equal(preview[:480,:640],color)
            assert np.array_equal(preview[:480,640:],expected)
            sample_checks.append(dict(role=role,sample=sample,preview_exact_match=True))
        image_hashes=set()
        for split in ['train','val','test']:
            rows=[json.loads(l) for l in (args.first/'manifests'/('images_'+split+'.jsonl')).read_text(encoding='utf-8').splitlines()]
            image_hashes.update(r['sha256'] for r in rows)
        assert not image_hashes & all_tum_hashes
        left=[dict(us=x) for x in [0,20000,40001]]
        right=[dict(us=x) for x in [10000,30000]]
        paired,ul,ur=associate(left,right,10000)
        assert paired==[(0,0,10000),(1,1,10000)] and ul==[2] and ur==[]
        try:
            inside(ROOT,'../outside.txt')
        except ValueError:
            pass
        else:
            raise AssertionError('Path escape accepted')
        result.update(exit_code=0,status='passed',manifest_count=len(first['manifest_hashes']),all_manifest_hashes_equal=True,sample_checks=sample_checks,synthetic_ties_threshold_unmatched='passed',path_escape='rejected as expected',image_tum_sha_overlap=0)
    except Exception as error:
        result.update(exit_code=1,status='failed',error=repr(error))
    result['seconds']=time.perf_counter()-started
    dump(run/'result.json',result)
    print(json.dumps(dict(run_dir=str(run),**result),ensure_ascii=True,indent=2))
    return result['exit_code']

if __name__=='__main__':
    sys.exit(main())
