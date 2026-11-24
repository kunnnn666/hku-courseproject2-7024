"""Frozen manifest data, independent sampling, explicit NCHW layout."""
import json
import numpy as np
import torch
from .image_baseline import load_reference, add_gaussian, sha256

def prepare(array, device, layout):
    tensor=torch.from_numpy(np.ascontiguousarray(array)).to(device)
    assert tensor.dtype==torch.float32 and tensor.ndim==4 and tensor.shape[1]==3
    return tensor.contiguous(memory_format=torch.channels_last if layout=='channels_last' else torch.contiguous_format)

class PatchData:
    def __init__(self, root, config):
        self.config=config;self.log=[];self.cache={};self.hashes={}
        frozen=json.loads((root/'handoffs/step02/handoff.json').read_text(encoding='utf-8'))['manifests']
        self.rows={}
        for split in ('train','val'):
            p=root/config['data'][split]
            self.hashes[str(p)]=sha256(p.read_bytes());assert self.hashes[str(p)]==frozen[str(p)]
            self.rows[split]=[json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()]
        p=root/config['data']['roots'];self.hashes[str(p)]=sha256(p.read_bytes());assert self.hashes[str(p)]==frozen[str(p)]
        self.image_root=json.loads(p.read_text(encoding='utf-8'))['bsds_root']
        assert len(self.rows['train'])==102 and len(self.rows['val'])==26
        assert not ({r['group'] for r in self.rows['train']}&{r['group'] for r in self.rows['val']})
        self.train_rng=np.random.default_rng(config['seed'])
        self.val_rng=np.random.default_rng(config['seed']+1)
        self.val_positions={}
        for row in self.rows['val']:
            size=config['patch'];self.val_positions[row['sample_id']]=(int(self.val_rng.integers(row['width']-size+1)),int(self.val_rng.integers(row['height']-size+1)))

    def batch(self, split, step, indices=None, fixed=False):
        refs=[];noises=[];size=self.config['patch']
        if indices is None:indices=self.train_rng.integers(len(self.rows[split]),size=self.config['batch']).tolist()
        for slot,index in enumerate(indices):
            row=self.rows[split][index];sid=row['sample_id']
            if sid not in self.cache:self.cache[sid]=load_reference(self.image_root,row)
            image=self.cache[sid]
            if split=='val':x,y=self.val_positions[sid]
            elif fixed:x,y=0,0
            else:x,y=int(self.train_rng.integers(image.shape[1]-size+1)),int(self.train_rng.integers(image.shape[0]-size+1))
            reference=image[y:y+size,x:x+size].copy()
            key=f'{split}:{sid}:{x}:{y}:{0 if fixed or split=="val" else step}:{slot}'
            noisy,meta=add_gaussian(reference,key,self.config['sigma'],self.config['seed'])
            self.log.append(dict(sample_id=sid,split=split,stage=self.config['stage'],group=row['group'],x=x,y=y,step=step,key=key,
                                 path=str(__import__('pathlib').Path(self.image_root)/row['relative_path']),file_sha=row['sha256'],noise_sha=sha256(noisy.tobytes()),**meta))
            refs.append(reference.transpose(2,0,1));noises.append(noisy.transpose(2,0,1))
        return np.stack(refs),np.stack(noises)
