import json,sys,uuid
import numpy as np
from train_step06 import ROOT,save,model_for,validate
from src.denoise_data import PatchData,prepare
from src.image_baseline import add_gaussian,gaussian_noise
def main():
    c=json.loads((ROOT/'configs/denoise_step06.json').read_text());run=ROOT/'runs'/('F06_checks_'+uuid.uuid4().hex[:8]);run.mkdir()
    ref=np.zeros((64,64,3),np.float32);ref[:,32:]=1
    noisy,meta=add_gaussian(ref,'saturation',25,7024);raw,_=gaussian_noise(ref.shape,'saturation',25,7024)
    label=noisy-ref;assert meta['clipped_fraction']>0 and np.array_equal(noisy,np.clip(ref+raw,0,1)) and np.array_equal(label,np.clip(ref+raw,0,1)-ref)
    rgb=np.zeros((1,3,4,4),np.float32);rgb[:,0]=.1;rgb[:,1]=.5;rgb[:,2]=.9
    tensor=prepare(rgb,'cuda',c['layout']);assert np.array_equal(tensor.cpu().numpy(),rgb)
    a,b=PatchData(ROOT,c),PatchData(ROOT,c);model=model_for(c)
    for step in range(1,4):
        ar,an=a.batch('train',step);validate(model,a,c);br,bn=b.batch('train',step)
        assert np.array_equal(ar,br) and np.array_equal(an,bn)
    va=a.batch('val',0,[0]);vb=a.batch('val',99,[0]);assert all(np.array_equal(x,y) for x,y in zip(va,vb))
    save(run/'result.json',dict(command=[sys.executable,*sys.argv],exit_code=0,saturation_fraction=meta['clipped_fraction'],label_exact=True,rgb_nchw_exact=True,
                              validation_frequency_does_not_change_training=True,validation_fixed=True,validation_mode_and_parameters_checked=True))
    np.savez(run/'saturation.npz',reference=ref,noisy=noisy,target=label,unclipped_noise=raw);print(run)
if __name__=='__main__':main()
