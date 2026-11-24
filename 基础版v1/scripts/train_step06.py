import argparse,json,time,uuid,sys,traceback
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.denoise_data import PatchData,prepare
from src.denoiser import ResidualCNN
from src.image_baseline import sha256
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
def model_for(c):
    assert c['network']==dict(layers=7,hidden=32,channels=3,kernel=3,batch_norm=False)
    return ResidualCNN().cuda()
def validate(model,data,c):
    previous=model.training;model.eval();before=[p.detach().clone() for p in model.parameters()];scores=[]
    with torch.no_grad():
        for i in range(26):
            arrays=data.batch('val',0,[i]);r,n=[prepare(x,'cuda',c['layout']) for x in arrays]
            mse=float((((n-model(n)).clamp(0,1)-r)**2).mean());scores.append(-10*np.log10(mse))
    assert all(torch.equal(p,b) for p,b in zip(model.parameters(),before));model.train(previous)
    return dict(mean_patch_psnr=float(np.mean(scores)),per_image_psnr=scores,count=26,parameters_unchanged=True,mode_restored=model.training==previous)
def train(c):
    run=ROOT/'runs'/('F06_repair_'+uuid.uuid4().hex[:10]);run.mkdir();start=time.perf_counter();report=dict(command=[sys.executable,*sys.argv],config=c,run_dir=str(run))
    try:
        assert torch.cuda.is_available();torch.manual_seed(c['seed']);torch.cuda.manual_seed_all(c['seed']);torch.backends.cudnn.benchmark=False
        data=PatchData(ROOT,c);model=model_for(c);optimizer=torch.optim.Adam(model.parameters(),lr=c['lr'])
        code={str(p):sha256(p.read_bytes()) for p in [Path(__file__),ROOT/'src/denoiser.py',ROOT/'src/denoise_data.py',ROOT/'src/image_baseline.py']}
        initial=[p.detach().clone() for p in model.parameters()];initial_hash=sha256(b''.join(p.cpu().numpy().tobytes() for p in initial))
        fixed=data.batch('train',0,list(range(c['batch'])),True);shapes=[]
        with torch.no_grad():
            x=prepare(fixed[1],'cuda',c['layout'])
            for layer in model.layers:x=layer(x);shapes.append(dict(layer=str(layer),shape=list(x.shape)))
        def fixed_loss():
            with torch.no_grad():
                r,n=[prepare(a,'cuda',c['layout']) for a in fixed];return float(((model(n)-(n-r))**2).mean())
        first=fixed_loss();torch.cuda.synchronize();resident=dict(allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved())
        timings=[];curve=[];validations=[];peaks=[];overheads=[];best=-float('inf')
        def checkpoint(step,name):
            torch.save(dict(model=model.state_dict(),config=c,step=step,short_test_only=True,resume_supported=False,torch_version=str(torch.__version__),input_hashes=data.hashes,code_hashes=code),run/name)
        for step in range(1,c['steps']+1):
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();tick=time.perf_counter()
            arrays=fixed if c['stage']=='fit' else data.batch('train',step)
            r,n=[prepare(a,'cuda',c['layout']) for a in arrays];target=n-r
            torch.cuda.synchronize();begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);begin.record()
            optimizer.zero_grad(set_to_none=True);loss=((model(n)-target)**2).mean();loss.backward()
            assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters())
            optimizer.step();end.record();torch.cuda.synchronize()
            timings.append(dict(step=step,training_total_s=time.perf_counter()-tick,gpu_compute_ms=begin.elapsed_time(end)))
            peaks.append(dict(allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.max_memory_reserved()));curve.append(float(loss))
            overhead=time.perf_counter()
            if step%c['validation_interval']==0 or step==c['steps']:
                score=validate(model,data,c);score['step']=step;validations.append(score)
                if score['mean_patch_psnr']>best:best=score['mean_patch_psnr'];checkpoint(step,'best.pt')
            if step%c['save_interval']==0 or step==c['steps']:checkpoint(step,f'step_{step}.pt')
            overheads.append(time.perf_counter()-overhead)
        final=fixed_loss();change=max(float((a-b).abs().max()) for a,b in zip(model.parameters(),initial));assert change>0
        if c['stage']=='fit':assert final<first
        model.eval();rr,nn=data.batch('val',0,[0]);tensor=prepare(nn,'cuda',c['layout'])
        with torch.no_grad():output=model(tensor)
        np.save(run/'input.npy',nn);np.save(run/'reference.npy',rr);np.save(run/'prediction.npy',output.cpu().numpy());checkpoint(c['steps'],'last.pt')
        for name,value in [('samples',data.log),('timings',timings),('curve',curve),('validation',validations),('shapes',shapes)]:save(run/(name+'.json'),value)
        report.update(exit_code=0,initial_parameter_hash=initial_hash,initial_fixed_loss=first,final_fixed_loss=final,parameter_change=change,parameters=sum(p.numel() for p in model.parameters()),resident_before_training=resident,
                      training_peak={k:max(p[k] for p in peaks) for k in ('allocated','reserved')},training_total_s=sum(t['training_total_s'] for t in timings),gpu_compute_s=sum(t['gpu_compute_ms'] for t in timings)/1000,
                      validation_save_s=sum(overheads),input_stride=list(tensor.stride()),dtype=str(tensor.dtype),shape=list(tensor.shape),input_hashes=data.hashes,code_hashes=code,device=torch.cuda.get_device_name(),torch=str(torch.__version__),best_patch_psnr=best)
    except Exception:report.update(exit_code=1,error=traceback.format_exc())
    report['end_to_end_s']=time.perf_counter()-start;save(run/'result.json',report);print(json.dumps(report));return report['exit_code']
def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/denoise_step06.json');p.add_argument('--stage',choices=['fit','benchmark','quick']);p.add_argument('--steps',type=int)
    args=p.parse_args();c=json.loads((ROOT/args.config).read_text(encoding='utf-8'))
    if args.stage:c['stage']=args.stage
    if args.steps:c['steps']=args.steps
    return train(c)
if __name__=='__main__':sys.exit(main())
