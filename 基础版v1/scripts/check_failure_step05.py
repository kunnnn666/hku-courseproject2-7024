"""Explicit fault injection into estimator control flow; never a real tracking result."""
import sys,uuid
import numpy as np
import run_step05 as runner
from run_step05 import ROOT,load,save
original=runner.ransac
calls=0
def injected(a,b,key,config):
    global calls
    calls+=1
    if calls==1:return dict(valid=False,reason='synthetic_injected_failure',candidates=len(a),degenerate_samples=0)
    return original(a,b,key,config)
before=set(ROOT.glob('runs/F05_est_*'))
runner.ransac=injected
command=[sys.executable,*sys.argv]
sys.argv=['scripts/check_failure_step05.py','--frames','3']
assert runner.main()==0
created=(set(ROOT.glob('runs/F05_est_*'))-before).pop()
poses=load(created/'trajectory.json')
assert [p['segment_id'] for p in poses]==[0,1,1]
assert poses[1]['incoming_valid'] is False and poses[1]['segment_start']
assert np.array_equal(poses[1]['T_world_camera'],np.eye(4))
assert poses[2]['incoming_valid'] is True
result=load(created/'result.json');result['experiment_kind']='synthetic_failure_injection_NOT_real_tracking';save(created/'result.json',result)
run=ROOT/'runs'/('F05_failurecheck_'+uuid.uuid4().hex[:8]);run.mkdir()
save(run/'result.json',dict(exit_code=0,command=command,injected_run=str(created),segment_ids=[0,1,1],failure_does_not_carry_pose=True))
print(run)
