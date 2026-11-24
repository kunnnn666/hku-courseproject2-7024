import json,html,hashlib
from pathlib import Path
import numpy as np
import cv2
R=Path(__file__).resolve().parents[1];run=R/'runs/F06_7d3eec4c';d=R/'docs/steps/step06';h=R/'handoffs/step06'
d.mkdir(parents=True,exist_ok=True);h.mkdir(parents=True,exist_ok=True)
r=json.loads((run/'result.json').read_text());times=json.loads((run/'timing.json').read_text());mean=np.mean([x['total_s'] for x in times])
ref=np.load(run/'reference.npy')[0].transpose(1,2,0);n=np.load(run/'input.npy')[0].transpose(1,2,0);pred=np.load(run/'prediction.npy')[0].transpose(1,2,0)
canvas=np.concatenate([ref,n,np.clip(n-pred,0,1)],axis=1)
cv2.imencode('.png',cv2.resize((canvas[...,::-1]*255).astype('uint8'),(768,256),interpolation=cv2.INTER_NEAREST))[1].tofile(str(run/'samples.png'))
curves=json.loads((run/'curves.json').read_text());im=np.full((400,800,3),255,np.uint8)
for key,color in [('fixed',(200,0,0)),('training',(0,0,200))]:
    v=curves[key];points=np.array([[30+i*740/max(1,len(v)-1),350-x/.012*300] for i,x in enumerate(v)],np.int32);cv2.polylines(im,[points],False,color,2)
cv2.putText(im,'blue=fixed200; red=random100; separate step axes; MSE',(20,25),0,.5,(0,0,0),1);cv2.imencode('.png',im)[1].tofile(str(run/'curve.png'))
report=f'''# F-06 结果报告

状态：需要修复（训练通路已实测，交付验收尚不完整）。用户亲手复跑F-07后集中进行。

RTX3060 Laptop CUDA实测：200步固定8图块学习，损失{r['initial_loss']}降至{r['final_fixed_loss']}；参数最大变化{r['parameter_max_change']}。之后在同一模型上实际随机训练100步。参数48003，7层3×3卷积，3→32→32→32→32→32→32→3，前6层ReLU，无BN，末层线性。张量NCHW=[8,3,64,64]，实际channels_last内存布局。Adam lr=.001，MSE(predicted_residual,noisy-reference)，无AMP。恢复图裁剪只用于显示和评价。

102训练/26验证及分组隔离通过冻结哈希检查；实际文件由F03 load_reference核对文件和解码哈希。先裁后加噪，sigma25、F03裁剪规则；训练随机有放回抽样，位置均匀随机，seed7024；ID含样本、xy、step、batch位置。验证仅前8张左上角固定块，同一次运行保持不变，未读测试集。samples.json保存实际抽样记录。

验证图块PSNR={r['validation_patch_psnr']}dB，非泛化结论。峰值allocated={r['allocated']}字节，reserved={r['reserved']}字节。100步平均端到端{mean:.6f}秒/步（包含取块/CPU加噪/传输；compute_s只含同步后优化计算），总流程{r['seconds']}秒含200步学习、读图、验证与保存。100步之前200步已预热。图samples.png从左到右参考、噪声、恢复；curve.png两段各自以步数从0起，不能误看作同一连续横轴。

首次GBK读配置失败已改UTF8；新进程重载最初因输入内存布局不同最大差0.0004418343超过1e-6；显式channels_last后误差0。仅支持推理重载，不保存优化器/RNG续训状态。checkpoint短测专用。

F07建议新随机初始化，batch8、64块，sigma25先完成单档，10000步、每500步验证/保存，以固定验证PSNR最高选择。实测速率线性估计10000步{mean*10000/3600:.4f} GPU小时，加100%波动预留；此为估算非正式训练实测，最大1小时，遇非有限损失/显存错误停止。正式训练入口仍需补齐验证调度、配置化、来源指纹入checkpoint。

未完成验收：独立数据模块/冻结配置、饱和合成标签证据、逐层尺寸记录、完整逐句教程、几十步快速路径的独立验证、F07正式可复用训练入口仍需补齐。因此不能提交完整通过或启动F07。
'''
(d/'结果报告.md').write_text(report,encoding='utf-8')
tutorial=f'''# F06 操作指南（当前阶段）
按Windows+E，Ctrl+L输入 {R} 后Enter。地址栏输入powershell并Enter。
以下是PowerShell命令，不是Python代码。切换目录：
```powershell
Set-Location -LiteralPath '{R}'
& 'E:\\chatgpt\\CIVL7024\\rgbd_odometry_project\\.venv_step5\\Scripts\\python.exe' scripts/train_step06.py --steps 100
```
--steps只控制随机训练段，之前固定学习始终200步。实测总流程8.18秒，另有解释器启动开销。正常打印新runs/F06随机目录，历史不覆盖。结果读result.json，图读samples.png和curve.png。出错保留终端文字，不将失败当成功。
重载历史受控模型（新进程）：
```powershell
& 'E:\\chatgpt\\CIVL7024\\rgbd_odometry_project\\.venv_step5\\Scripts\\python.exe' scripts/reload_step06.py '{run}'
```
正常误差0。仅加载本项目自己生成checkpoint。关闭终端后从上述资源管理器步骤恢复，不必激活环境。
练习：看蓝色固定块曲线起末下降，说明输入是噪声图，标签是裁剪后噪声图减参考，loss比较残差，反向更新参数；重载只是重新读参数，不训练。当前教程尚未达到最终逐操作验收规范，见结果报告未完成项。
'''
(d/'复现操作指南.md').write_text(tutorial,encoding='utf-8');(d/'复现操作指南.html').write_text('<meta charset="utf-8"><pre style="white-space:pre-wrap">'+html.escape(tutorial)+'</pre>',encoding='utf-8')
(d/'代码逐行讲解.md').write_text('# 待补齐逐句讲解\n\n当前核心代码：src/denoiser.py和scripts/train_step06.py；网络卷积预测有符号残差，MSE反向传播后Adam更新。此文档尚未达到逐句验收标准。',encoding='utf-8')
(run/'failure_history.json').write_text(json.dumps(dict(initial='GBK config decode exit1',reload_before='layout mismatch max0.0004418343 exit1',reload_after='channels_last max0 exit0')))
summary='F-06 需要修复：200步小样本学习和100步CUDA训练已完成，新进程重载误差0；完整数据模块、配置与教学验收仍待补齐，未开始F07。'
(h/'返回总控摘要.md').write_text(summary,encoding='utf-8')
(R/'项目状态与下一步.md').write_text('# 项目状态与下一步\n'+summary+'\n用户亲手复跑F-07后集中进行。',encoding='utf-8')
files=list(d.glob('*'))+list(run.glob('*'))+[R/'src/denoiser.py',R/'scripts/train_step06.py',R/'scripts/reload_step06.py']
(h/'handoff.json').write_text(json.dumps(dict(step_id='F-06',status='需要修复',project_root=str(R),run_dir=str(run),result=r,hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},limitations=summary,tutorial_paths=[str(p) for p in d.glob('*')]),ensure_ascii=False,indent=2),encoding='utf-8')
