"""Documentation derived from the repair evidence; no training."""
import json,html,hashlib,ast
from pathlib import Path
import numpy as np
import cv2
from train_step06 import ROOT,save
D=ROOT/'docs/steps/step06';H=ROOT/'handoffs/step06'
run_names=['F06_repair_81c436a3ad','F06_repair_7c45dc5ba7','F06_repair_ad7b5027ac','F06_checks_8226eaa7']
runs=[ROOT/'runs'/n for n in run_names]
load=lambda p:json.loads(p.read_text(encoding='utf-8'))
results=[load(p/'result.json') for p in runs];fit,bench,quick=results[:3]
assert len({r['initial_parameter_hash'] for r in results[:3]})==1
for p in runs[:3]:
    losses=load(p/'curve.json');assert len(losses)==load(p/'result.json')['config']['steps']
    im=np.full((480,850,3),255,np.uint8);top=max(losses)*1.05
    pts=np.array([[70+i*700/max(1,len(losses)-1),410-v/top*330] for i,v in enumerate(losses)],np.int32)
    cv2.polylines(im,[pts],False,(180,60,0),2)
    for j in range(6):
        y=410-j*66;cv2.putText(im,f'{top*j/5:.4f}',(4,y),0,.4,(0,0,0),1)
    cv2.putText(im,f'MSE vs update step: 1 to {len(losses)}',(60,35),0,.7,(0,0,0),1)
    cv2.imencode('.png',im)[1].tofile(str(p/'curve.png'))
    ref,noisy,pred=[np.load(p/n)[0].transpose(1,2,0) for n in ('reference.npy','input.npy','prediction.npy')]
    out=np.concatenate([ref,noisy,np.clip(noisy-pred,0,1)],axis=1)
    out=cv2.resize((out[...,::-1]*255+.5).astype(np.uint8),(768,256),interpolation=cv2.INTER_NEAREST)
    canvas=np.full((300,768,3),255,np.uint8);canvas[44:]=out
    cv2.putText(canvas,'reference | noisy sigma25 | restored (validation patch0)',(8,25),0,.55,(0,0,0),1)
    cv2.imencode('.png',canvas)[1].tofile(str(p/'samples.png'))
table='|阶段|更新次数|初始/最终固定块MSE|训练总秒|GPU计算秒|验证保存秒|流程总秒|\n|---|---:|---|---:|---:|---:|---:|\n'
for r in results[:3]:table+=f"|{r['config']['stage']}|{r['config']['steps']}|{r['initial_fixed_loss']:.8f}/{r['final_fixed_loss']:.8f}|{r['training_total_s']:.4f}|{r['gpu_compute_s']:.4f}|{r['validation_save_s']:.4f}|{r['end_to_end_s']:.4f}|\n"
budget=(bench['training_total_s']/100*10000+bench['validation_save_s']/2*20)*2
report=f'''# F-06 修复结果报告

状态：F-06 待总控复审。用户亲手复跑：F-07 后集中进行。历史runs/F06_7d3eec4c及失败记录保留；旧交接和脚本快照在handoffs/step06/before_repair。未开始F-07。

## 实测阶段

{table}

三个独立进程都新建模型、Adam、固定seed=7024；初始参数SHA相同，100步没有接续fit模型。quick恰好20步，没有额外200步。每次步数由curve.json长度及checkpoint.step核验。无更新预热；仅用no_grad做逐层形状与初始损失前向，这会预热CUDA/cuDNN。Adam状态首次创建发生在第1个计时更新内。

固定8张训练图的左上块200步损失下降，说明学习通路有效，不证明泛化。随机训练有放回抽样，100步800块，每步记录来源/组别/坐标/种子/噪声哈希。固定fit图块复用首次记录，不重新抽样。实际训练/验证图文件由F03 load_reference核验文件与解码指纹、尺寸、RGB转换。

## 数据与标签

最终训练102、验证26，清单及roots按F02交接SHA核验，训练验证组无交集。只读这两份图像清单，不加载测试或隔离图。沿用F02对测试/隔离边界的已审核证据，不重新扫描测试图片。data模块独立可复用。

先裁RGB float32 [0,1]的64×64，再调用F03 add_gaussian，sigma25，PCG64稳定种子；标签为已裁剪noisy−reference。ID含split、sample_id、xy、步数和batch槽位，固定块步数固定0。训练与验证NumPy RNG分别seed7024/7025，验证坐标在初始化时独立生成，26张各1块；验证噪声固定。验证插入三次与不插入的训练抽样逐元素一致，结果见F06_checks_8226eaa7。饱和合成黑白图，逐元素核验clip(reference+noise)及标签，原始数组保存saturation.npz。RGB三通道哨兵值验证NCHW转换不串色。

## 网络、布局与加载

```mermaid
flowchart LR
 A[RGB N×3×64×64] --> B[Conv3→32 + ReLU]
 B --> C[5组 Conv32→32 + ReLU]
 C --> D[Conv32→3 有符号残差]
 D --> E[noisy减残差，显示时clip]
```

7层3×3卷积padding1，隐藏32，无BN，共48003参数；shapes.json保存每层输出尺寸。PyTorch提供卷积、自动求导和Adam，项目实现网络组合与数据训练流程；轻量DnCNN思路来自F02引用，不称论文完整复现。MSE比较预测残差和实际残差，不裁剪标签；恢复图仅在PSNR和预览时裁剪。

逻辑NCHW float32，显式channels_last由同一个prepare函数决定。训练保存前与新进程重载stride均[12288,1,192,3]（一张验证块），逻辑shape=[1,3,64,64]。容差在配置中预先固定atol=1e-6、rtol=1e-5；100步与quick重载allclose通过，最大绝对误差0。加载不训练，只读本项目自己生成的受控last.pt，checkpoint含网络配置、阶段、步数、版本、清单/代码SHA。仅推理重载，不支持断点续训（无优化器/RNG恢复）。历史布局失配最大差0.0004418343及GBK失败保持记录，没有放宽容差。

## 计时显存口径

100步训练总耗时{bench['training_total_s']:.6f}秒，逐步开始先CUDA同步，包含取块、首次文件读取及校验、CPU加噪、传输、前后向和Adam；GPU event计算时间{bench['gpu_compute_s']:.6f}秒从前向前到optimizer.step后，包含有限梯度检查带来的同步等待，不是纯算子理论速度。逐步训练时段求和不含间隔验证/保存；后者单独{bench['validation_save_s']:.6f}秒。流程总时间{bench['end_to_end_s']:.6f}秒另含初始化、清单、无梯度前向、最终保存等，不含Python导入启动。

100步前常驻allocated={bench['resident_before_training']['allocated']}，reserved={bench['resident_before_training']['reserved']}字节；训练时段峰值allocated={bench['training_peak']['allocated']}，reserved={bench['training_peak']['reserved']}字节。每步开始reset，结束立即读峰值再进入验证/保存；取所有更新时段最大值，含Adam首次状态创建、参考初始参数副本、当前输入、梯度。reserved含先前前向及间隔验证留下的缓存；不能解释为只由训练新申请。未报告整流程峰值，未把阶段峰值称作GPU整卡占用。无AMP、无OOM；独立GPU桌面程序仍可能影响速度。

## 验证与选模

成绩全称“26张图的固定图块验证PSNR”，每图一个固定64块，RGB data_range=1，先逐图PSNR再均值，不是整图PSNR。100步最佳{bench['best_patch_psnr']:.6f}dB，fit最佳{fit['best_patch_psnr']:.6f}dB；不同阶段不作正式模型比较。validation.json保留每图分数及每次调用步数。eval/no_grad，参数逐元素未变，结束恢复原训练模式；采样隔离另经验证。每50步保存step_N.pt，验证分数严格更高才覆盖该运行best.pt，平局保留早者；最终last.pt独立。本轮best仅检查短测选模分支，不是F07正式选模。

## F07预算（未执行）

建议仍独立随机初始化，batch8、patch64、sigma25，10000更新步=80000抽样块；每500步固定26块验证和保存，按平均图块PSNR最高选模型，后续是否整图验证由总控明确。用100步实测速率加20次验证保存，再乘2波动余量，约{budget:.1f}秒，即{budget/3600:.4f} GPU小时；这是估算，另预留启动/最终保存，建议10分钟窗口、硬上限1小时。损失/梯度非有限、显存错误、数据哈希不符即停止并保留失败result；验证连续多次不改善可在F07确定早停，不在本步临时调参。

入口train(config)和CLI已覆盖新初始化、间隔验证、保存、选best；正式运行应由F07单独复制配置和明确产物标识，本步checkpoint始终short_test_only，不准作为正式成绩。现未承诺续训。

## 总控重点

1. 独立初始化、标签饱和与数据/验证随机数隔离证据。
2. 计时显存边界、Adam首次更新及26张固定图块成绩口径。
3. 统一布局重载容差、选模保存入口及短测/正式训练边界。
'''
(D/'结果报告.md').write_text(report,encoding='utf-8')
py='E:\\chatgpt\\CIVL7024\\rgbd_odometry_project\\.venv_step5\\Scripts\\python.exe'
def cmd(script,args=''):return f"& '{py}' scripts/{script} {args}".rstrip()
tutorial=f'''# F-06 修复后零基础复现指南

状态待总控复审；用户亲手复跑F-07后集中进行。只用已下载训练/验证图片，无需安装。

## 1 打开项目和PowerShell

目的：在正确工程运行，避免覆盖旧项目。按Windows+E打开文件资源管理器；按Ctrl+L选地址栏，输入 `{ROOT}`，按Enter。正常看到src、scripts、runs。找不到先查E盘，不新建空项目替代。地址栏输入powershell，按Enter，打开终端。
复制下面PowerShell命令，按Enter：
```powershell
Set-Location -LiteralPath '{ROOT}'
Get-Location
```
Set-Location切换目录，-LiteralPath按原样处理路径，Get-Location应显示本项目；通常立即完成，不产生输出文件。后面代码框均为PowerShell，不粘贴到Python的>>>。

## 2 快速20步

目的：用几秒验证训练和保存。前置：第1步路径正确、旧解释器及图片仍在。复制并按Enter：
```powershell
{cmd('train_step06.py','--stage quick --steps 20')}
$LASTEXITCODE
```
&调用绝对Python，--stage选择阶段，--steps指定实际更新次数。配置默认configs/denoise_step06.json。正常JSON exit_code=0、config.steps=20；最后0是进程退出码。实测流程4.35秒，Python导入另计，通常等十余秒。确实只有20次更新，没有隐藏200步。返回run_dir的新F06_repair目录，重复运行不会覆盖历史。若仍在计算先等PS提示符回来。
用资源管理器Ctrl+L粘贴输出run_dir，Enter。右键result.json→打开方式→记事本，查看阶段、初末损失与耗时；双击curve.png和samples.png查看交付历史示例。新训练入口直接生成JSON/NPY/PT，PNG由打包脚本为本次交付生成，不能误认为每次训练自动生成PNG。

## 3 实际完整短测两阶段

仍在同一PowerShell，分别复制执行，每条等退出后再下一条：
```powershell
{cmd('train_step06.py','--stage fit --steps 200')}
{cmd('train_step06.py','--stage benchmark --steps 100')}
```
fit固定8块学习200步，流程6.88秒；benchmark独立新模型随机100步，流程5.91秒，均另加进程启动时间。每次唯一目录；不要把fit最后参数拿去当benchmark起点。训练数据RGB浮点0～1；网络预测应减掉的残差，目标为实际裁剪噪声图减参考。loss.backward算梯度，Adam改变参数，下降只说明学习通路有效。

## 4 数值与随机数隔离检查

目的：验证标签和验证频率不改变训练输入。复制：
```powershell
{cmd('verify_step06.py')}
$LASTEXITCODE
```
正常打印新的F06_checks路径、退出0，等待约数秒至十余秒。打开其result.json，label_exact、rgb_nchw_exact、validation_frequency_does_not_change_training应为true。saturation.npz是合成数组证据，不是照片。

## 5 新进程重载

下列命令对交付100步模型已实测，直接复制即可；重载自己的新模型时，将最后引号里的目录改为对应run_dir：
```powershell
{cmd('reload_step06.py',repr(str(runs[1])))}
$LASTEXITCODE
```
路径实际粘贴时使用普通单引号包裹，Windows双反斜杠也可解析。脚本读取last.pt里网络配置，不重新训练。正常allclose=true、max_absolute_error=0、atol=0.000001、rtol=0.00001，stride=[12288,1,192,3]。耗时约数秒加导入，reload.json写回该目录。只加载本项目生成的受控checkpoint。不要通过放宽容差隐藏布局错误。

## 6 读图与练习

资源管理器打开 `{runs[1]}`。双击curve.png；横向为1～100更新步、纵向MSE，有波动不必每步下降。samples.png依次参考/噪声/恢复，显示范围相同。打开validation.json，找step50和100及各自26个分数；说明它们是图块成绩而非整图成绩。
练习：写出输入、残差标签、MSE、参数更新、重载各自作用；再在result.json找training_total_s与gpu_compute_s，解释数据读取/加噪/传输为何造成差异。最后比较reload.json与result.json中的stride，说明NCHW逻辑形状相同不代表内存布局自动相同。

## 7 报错和恢复

找不到Python：检查共享解释器路径，不升级或移动旧环境。Frozen哈希断言/文件缺失：保留新目录result.json的error，检查磁盘，不能编辑清单绕过。CUDA不可用/OOM：保留错误，关闭自己不需要的GPU程序后交总控，不能改用CPU冒充显存结果。重载失败：先核对目录确为修复版last.pt、配置与布局；旧checkpoint不是同一格式。
关闭终端前等PS提示符。下次从Windows+E第1步恢复，不必activate。HTML双击用浏览器离线读，内容与本MD一致。配置JSON应在编辑器打开，不直接粘贴到PowerShell执行。本步不要求手动改配置或开始F07。
'''
# Use literal Windows path instead of Python repr escapes in the command example.
tutorial=tutorial.replace(repr(str(runs[1])),"'"+str(runs[1])+"'")
(D/'复现操作指南.md').write_text(tutorial,encoding='utf-8');(D/'复现操作指南.html').write_text('<!doctype html><meta charset="utf-8"><title>F06操作指南</title><style>body{margin:35px auto;max-width:1000px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:16px/1.8 system-ui}</style><pre>'+html.escape(tutorial)+'</pre>',encoding='utf-8')
files=['src/denoiser.py','src/denoise_data.py','scripts/train_step06.py','scripts/reload_step06.py','scripts/verify_step06.py']
notes={'prepare':'把NCHW数组复制为明确连续输入，搬到指定设备后按配置设置内存格式；逻辑维度不变。','__init__':'加载并核验冻结清单，隔离训练/验证组，创建独立RNG与固定验证位置。','batch':'按阶段选原图与裁块位置，先裁后调用F03加噪，记录来源/指纹，再将HWC转CHW并堆叠。','validate':'保存原模式，eval和no_grad遍历26张固定块，逐图算PSNR并平均，确认参数不变后恢复模式。','train':'创建唯一目录及新模型/Adam，控制更新、验证和保存时序，将实际数值写入证据。','checkpoint':'只保存可推理状态与配置来源；未保存优化器/RNG，因此不能冒充续训。','fixed_loss':'只前向计算相同固定块损失，便于比较初末，不在这里更新参数。','model_for':'检查checkpoint网络配置确为本实现结构，再创建CUDA模型。','main':'解析命令行并调用入口；导入模块时不会自动开始训练。','forward':'按已定义卷积/ReLU顺序前向，末层不限制残差正负。'}
lines=['# F-06 核心代码逐行讲解\n\n先看报告的结构图。每个非空源码行如下；一行若含多条分号语句，从左至右执行。参数、路径和维度保留原样，便于与编辑器行号对照。\n']
for name in files:
    source=(ROOT/name).read_text(encoding='utf-8-sig');nodes=[n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.FunctionDef)];lines.append('\n## '+name+'\n')
    for i,line in enumerate(source.splitlines(),1):
        if not line.strip():continue
        owners=[n for n in nodes if n.lineno<=i<=n.end_lineno];owner=min(owners,key=lambda n:n.end_lineno-n.lineno).name if owners else ''
        note=notes.get(owner,'模块级导入/定义或入口开关，提供后续执行所需对象。')
        if 'assert ' in line:note+=' assert要求条件成立，否则停止，不能把失败包装为通过。'
        if 'zero_grad' in line:note+=' 依次清空旧梯度、前向算残差MSE、反向计算新梯度；下一行检查有限性后Adam更新。'
        if 'reset_peak' in line:note+=' 此处重置训练时段峰值并同步GPU，之后才开始该步总计时。'
        if 'peaks.append' in line:note+=' 在验证和保存之前读显存峰值，避免混称整流程峰值。'
        if 'target=n-r' in line:note+=' n已裁剪，因此标签含边缘饱和效果，不能改用原始未裁剪噪声。'
        if 'allclose' in line:note+=' 容差从配置预先读取；同时报告最大绝对差，加载过程中无优化步骤。'
        if 'transpose' in line:note+=' transpose只变换维度顺序；prepare随后明确物理布局，避免历史重载差异。'
        lines.append(f'\n**第{i}行**\n```python\n{line}\n```\n{note}\n')
(D/'代码逐行讲解.md').write_text(''.join(lines),encoding='utf-8')
summary=f'F-06 修复完成，待总控复审。独立新模型100步、固定拟合200步、快速20步均退出0；26张固定图块验证，采样与验证频率隔离检查通过。100步训练总2.168秒/GPU事件1.057秒，阶段显存峰值38,292,992/48,234,496字节；重载预设atol1e-6 rtol1e-5，最大差0。历史失败保留。用户亲手复跑F-07后集中进行。未启动F-07。重点核对独立初始化/标签与数据隔离、计时显存及图块口径、布局重载及短测选模边界。'
(H/'返回总控摘要.md').write_text(summary+'\n\n教程：'+str(D)+'\n100步结果：'+str(runs[1]),encoding='utf-8')
(ROOT/'项目状态与下一步.md').write_text('# 项目状态与下一步\n\n'+summary,encoding='utf-8')
paths=[ROOT/n for n in files]+[ROOT/'configs/denoise_step06.json',Path(__file__),ROOT/'项目状态与下一步.md']+list(D.glob('*'))+[H/'返回总控摘要.md']
for p in runs:paths+=list(p.glob('*'))
save(H/'handoff.json',dict(step_id='F-06',status='待总控复审',project_root=str(ROOT),interpreter=py,runs=results,checks=results[-1],reload=[load(p/'reload.json') for p in runs[1:3]],
 config=str(ROOT/'configs/denoise_step06.json'),budget=dict(steps=10000,validation_interval=500,batch=8,patch=64,estimated_seconds_with_2x_margin=budget,hard_limit_s=3600,new_initialization=True),
 user_replay='F-07 后集中进行',limitations=['固定图块PSNR非整图','仅推理重载不支持续训','阶段峰值不是全流程峰值','教师PyTorch许可仍待确认'],tutorial_paths=[str(p) for p in D.glob('*')],hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}))
print(summary)
