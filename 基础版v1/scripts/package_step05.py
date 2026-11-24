"""Generate teaching and handoff from completed F05 evidence."""
import sys,json,html,hashlib,ast
from pathlib import Path
from run_step05 import ROOT,load,save
DOC=ROOT/'docs/steps/step05';HAND=ROOT/'handoffs/step05'
DOC.mkdir(parents=True,exist_ok=True);HAND.mkdir(parents=True,exist_ok=True)
runs=sorted(ROOT.glob('runs/F05_*'));results=[(p,load(p/'result.json')) for p in runs if (p/'result.json').exists()]
long=next((p,r) for p,r in results if r.get('frames')==792)
evaluation=next((p,r) for p,r in results if r.get('estimate_run')==str(long[0]))
def write(p,t):p.write_text(t,encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
table='|运行|退出码|帧/关联|成功/失败|秒|\n|---|---:|---:|---|---:|\n'
for p,r in results:table+=f"|{p.name}{'（人工失败注入，非真实跟踪）' if r.get('experiment_kind') else ''}|{r['exit_code']}|{r.get('frames',r.get('associated','—'))}|{r.get('successes','—')}/{r.get('failures','—')}|{r.get('seconds','未计时')}|\n"
poses=load(long[0]/'trajectory.json');diags=load(long[0]/'diagnostics.json');segments=[]
for s in sorted({p['segment_id'] for p in poses}):
    group=[p for p in poses if p['segment_id']==s];segments.append(dict(segment_id=s,start=group[0]['pair_id'],end=group[-1]['pair_id'],frames=len(group)))
save(long[0]/'segments.json',segments)
save(long[0]/'failures.json',[d for d in diags if not d['valid']])
max_step=max(float(__import__('numpy').linalg.norm(d['T_target_source'] and __import__('numpy').array(d['T_target_source'])[:3,3])) for d in diags if d['valid'])
report=f'''# F-05 结果报告

2026-11-07。状态：F-05 待总控审核。用户亲手复跑：F-07 后集中进行。

## 结果与范围

已完成原始开发序列完整792帧、791次转换。成功{long[1]['successes']}，失败{long[1]['failures']}，共{long[1]['segments']}段。转换成功率的分母为791；每段起点只是局部初始化，不算一次成功转换。实际原始输入覆盖792/792。未用独立测试、去噪网络或旧Open3D。

{table}

完整评测：{json.dumps(evaluation[1],ensure_ascii=False,indent=2)}

上面的ATE为米，RPE平移为米、旋转为度。仅开发序列初步诊断，不是F-08正式结论。主段关联覆盖率以792为分母；参考缺失的帧不进入误差。评测不将不同段分别对齐再混合。段外误差未计。分段详情见完整运行segments.json，失败列表见failures.json。最大成功相邻变换平移模长{max_step:.6f}米（观察值，不是预设成功门限）。

## 冻结观测与深度

输入dev_pairs与roots_and_cameras按F-02交接哈希核验，每张RGB和深度按清单核验SHA、尺寸、类型。reads.json枚举实际读取图像，inputs.json保存实际行；未读取test图像。五帧F-04原图接口样例另由audit_step05逐项检查pair_id、时间、尺寸、RGB哈希、特征索引/坐标，直接用其mutual重算；与短段相应结果矩阵一致。0→30单独运行，不拼接进连续轨迹。

深度640×480 uint16，raw/5000得米，0无效，不重复校正，不去畸变。内参525,525,319.5,239.5。最近邻floor(uv+0.5)，先拒绝原浮点越界/非有限，再拒绝舍入后的越界；不把越界点钳制成合法点。范围[0.2,5.0]米为预设室内保守范围，远距离深度较不可靠，并非根据GT调参。反投影用原浮点uv，而非整数深度索引。两侧都有效才入几何。counts分别记录描述子候选、两侧有效深度和共同有效三维点。

手算合成例：像素(320,240)，raw=10000 → z=2m；X=(320−319.5)×2/525=0.0019047619m，Y同值。验证器实际返回同值。坐标x向右、y向下、z向前。

## 自实现几何与方向

列向量p_target=R p_source+t。刚体解先算质心，中心化，H=AᵀB，NumPy SVD得到U,S,Vᵀ；R=V diag(1,1,det(VUᵀ)) Uᵀ，t=mean(B)−R mean(A)。两侧中心化点的第二奇异值<1e-6米判退化，拒绝共线/重复；允许平面点。SVD数值分解属于NumPy能力。RANSAC抽3个不同索引，每对种子为SHA256('F05:'+source_id+'->'+target_id+':7024')前8字节大端整数，NumPy默认PCG64。200次，残差欧氏距离≤0.03m，最少max(12,ceil(0.25N))内点。优先内点多，再优先内点残差和小，完全平局保留先出现候选。两轮内点重估与重新阈值检查，最终内点不足输出失败。没有根据ATE调参。

T_world_target=T_world_source @ inverse(T_target_source)。t_target_source是点坐标变换的平移，不能直接当相机世界位移。每段起点I；失败目标开启新段I并标incoming_valid=false/failure_reason，不能把该初始化解释为跟踪成功。trajectory.json的valid只表示该段局部位姿矩阵存在，跟踪成功必须读incoming_valid和segment_start。断段不隐式接起来。

## 数值与事后评价

verify_step05验证恒等、已知旋转平移、R正交/det、70真内点+30离群点、共线/重复、空有效候选、边界/零深度、非零旋转三帧链。固定误差容差1e-12。评价器用纯坐标变换轨迹验证ATE近0，用2倍尺度验证SE(3)对齐不能吸收尺度误差，并确认断段不跨段算RPE。audit_step05比较两份32帧轨迹逐字节一致，诊断去掉seconds后数值精确一致。check_failure_step05另外在进程内替换第一次RANSAC返回为人工失败，实际执行累计控制流，得到segment_id=[0,1,1]，失败目标I、下一步正常成功；该运行明确标为synthetic_failure_injection_NOT_real_tracking，不计入真实成功率，不修改源数据或正式模块。

估计入口不导入评测模块、不读取评价索引或groundtruth；仅在轨迹落盘后独立evaluate_step05读取冻结dev_evaluation_index。GT按索引时间精确查找，无插值，原始字段timestamp tx ty tz qx qy qz qw。平移米，单位四元数转换旋转。以关联数量最多的连续段计算不缩放刚体对齐ATE；不足3点/共线时rigid拒绝，评测result报错而不造0误差。RPE只在同一主段，以1秒±0.02秒选最近目标时间，平局较早索引；E=inv(delta_GT) delta_est，平移范数和旋转角分别RMSE。原始轨迹保留，aligned数组只在metrics.json。

## 图、失败与限制

评测目录raw_xy.png为局部XY投影，aligned_xy.png为主段对齐后与参考XY投影；绘图y向下，点为起点，尺度来自实际坐标但不显示全部三维结构。tracking.png横向转换索引，绿色为内点数量，红色为失败。真实失败数量如上；若failures为空，不能为了练习虚构断点，可看合成断段评测。没有故意损坏源数据测试I/O异常；真实文件全部核验。读取目录搜索无适用AGENTS，无运行中的Python写任务。初始文件搜索命令因不存在AGENTS返回1，不是实验失败。

教师对ORB/SVD许可仍待确认。自己实现反投影、刚体流程、RANSAC、累计和评测，不声称发明算法或实现数值库。数据出处及许可沿用references/F02_数据与DnCNN说明.md（TUM/CC BY 4.0）。无需新增依赖。

## F-08接口

backproject(uv, uint16_depth, K, limits)返回点和有效mask；ransac(a,b,key,config)返回T_target_source/内点/状态。F-08在调用match_images之前提供原图/同份噪声/滤波/网络RGB float32，几何接收相同原深度。共享前端参数、内参、深度范围、RANSAC阈值/种子规则及分段评测规则。当前CLI只实现original开发序列，不假称已支持正式测试或网络版本。新增版本入口应记录像素指纹，不能按每种方法GT选参数。
'''
write(DOC/'结果报告.md',report)
py=sys.executable
command=lambda script,args='':f"& '{py}' '{ROOT/'scripts'/script}' {args}".rstrip()
tutorial=f'''# F-05 零基础复现操作指南

用户亲手复跑：F-07 后集中进行。以下为执行对话实测命令；只读旧数据与解释器，不下载，不安装，不训练。状态待总控审核。

## 1 打开项目与终端

目的：站在正确的工程。前置：E盘路径及旧.venv_step5仍在。
按Windows+E打开文件资源管理器，按Ctrl+L选择地址栏，粘贴 `{ROOT}`，按Enter。应看到scripts、src、runs。若打不开先检查盘符，不新建空工程代替。
在地址栏输入powershell按Enter。复制以下PowerShell命令，按Enter：

```powershell
Set-Location -LiteralPath '{ROOT}'
Get-Location
```

-LiteralPath按原样解释路径；Get-Location应显示本项目。通常立即返回，不保存文件。以下命令都在这个窗口执行，不粘贴到Python的>>>里。

## 2 先做合成验算

目的：先确认公式方向，避免把真实图像误差与代码错误混淆。复制并按Enter：

```powershell
{command('verify_step05.py')}
$LASTEXITCODE
```

&调用引号中的绝对解释器，后面为脚本。等待数秒，正常显示新的runs/F05_verify_随机号路径，退出0。资源管理器Ctrl+L粘贴显示路径，右键result.json→打开方式→记事本，可见70内点、深度手算、累计与尺度检查结果。若AssertionError，保留终端错误，不继续宣称通过，不添加-O（会关闭断言）。

## 3 快速跑32帧

目的：短时间获得真实几何轨迹。前置：第2步成功。复制：

```powershell
{command('run_step05.py','--frames 32')}
$LASTEXITCODE
```

--frames 32表示从冻结清单取前32行，默认--gap 1相邻。实测4.47/4.37秒。输出run_dir是新目录，exit_code=0、frames=32、successes=31、failures=0；每次新随机目录，不覆盖。trajectory.json是相机位姿，diagnostics.json为逐次匹配/深度/内点/失败，inputs.json是本次输入。关闭窗口后这些文件仍在。
若找不到Python检查旧环境路径；若哈希断言失败不要修改冻结清单，记录run_dir/result.json并交总控。缺文件/坏图同样查看error，不用预览图替换深度。

## 4 单独评测已保存短段

目的：轨迹生成完后才读取参考。下面命令已对历史短段实际运行，初学者可直接复制；若评测自己的新结果，把--run后面的文件夹换成第3步run_dir：

```powershell
{command('evaluate_step05.py','--run runs/F05_est_209f3e2ef6')}
$LASTEXITCODE
```

--run选择估计目录，相对路径从本项目根解释。等待数秒，正常输出新的F05_eval目录、退出0；历史短段ATE约0.01099米，RPE平移约0.02129米（5组一秒配对）。metrics.json保存对齐轨迹和误差，原始trajectory不被覆盖。评测短到不足三点或共线时会拒绝，不能把无指标写成0。

## 5 完整序列与证据复核

有时间再复制以下命令：

```powershell
{command('run_step05.py','--frames 792')}
```

完整实测{long[1]['seconds']:.2f}秒；每100帧显示进度，运行期间等待PS提示符回来。每次新目录。完整历史结果是 `{long[0]}`，对应评测 `{evaluation[0]}`。历史长段评测复跑命令：

```powershell
{command('evaluate_step05.py','--run '+str(long[0].relative_to(ROOT)))}
```

短段一致性/跨步接口审计（引用两次交付短段和F-04历史数据）：

```powershell
{command('audit_step05.py')}
```

审计成功输出新F05_audit目录，退出0。它不是自动寻找你最新两次的脚本；保存的compared字段明确历史目录。

## 6 看图与练习

资源管理器打开完整评测目录，双击raw_xy.png和aligned_xy.png，点为起点；图像横向对应x，纵向对应y（向下）。打开tracking.png，横轴为转换顺序，绿色高度是内点数，红色是失败。再用记事本打开完整估计目录failures.json和segments.json。若[]表示实测没有失败，不要虚构断点；解释若失败会怎样从I重新开始新段。
练习：从trajectory.json前两帧的T_world_camera读出平移列，与起点比较运动方向；矩阵最后一列前三个数为米，不能改用diagnostics里的t_target_source。再手算raw=10000、(u,v)=(320,240)的Z、X、Y，答案Z=2m，X=Y=0.0019047619m，对照合成result.json的depth_hand_example。

深度是每个像素向前的Z距离；旋转改变坐标轴方向，平移改变原点。内点是满足3厘米残差规则的候选，不保证每条物理对应正确。求逆把“源点变到目标系”变成“目标相机放回世界”。断段意味着无法连接，局部零点不代表瞬间回到真实起点。ATE比较整体位置，RPE比较固定时间间隔内的相对运动；不缩放对齐不会掩盖米制尺度错误。

## 7 关闭与恢复

看到PS提示符后点×关闭。下次从第1步打开，不用激活环境，绝对解释器路径已指定。教程HTML双击用浏览器离线读；它包含与本Markdown相同全文。配置configs/geometry_step05.json是JSON数据，不是PowerShell命令，本步不要求用户修改。
'''
write(DOC/'复现操作指南.md',tutorial);write(DOC/'复现操作指南.html','<!doctype html><meta charset="utf-8"><title>F05操作指南</title><style>body{margin:35px auto;max-width:1000px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:16px/1.8 system-ui}</style><pre>'+html.escape(tutorial)+'</pre>')
explain=['# F-05 代码逐行讲解\n\n列向量公式与NumPy的行存储点数组不同：transform用a@R.T+t等价于对每个列点Rp+t。先读报告的坐标与RANSAC定义。\n']
notes={'backproject':'把uv定位到原深度网格，先判边界，按5000换米，再用原浮点uv还原XYZ；mask必须在两侧同时成立。','rigid':'先中心化，检查第二奇异值防共线，再构造协方差与反射修正，最后用质心计算平移。','transform':'点数组每行一个XYZ，所以乘R转置等价于列向量左乘R。','ransac':'固定种子抽三个点；退化样本跳过；比较内点数和误差和，再重估并复检，失败返回明确reason。','evaluate':'只选择一个主连续段，位置刚体对齐不缩放，再算同段固定时间差RPE，不跨段。','quaternion_pose':'从TUM的qx,qy,qz,qw转换旋转矩阵，平移仍为米。','main':'估计入口只读开发观测和配置，逐帧反投影与鲁棒估计，求逆累计；失败重新初始化局部段并保留状态。'}
for name in ('src/rgbd_geometry.py','src/trajectory_metrics.py','scripts/run_step05.py','scripts/evaluate_step05.py','scripts/verify_step05.py','scripts/audit_step05.py','scripts/check_failure_step05.py'):
    source=(ROOT/name).read_text();nodes=[n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.FunctionDef)];explain.append('\n## '+name+'\n')
    for n,line in enumerate(source.splitlines(),1):
        if not line.strip():continue
        owners=[f for f in nodes if f.lineno<=n<=f.end_lineno];owner=min(owners,key=lambda f:f.end_lineno-f.lineno).name if owners else '入口'
        purpose=notes.get(owner,'按本行变量/路径组织输入、执行检查或保存结果；这是模块入口的顺序执行语句。')
        if 'assert ' in line:purpose='数值验收条件：不满足立即抛错，不把失败改成成功。不要使用Python -O。'
        if 'save(' in line:purpose='把此时真实计算值保存成JSON；下游读取这个文件，而非从图上猜数字。'
        if 'import ' in line:purpose='引入此行列出的标准库或基础库能力，未安装新依赖。'
        explain.append(f'\n**第{n}行**\n```python\n{line}\n```\n{purpose}\n')
write(DOC/'代码逐行讲解.md',''.join(explain))
summary=f'F-05 待总控审核。完整开发792帧，{long[1]["successes"]}/791转换成功，失败{long[1]["failures"]}，{long[1]["segments"]}段。主段ATE={evaluation[1]["ate_rmse_m"]:.6f}米，RPE平移={evaluation[1]["rpe_translation_rmse_m"]:.6f}米；仅开发诊断。合成与短段复跑通过，未读取正式测试，估计与GT评测分开。用户亲手复跑F-07后集中进行。重点核对：米制和变换方向；失败/覆盖及不缩放同段评测；输入隔离与ORB/SVD贡献边界。停在F-05。\n'
write(HAND/'返回总控摘要.md',summary+'\n教程：'+str(DOC)+'\n完整运行：'+str(long[0])+'\n评测：'+str(evaluation[0]))
backup=HAND/'state_before.md'
if not backup.exists():backup.write_bytes((ROOT/'项目状态与下一步.md').read_bytes())
write(ROOT/'项目状态与下一步.md','# 项目状态与下一步\n\n'+summary+'\nF-01至F-04已通过审核。F-06未开始。\n\n历史状态见handoffs/step05/state_before.md。\n')
files=list(DOC.glob('*'))+list(HAND.glob('*.md'))+[ROOT/'项目状态与下一步.md',Path(__file__)]
files += [ROOT/p for p in ('src/rgbd_geometry.py','src/trajectory_metrics.py','scripts/run_step05.py','scripts/evaluate_step05.py','scripts/verify_step05.py','scripts/audit_step05.py','scripts/check_failure_step05.py','configs/geometry_step05.json')]
for run in runs:files+=list(run.glob('*'))
save(HAND/'handoff.json',dict(step_id='F-05',status='待总控审核',project_root=str(ROOT),interpreter=sys.executable,run_dirs=[str(p) for p in runs],
 commands=[r for _,r in results],coordinate_convention='T_target_source; T_world_target=T_world_source@inverse(T_target_source); meters',
 segments=segments,config=str(ROOT/'configs/geometry_step05.json'),input_hashes=long[1]['hashes'],validation='合成/32帧字节一致/F04四组原图对应交叉核验',
 limitations=['仅开发原图','ORB/SVD教师许可未知','评测只主连续段','跨平台一致性未验证'],tutorial_paths=[str(p) for p in DOC.glob('*')],user_replay='F-07 后集中进行',hashes={str(p):sha(p) for p in files if p.is_file()}))
print(summary)
