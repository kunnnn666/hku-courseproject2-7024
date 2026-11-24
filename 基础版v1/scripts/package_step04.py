"""Build F04 teaching documents from actual saved evidence; no dataset changes."""
import csv
import hashlib
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/'runs/F04_20261107_100012_1a22f6'
VERIFY = ROOT/'runs/F04_verify_3ecfbc21'
PY = r'E:\chatgpt\CIVL7024\rgbd_odometry_project\.venv_step5\Scripts\python.exe'
DOC = ROOT/'docs/steps/step04'
HAND = ROOT/'handoffs/step04'
for folder in (DOC,HAND):
    folder.mkdir(parents=True,exist_ok=True)
def read(path):
    return json.loads(path.read_text(encoding='utf-8'))
def write(path,text):
    path.write_text(text,encoding='utf-8')
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
records = read(RUN/'matches.json')
result = read(RUN/'result.json')
verify = read(VERIFY/'result.json')
config = read(RUN/'config.json')
table = '|行对|条件|噪声 sigma|源/目标特征|raw|比例通过|双向通过|RGB 时间差/秒|\n|---|---|---:|---|---:|---:|---:|---:|\n'
flat = []
for r in records:
    row = dict(pair=str(r['pair']),method=r['method'],sigma=r['sigma'],source_features=r['source_features'],target_features=r['target_features'],raw=r['counts']['raw'],ratio_pass=r['counts']['ratio_pass'],mutual=r['counts']['mutual'],dt_seconds=r['rgb_dt_seconds'])
    flat.append(row)
    table += f"|{r['pair']}|{r['method']}|{r['sigma']}|{r['source_features']}/{r['target_features']}|{row['raw']}|{row['ratio_pass']}|{row['mutual']}|{row['dt_seconds']:.6f}|\n"
with (RUN/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
write(RUN/'comparison.md',table)
example = next(r for r in records if r['pair']==[0,30] and r['method']=='noisy' and r['sigma']==50)
exercise = json.dumps(example['raw'][0],ensure_ascii=False)
command = f"& '{PY}' '{ROOT / 'scripts/run_step04.py'}'"
checkcommand = f"& '{PY}' '{ROOT / 'scripts/verify_step04.py'}'"
report = f'''# F-04 结果报告

状态：F-04 待总控审核。用户亲手复跑：F-07 后集中进行。

连续图像中的同一物体纹理可能被 ORB 再次检测。为了只比较处理方法的影响，各方法使用同样的两张帧、同一份人工噪声及同样的前端参数。匹配数量不是正确率，也不是轨迹精度。

## 实际范围与输入

事先固定开发清单零基行号 [0,1]、[1,2]、[2,3]，另设较大间隔 [0,30]。五张实际 RGB 均核验冻结文件 SHA-256、解码为 480×640×3 uint8、源时间早于目标。没有读取深度图、GT 或独立测试文件。只解析 dev_pairs.jsonl 与 roots_and_cameras.json；后者含其他序列配置，但只取 dev 根路径。输入清单哈希见 result.json，具体文件哈希/时间/pair_id 见 frames.json。

默认三档噪声 10/25/50，base_seed=7024。直接复用 F-03 add_gaussian(rgb, pair_id, sigma, 7024)，稳定 ID 为每张帧自身的 pair_id，非两个帧的组合 ID。F-03 使用 SHA-256 派生 PCG64 种子，sigma/255，生成后裁剪至 [0,1]。同一帧缓存一次，各帧对共用。滤波直接读取该浮点噪声输入，7×7、核 sigma=1 像素、reflect101。输入 RGB float32 的 C 顺序字节 SHA 在 input_arrays.json；RGB/HWC/float32/范围及尺寸结合 frames.json 解读。PNG 只作预览。

## 前端与筛选规则

ORB 为 OpenCV 4.10.0 提供的基础能力，未声称自行实现。自己的贡献是 Hamming 距离计算、筛选、异常分支、实验和接口。教师 ORB 许可仍待确认。没有安装依赖或修改共享环境。NumPy 1.26.4。

F-03 rgb_to_uint8 先用 float64 乘255，加0.5取 floor，恰好半整数向上取整。然后 cv2.COLOR_RGB2GRAY。OpenCV 解码 BGR 在入口反转为 RGB。ORB 固定 WTA_K=2，普通 Hamming 是二进制不同位的数量（0～256）；不是字节欧氏距离。全部组 ORB 参数相同，详见 config.json：nfeatures=500、scaleFactor=1.2、nlevels=8、edgeThreshold=31、firstLevel=0、scoreType=0、patchSize=31、fastThreshold=20。未调参。

先按 x/y/octave/angle/response/描述子字节/原索引排列特征。每个源描述子寻找距离最小的两个目标；距离并列时 raw 取较小目标索引。raw 是每个有目标候选的源点的一条最近邻，不是全部两两组合；不足两个候选时记录 missing_second，不通过比例筛选。比例严格 d1 < 0.75*d2，两个0距离也拒绝。反向执行同一规则，再保留彼此确认的 pair。源索引顺序输出，最终两侧索引唯一。阶段通过率若计算，比例通过/raw 与 mutual/ratio_pass 的分母不能混用；分母0时无定义，本交付只给计数。

## 实际结果

{table}

原图每帧对只有一次结果，不能把它复制三次算独立样本。前三行对是相邻短段，0→30 单独解释。所有组达到500个特征上限，所以这里不能根据特征总数判断质量。噪声会造成不稳定纹理，模糊会抹去细节；重复的书本纹理、桌面文字也可能导致误配。

0→30 的 sigma50：原图61、加噪37、滤波51条双向对应。该片段匹配较少。sigma10 时滤波61低于加噪63，保留了没有改善的反例。这些都是描述子筛选结果，不叫几何内点。

已实际查看 0_30_original_0_mutual.png、0_30_noisy_50_mutual.png、0_30_filtered_50_mutual.png 和 0_30_noisy_50_raw.png：筛选前连线交叉明显，筛选后较集中，但没有用几何或真值判别哪条正确。每图显示阶段列表前40条（源索引顺序，不按好看程度挑选），标题区分 total 和 shown；加噪筛选后37条全部展示。预览横向640像素、纵向60像素偏移只用于绘图，不写入坐标。

## 核验与证据

最终运行 F04_20261107_100003_e4a15a / F04_20261107_100012_1a22f6，退出0，实际8.013/7.599秒。初次运行 F04_20261107_095929_4d4069 保留；随后增加对 F-03 默认配置相等的检查，并完成上述两次最终运行。没有实际失败被改写为成功。

独立检查 F04_verify_3ecfbc21，退出0，2.176秒：28条结果的索引/边界/阶段数量/双侧唯一性通过；五帧全部噪声与滤波哈希重算一致。两次 matches、frames、input_arrays、config JSON 文件逐字节相同；计时、运行目录、result 不纳入字节一致要求。合成精确距离、并列候选、缺第二邻居、不同尺寸空白图通过。合成平移期望(+9,+6)像素，182条对应，中位位移(9,6)，误差中位0，97.25%误差<2像素；阈值为至少20条、中位误差<1像素、80%以上<2像素。它不是对真实图像几何精度的验证。

补充检查见 supplemental.json（同脚本打包时产生）：OpenCV Hamming 独立对照、两个重复源候选被反向歧义拒绝、no_matches 分支。缺失文件/坏图的入口有明确异常和退出1，未破坏实际数据注入故障；真实数据均成功。跨平台/不同 OpenCV 版本逐字节一致未验证。没有开展轨迹、训练、正式测试；这些不属于 F-04。

## F-05 直接消费接口

读取本运行 matches.json 的 JSON 数组，每个元素一个帧对/条件。source/target 各含 pair_id、timestamp（TUM 秒字符串）、manifest_row、rgb_index、shape=[H,W,3]、path、sha256；pair 为开发清单零基行号。method=original/noisy/filtered；sigma 是8位尺度，完整噪声/滤波/ORB设置见同目录 config.json。source_input_hash/target_input_hash 指实际 RGB 浮点输入；noise_hash 指该帧对应噪声输入，原图为null。

只把 mutual 当待几何检验的候选。每条含 source_index/target_index（对应 source_points/target_points 零基索引）、source_uv/target_uv=[u,v] 浮点像素、distance/second_distance（整数 Hamming）。u向右、v向下、左上角原点，坐标来自未缩放原图，可与同帧原尺寸深度关联；F-04 不取深度。不同尺寸输入由模块分别处理，绝不静默重采样；实验入口则强制本开发序列480×640。

status=ok 有对应；no_descriptors 任一侧无描述子；insufficient_neighbors 任一侧不足2条；no_matches 两侧足够但筛选后空。后三种应交给 F-05 跳过/记录，不能伪造运动。输入缺失/坏图/哈希变动是运行失败，result.json 保留 traceback、exit_code=1。输出不含真值。

## 待总控核查

1. 开发帧、冻结清单与 F-03 共用输入证据，未接触正式测试。
2. 双向比例规则、合成方向、复跑一致性与 F-05 原尺寸坐标接口。
3. 图示的显示上限、低匹配/未改善片段，以及 ORB 库贡献边界；不能把对应数量当定位提升。
'''
write(DOC/'结果报告.md',report)
tutorial = f'''# F-04 零基础复现操作指南

状态：待总控审核。用户亲手复跑约定在 F-07 后集中进行。以下命令由执行对话实际运行；不需要下载、安装或训练。机器实测一次7.6～8.0秒，电脑忙时可以更慢。

## 1 打开工程

目的：确认站在本项目，而非旧项目。前置条件：E盘、旧解释器及开发RGB原文件仍在原位置。
按 Windows+E 打开文件资源管理器，按 Ctrl+L 点选地址栏，粘贴 `{ROOT}`，按 Enter。正常看到 src、scripts、runs、docs。若找不到，检查盘符，不要新建空文件夹代替项目。
在资源管理器地址栏输入 `powershell`，按 Enter，打开蓝色或黑色命令窗口。下文代码框是 PowerShell 命令，不是 Python 代码；一次复制一块后按 Enter。

```powershell
Set-Location -LiteralPath '{ROOT}'
Get-Location
```

Set-Location 切换目录；-LiteralPath 表示按原样解释路径；引号保护空格。Get-Location 显示路径，应为上述工程。此步通常立即完成，不产生文件。出错时先重新检查资源管理器路径。

## 2 一次快速复跑

目的：读取现有五帧，对固定四组帧对完整跑原图与三档噪声/滤波。前置：第1步路径正确。复制到 PowerShell 按 Enter：

```powershell
{command}
$LASTEXITCODE
```

& 是调用操作符，第一对引号里是绝对 Python 解释器，第二对是本项目脚本。无额外参数，配置来自 configs/frontend_step04.json。不要粘贴到 Python 的 >>> 窗口。正常最后 JSON 含 status=passed、exit_code=0、record_count=28、run_dir；最后一行0为进程退出码。等待约8秒；若尚未回到 PS 提示符先等待。每次自动新建 runs/F04_日期_时间_随机号，旧结果保留。

当前已验证结果目录为 `{RUN}`。打开资源管理器 Ctrl+L 输入此路径 Enter；双击 result.json 用记事本查看（或右键→打开方式→记事本）。自己的复跑请改用输出 run_dir，别误把历史成功当本次成功。文件包括 matches.json（对应）、frames.json（帧来源）、input_arrays.json（浮点输入指纹）、config.json、timings.json 和56张图。comparison.csv 是本次交付从保存结果导出的汇总，快速入口本身不重建这张额外表。

若找不到解释器：检查旧 .venv_step5 路径，勿重装或移动环境。若 Frozen manifest differs / RGB hash differs：停下并把新目录 result.json 交总控，不编辑清单来消除报错。若 Missing/outside RGB path：检查数据盘路径；坏图报 Expected 480x640x3。若中断，可重执行生成新目录，保留失败目录。

## 3 两次运行与数值核验

目的：不是靠肉眼说一样，而是比较文件与数组。前置：用同一配置连续执行第2步两次。然后复制：

```powershell
{checkcommand}
$LASTEXITCODE
```

此入口选 runs 下按名称排序最后两份 F04_20 开头目录；不要同时启动其他 F-04 运行，也不要在两次间改代码或配置。正常显示 status=passed、exit_code=0、records=28、compared_runs 两个绝对路径；耗时约2.2秒。核验结果保存到新的 runs/F04_verify_随机号/result.json。资源管理器按修改日期排序可找到最新文件夹。
它排除耗时，比较匹配、帧、配置、输入指纹文件字节；还重算噪声与滤波，测试空白图、距离、平局及向右9向下6像素的合成平移。不要给 Python 添加 -O，它会关闭 assert 检查。若 AssertionError，先看 compared_runs 是否确为同配置两次；不要删失败证据，重跑正确的两次后再次检查。

## 4 看筛选前后及练习

目的：理解“长得像”只是候选。资源管理器打开上述已验证目录，双击 `0_30_noisy_50_raw.png`；再双击 `0_30_noisy_50_mutual.png`，用左右方向键或切换照片窗口比较。标题 raw total=500 shown=40，mutual total=37 shown=37。raw是最近邻；mutual是双方都通过比例且互相确认。只画前40条，图上数字是该阶段列表的零基行号，不是特征索引。不同阶段相同数字不保证同一条对应。

再打开 original_0_mutual、filtered_50_mutual（同样0_30前缀）。前者61条，后者51条；噪声组37条。滤波看起来更平滑，但不能据此宣布定位更准。

练习：在 raw 图中找一条看起来可疑的交叉线，读它靠左端标出的数字。用记事本打开 matches.json，Ctrl+F 搜索 noisy，定位 pair=[0,30]、sigma=50 的记录，再在 raw 列表按从0开始的行序查 source_uv/target_uv。可先用第0条练习，实测记录为：

```json
{exercise}
```

这是 JSON 数据，不是 PowerShell 命令。两个 uv 坐标分别在自己的640×480图里；右图在拼图上多了640像素，标题多了60像素，但数据表没有这些偏移。练习答案需写出你选中连线的阶段行号、两个(u,v)及怀疑原因。书本重复纹理、噪声或遮挡都可能让不同物体点外观相似；仅凭连线无法断言错误，需要 F-05 的几何检验。使用第0条只是操作示例，不把它自动判为错配。

## 5 关闭与恢复

确认看到 PS 提示符后点击窗口右上角×关闭。文件已保存在 runs。下次从 Windows+E 和第1步重新打开即可，不需要激活环境。教程 MD 可用记事本读；双击同目录复现操作指南.html，用浏览器离线阅读同一全文，不连接网络。

## 参数与计算约定

配置文件是 JSON，不能在终端直接执行。本次无需修改。sigma=10/25/50 是噪声幅度，不是滤波核 sigma=1；滤波核大小7、边界reflect101。RGB是红绿蓝顺序，OpenCV读图是BGR，入口已转换。float32 [0,1]用于加噪和滤波；统一四舍五入到 uint8 后按 RGB2GRAY 送给ORB。ORB WTA_K=2配普通Hamming，比例0.75固定。重复纹理的最近两个距离接近时会被拒绝，双向确认仍不等于几何正确。
'''
write(DOC/'复现操作指南.md',tutorial)
write(DOC/'复现操作指南.html','<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>F-04 复现操作指南</title><style>body{max-width:1000px;margin:35px auto;padding:20px;background:#fafafa}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:16px/1.8 system-ui}</style><body><pre>'+html.escape(tutorial)+'</pre></body></html>')
summary = f'''F-04 已执行，待总控审核，停在本步。固定开发行对0→1、1→2、2→3及0→30，覆盖原图和sigma10/25/50加噪/滤波，共28条记录。两次最终运行退出0（8.013/7.599秒），四份关键JSON逐字节一致；数值核验退出0。合成平移(+9,+6)获得182条对应，中位位移正确；空白/邻居不足/并列分支已验证。0→30 sigma50对应数61/37/51，sigma10滤波61低于加噪63，未改善案例已保留。ORB为OpenCV能力，教师许可待确认；未运行几何、深度、GT、训练或正式测试。用户亲手复跑：F-07 后集中进行。
结果：{RUN}
教程：{DOC / '复现操作指南.md'}
请重点核对：输入隔离与共用噪声；筛选规则/坐标方向/复跑证据；可视化限额和贡献边界。详见结果报告和handoff.json。\n'''
write(HAND/'返回总控摘要.md',summary)
# Preserve prior control state before updating the project-facing status.
backup = HAND/'before_step04'
backup.mkdir(exist_ok=True)
for name in ('README.md','项目状态与下一步.md'):
    if not (backup/name).exists():
        (backup/name).write_bytes((ROOT/name).read_bytes())
write(ROOT/'项目状态与下一步.md','# 项目状态与下一步\n\n2026-11-07：**F-04 待总控审核**。F-01至F-03已审核通过。\n\n'+summary+'\n## 历史审核状态（执行前原文）\n\n'+(backup/'项目状态与下一步.md').read_text(encoding='utf-8'))
write(ROOT/'README.md',(backup/'README.md').read_text(encoding='utf-8')+f'\n\n## F-04 特征与匹配（待总控审核）\n\n```powershell\n{command}\n{checkcommand}\n```\n\n核验前先连续运行前一命令两次。教程：docs/steps/step04/复现操作指南.md；接口：结果报告.md。仅开发集，OpenCV ORB基础能力+自实现筛选，不是几何内点。\n')
print('Documents and comparison generated:',DOC)
