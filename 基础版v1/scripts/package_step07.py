"""Freeze an accepted run and derive its teaching documents from the logs."""
import argparse
import html
import json
from pathlib import Path
import sys
import uuid

import cv2
import numpy as np

from train_step07 import ROOT, fingerprint, write_json


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def plot(path, series, title, ylabel):
    canvas = np.full((530, 960, 3), 255, np.uint8)
    lo = min(min(y) for _, _, y in series)
    hi = max(max(y) for _, _, y in series)
    pad = max((hi - lo) * .06, 1e-9)
    lo, hi = lo - pad, hi + pad
    steps = max(max(x) for _, x, _ in series)
    colors = [(176, 71, 18), (46, 111, 193)]
    for tick in range(6):
        y = 420 - 66 * tick
        cv2.line(canvas, (106, y), (906, y), (226, 226, 226), 1)
        cv2.putText(canvas, f"{lo+(hi-lo)*tick/5:.5g}", (8, y+4), 0, .47, (0, 0, 0), 1)
        x = 106 + 160 * tick
        cv2.putText(canvas, str(round(steps*tick/5)), (x-12, 446), 0, .45, (0, 0, 0), 1)
    for index, (label, x, y) in enumerate(series):
        points = np.column_stack((106+800*np.asarray(x)/steps,
                                  420-330*(np.asarray(y)-lo)/(hi-lo))).astype(np.int32)
        color = colors[index]
        cv2.polylines(canvas, [points], False, color, 1, cv2.LINE_AA)
        if len(x) <= 20:
            for point in points:
                cv2.circle(canvas, tuple(point), 3, color, -1)
        cv2.putText(canvas, label, (106 + 300*index, 495), 0, .5, color, 1)
    cv2.putText(canvas, title, (20, 35), 0, .7, (0, 0, 0), 1)
    cv2.putText(canvas, ylabel, (20, 66), 0, .5, (0, 0, 0), 1)
    cv2.putText(canvas, "Optimizer update step", (385, 469), 0, .5, (0, 0, 0), 1)
    cv2.imencode(".png", canvas)[1].tofile(str(path))


def powershell(script, *arguments):
    tokens = [Path(sys.executable).as_posix(), (ROOT / "scripts" / script).as_posix(), *arguments]
    return "& " + " ".join("'" + token.replace("'", "''") + "'" for token in tokens)


def main(args):
    formal, quick = args.formal.resolve(), args.quick.resolve()
    evidence = [read(path / "result.json") for path in args.evidence]
    assert all(row["exit_code"] == 0 for row in evidence)
    report, teaching = read(formal / "result.json"), read(quick / "result.json")
    assert report["initial_parameter_sha256"] == teaching["initial_parameter_sha256"]
    assert report["status"] == "completed" and report["completed_steps"] == 10000
    audit = next(row for row in evidence if row.get("source") == str(formal))
    probes = [row for row in evidence if row.get("allclose") and Path(row["checkpoint"]).parent == formal]
    assert {row["role"] for row in probes} == {"best", "last"}
    demo = next(row for row in evidence if row.get("scope") and Path(row["checkpoint"]).parent == formal)
    validation = read_rows(formal / "validation.jsonl")
    update = read_rows(formal / "updates.jsonl")
    assert audit["steps"] == len(update) == 10000 and audit["train_unique_images"] == 102
    best = next(row for row in validation if row["step"] == audit["best"])
    last = validation[-1]
    output = ROOT / "runs" / f"F07_freeze_{uuid.uuid4().hex[:10]}"
    output.mkdir()
    plots = output / "figures"
    plots.mkdir()
    vsteps = [row["step"] for row in validation]
    plot(plots / "training_loss.png", [("Per-update loss", list(range(1, 10001)), [r["loss"] for r in update])],
         "F-07 training loss", "Residual MSE")
    plot(plots / "validation_psnr.png", [(label, vsteps, [row[key] for row in validation]) for label, key in
         (("Restored", "mean_restored_psnr_db"), ("Noisy input", "mean_noisy_psnr_db"))],
         "Fixed validation patches: 26 images", "Mean per-image PSNR (dB)")
    plot(plots / "validation_mse.png", [("Predicted residual", vsteps, [r["mean_residual_mse"] for r in validation])],
         "Fixed validation residual error", "Mean residual MSE")
    qloss = [row["loss"] for row in read_rows(quick / "updates.jsonl")]
    plot(plots / "teaching_loss.png", [("Per-update loss", list(range(1, 21)), qloss)],
         "Teaching run, 20 updates", "Residual MSE")
    demo_path = Path(demo["run_dir"])
    views = [cv2.imdecode(np.fromfile(demo_path / f"{name}.png", np.uint8), cv2.IMREAD_COLOR)
             for name in ("reference", "noisy", "restored")]
    height, width = views[0].shape[:2]
    comparison = np.full((height + 40, width * 3, 3), 255, np.uint8)
    comparison[40:] = np.concatenate(views, axis=1)
    for i, label in enumerate(("Reference", "Noisy sigma=25", "Restored by best")):
        cv2.putText(comparison, label, (width*i+10, 26), 0, .65, (0, 0, 0), 1)
    cv2.imencode(".png", comparison)[1].tofile(str(plots / "validation_image.png"))

    frozen = {role: dict(path=str(formal / f"{role}.pt"), sha256=fingerprint(formal / f"{role}.pt"),
                         step=audit[role]) for role in ("best", "last")}
    config = dict(path=str(formal / "config.json"), sha256=report["config_sha256"])
    assert all(row["checkpoint_sha256"] == frozen[row["role"]]["sha256"] for row in probes)
    assert demo["checkpoint_sha256"] == frozen["best"]["sha256"]
    freeze = output / "frozen_model.json"
    write_json(freeze, dict(phase="F-07", status="frozen_pending_review", run=str(formal),
                           checkpoints=frozen, config=config, data=report["input_hashes"],
                           code=report["code_hashes"], acceptance=[str(p) for p in args.evidence]))

    doc = ROOT / "docs/steps/step07"
    handoff_dir = ROOT / "handoffs/step07"
    doc.mkdir(parents=True, exist_ok=True)
    handoff_dir.mkdir(parents=True, exist_ok=True)
    result_table = "|步|恢复 PSNR (dB)|含噪 PSNR (dB)|残差 MSE|\n|---:|---:|---:|---:|\n"
    for row in validation:
        result_table += f"|{row['step']}|{row['mean_restored_psnr_db']:.5f}|{row['mean_noisy_psnr_db']:.5f}|{row['mean_residual_mse']:.8f}|\n"
    report_text = f"""# F-07 结果报告

状态为 F-07 待总控审核。用户亲手复跑安排在 F-07 后集中进行。

10000 次正式更新全部完成。best 在第 {audit['best']} 步，平均恢复 PSNR 为 {best['mean_restored_psnr_db']:.5f} dB。last 在第 10000 步，平均恢复 PSNR 为 {last['mean_restored_psnr_db']:.5f} dB。含噪基线为 {best['mean_noisy_psnr_db']:.5f} dB。每个数来自 26 张验证图各取一个固定图块，先逐图计算 PSNR，再求算术均值。整图样例另列。

|阶段|更新|验证与保存|进程内耗时（秒）|
|---|---:|---|---:|
|教学|20|10、20|{teaching['seconds']:.3f}|
|正式|10000|500 到 10000，每 500 步|{report['seconds']:.3f}|

## 数据、配置与独立初始化

沿用 F-02 最终清单，训练 102 张、验证 26 张，组别无交集。冻结清单和根配置按 SHA-256 核验。F-03 loader 核对原文件 SHA、解码 BGR 指纹、尺寸与通道，再转 RGB float32。正式日志覆盖 102 张训练图与 80000 个有放回随机裁块。读取范围限定在 BSDS300 训练和验证。

模型为七层 3×3 卷积，隐藏 32 通道，前六层 ReLU，无 BN，末层输出有符号三通道残差。PyTorch 提供卷积、自动求导与 Adam，项目实现网络组合、数据、训练、固定验证和选模。沿用 F-06 轻量 DnCNN 思路。batch 8，64×64，sigma25/255，先裁后加噪，再裁剪 noisy 到 [0,1]，以 noisy−reference 为 MSE 目标。逻辑 NCHW，物理 channels_last。学习率 0.001，无 AMP。为明确 FP32 运算，F-07 两入口同时关闭 cuDNN TF32。与 F-06 速度不能直接当成同设置对照。

两阶段各以 seed 7024 创建模型和空状态 Adam，初始参数 SHA 相同。训练与验证的 NumPy 采样器分别用 7024、7025。验证块与噪声在训练前生成一次，训练时只复用。每次验证 eval、no_grad，结束恢复原模式。没有预热更新，第一次 Adam 状态创建计入训练。

## 时间、显存与失败

正式运行 {report['started']} 开始，{report['ended']} 结束。设备 {report['device']}，PyTorch {report['torch']}，CUDA {report['cuda']}，NumPy {report['numpy']}，OpenCV {report['opencv']}。完整流程 {report['seconds']:.3f} 秒，更新时段合计 {report['update_s']:.3f} 秒（含裁块、校验/读取、传输、同步），GPU 事件合计 {report['gpu_event_s']:.3f} 秒（前向至 Adam 完成，含有限性检查所带来的流等待）。验证及保存共 {report['validation_save_s']:.3f} 秒。流程总时间还包含初始化、日志、检查等开销，Python 导入另计。

更新前 allocated/reserved 为 {report['resident_before_updates']['allocated']:,}/{report['resident_before_updates']['reserved']:,} 字节。更新阶段峰值为 {report['peak_update_allocated']:,}/{report['peak_update_reserved']:,} 字节。逐步重置并在验证前读取，包含第一次 Adam 状态与常驻验证张量，reserved 可含既有缓存。这是训练时段分配器统计。桌面 GPU 负载、文件缓存和步首开销会带来波动。训练步内耗时最小/中位/最大为 {min(r['seconds'] for r in update):.6f}/{np.median([r['seconds'] for r in update]):.6f}/{max(r['seconds'] for r in update):.6f} 秒。

初次教学入口缺少 Path 导入，退出 1，0 步。记录在 runs/F07_teaching_56e1b48fb9/result.json，修复后用新目录重跑通过。每步写 JSONL，间隔 checkpoint 原子替换，失败记录保留。训练在每次更新边界检查一小时上限，机器端完整运行远低于此值。异常重启须确认故障原因后从独立新目录重新初始化。

## 验证曲线与冻结

{result_table}

best 采用严格大于，平局保留较早步。审计重算逐图均值，核对所有 20 个验证及保存节点、10000 条有限损失、正参数变化和 best/last 对应节点的权重一致。冻结清单为 {freeze}。权重留在正式运行目录，由冻结 SHA 标识。若 score 回落，日志及曲线照实保留。

|角色|步数|新进程最大绝对误差|allclose|
|---|---:|---:|---|
"""
    for row in sorted(probes, key=lambda row: row["role"]):
        report_text += f"|{row['role']}|{row['step']}|{row['max_abs_error']}|{row['allclose']}|\n"
    report_text += f"""
best/last 各自保存同一固定验证块的输入与该权重的输出。重载前验 SHA、shape、有限性，固定 atol=1e-6、rtol=1e-5。实际 torch.float32，shape [1,3,64,64]，stride [12288,1,192,3]。权重与输入的哈希见各自重载 result.json。

## 图像与后续使用

整图演示采用固定首张验证图 {demo['sample_id']}，sigma25，输入 {demo['noisy_psnr_db']:.4f} dB，恢复 {demo['restored_psnr_db']:.4f} dB。演示路径 {demo_path}。这是一张整图样例，正式选模依据仍为上表固定图块。

"""
    for name in ("training_loss", "validation_psnr", "validation_mse", "validation_image"):
        report_text += f"![{name}]({(plots / (name + '.png')).as_posix()})\n\n"
    report_text += """后续可通过 infer_step07.load_model(checkpoint) 加载，再调用 restore(model, rgb, config["layout"])。调用方传已经加噪的 RGB HWC float32 [0,1]，返回同尺寸 RGB HWC float32 [0,1]。函数只做去噪，无需量化。后续对比的噪声由调用方统一生成。sigma10/50、独立测试与定位效果待后续实测。checkpoint 仅用于推理，教师对 PyTorch 接口许可仍待确认。
"""
    (doc / "结果报告.md").write_text(report_text, encoding="utf-8")

    card = f"""# F-07 模型说明卡

用途为人工 Gaussian sigma25 噪声的 RGB 残差去噪。权重由固定种子随机初始化并在 BSDS300 训练图块上学习。配置与文件哈希见冻结清单 {freeze}。

|角色|步数|固定图块验证 PSNR|SHA-256|
|---|---:|---:|---|
|best|{audit['best']}|{best['mean_restored_psnr_db']:.5f}|{frozen['best']['sha256']}|
|last|10000|{last['mean_restored_psnr_db']:.5f}|{frozen['last']['sha256']}|

输入为 RGB HWC float32 [0,1]。内部准备 NCHW channels_last CUDA 张量。网络输出有正负的噪声估计，restore 返回 clip(noisy−residual,0,1)。整图推理采用卷积 padding=1，无拼块。模型仅保存推理状态，重启训练须重新初始化。

调用时把项目 scripts 目录加入 Python 搜索路径，导入 infer_step07 中的 load_model 与 restore。传入冻结清单中的 best.pt 绝对路径。只处理项目受控权重，加载使用 weights_only=True。正式配置为 {config['path']}。

选模依赖 26 张固定验证图块。训练仅 sigma25，跨噪声强度、真实传感器噪声、图像泛化与定位收益均有待各自实验。训练框架为 PyTorch。项目自己定义轻量网络、训练逻辑和选模协议。
"""
    (doc / "模型说明卡.md").write_text(card, encoding="utf-8")
    tutorial = make_tutorial(formal, quick, teaching, report, demo, plots)
    (doc / "复现操作指南.md").write_text(tutorial, encoding="utf-8")
    (doc / "复现操作指南.html").write_text(
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>F-07 操作指南</title>'
        '<style>body{max-width:1000px;margin:2rem auto;padding:1rem}pre{white-space:pre-wrap;'
        'overflow-wrap:anywhere;font:16px/1.8 system-ui}</style><pre>'+html.escape(tutorial)+'</pre></html>',
        encoding="utf-8")
    core = [ROOT / "scripts" / name for name in ("train_step07.py", "infer_step07.py", "audit_step07.py")]
    write_walkthrough(doc / "代码逐行讲解.md", core)
    summary = (f"F-07 正式训练完成，待总控审核。10000 次独立初始化更新，20 个验证/保存节点。"
               f"best 第 {audit['best']} 步，{best['mean_restored_psnr_db']:.5f} dB，last 第 10000 步，"
               f"{last['mean_restored_psnr_db']:.5f} dB，均为 26 张固定图块的平均 PSNR。"
               "教学 20 步、正式 best 和 last 的新进程重载均通过。用户亲手复跑 F-07 后集中进行。")
    focus = ("1. 冻结数据与独立初始化，以及 10000 步真实记录。\n"
             "2. 20 个验证节点、best 与 last 权重、冻结哈希。\n"
             "3. 新进程重载、整图演示及教学产物隔离。")
    (handoff_dir / "返回总控摘要.md").write_text(
        f"{summary}\n\n{focus}\n\n正式运行 {formal}\n冻结清单 {freeze}\n教程 {doc / '复现操作指南.md'}\n",
        encoding="utf-8")
    state = ROOT / "项目状态与下一步.md"
    (output / "state_before.md").write_bytes(state.read_bytes())
    state.write_text("# 项目状态与下一步\n\n**F-07 待总控审核**\n\n"+summary+
                     f"\n\n冻结清单 {freeze}\n交接 {handoff_dir / 'handoff.json'}\n后续由总控审核安排。\n", encoding="utf-8")
    files = [*core, Path(__file__), freeze, state, *doc.glob("*"), *plots.glob("*.png"),
             handoff_dir / "返回总控摘要.md", *formal.glob("*"), *quick.glob("*")]
    for path in args.evidence:
        files.extend(path.glob("*"))
    failed = ROOT / "runs/F07_teaching_56e1b48fb9/result.json"
    files.append(failed)
    handoff = dict(step_id="F-07", status="待总控审核", project_root=str(ROOT), interpreter=sys.executable,
                   formal_run=report, teaching_run=teaching, verification=evidence, prior_failure=read(failed),
                   frozen_models=frozen, frozen_config=config, freeze_manifest=str(freeze),
                   validation_curve=str(formal / "validation.jsonl"), selection=best,
                   training_curve=str(formal / "updates.jsonl"), user_replay="F-07 后集中进行",
                   limitations=["26 fixed validation patches", "trained at sigma=25", "inference-only checkpoints",
                                "PyTorch course permission pending"], tutorials=[str(p) for p in doc.glob("*")],
                   hashes={str(p): fingerprint(p) for p in files if p.is_file()})
    write_json(handoff_dir / "handoff.json", handoff)
    print(f"freeze_manifest={freeze}")


def make_tutorial(formal, quick, teaching, report, demo, plots):
    quick_command = powershell("train_step07.py", "--mode", "teaching")
    quick_reload = powershell("infer_step07.py", "--run", quick.as_posix(), "--role", "last")
    demo_command = powershell("infer_step07.py", "--run", formal.as_posix(), "--role", "best", "--demo")
    formal_command = powershell("train_step07.py", "--mode", "formal")
    return f"""# F-07 亲手复现指南

用户亲手复跑安排在 F-07 后集中进行。按下面三个练习即可验收使用流程。需要 E 盘项目、已核验解释器、冻结数据与 CUDA。所有代码框都是 PowerShell 命令。

## 练习一 · 跑 20 次教学更新

1. 按 Windows+E 打开文件资源管理器。按 Ctrl+L 选中地址栏，粘贴 `{ROOT}`，按 Enter。应看到 scripts、configs 和 runs。若打不开，先检查 E 盘路径。
2. 仍在资源管理器地址栏输入 powershell，按 Enter 打开终端。复制下面两行，按 Enter。第一行进入项目根目录，第二行显示当前目录。

```powershell
Set-Location -LiteralPath '{ROOT}'
Get-Location
```

3. 复制下面的命令，按 Enter。& 启动绝对路径的 Python，第二个路径指定本项目的训练入口。--mode teaching 读取 configs/denoise_step07_quick.json，进行 20 次更新，每 10 步验证并保存。JSON 配置可用记事本打开阅读。

```powershell
{quick_command}
$LASTEXITCODE
```

实测进程内 {teaching['seconds']:.3f} 秒，另有 Python 和 CUDA 启动时间。耐心等待提示符回来。本次实测输出摘录如下，复跑的目录随机后缀会不同。

```text
run_dir={quick}
step=10 mean_patch_psnr=20.402660
step=20 mean_patch_psnr=20.413959
status=completed completed_steps=20
0
```

4. 用鼠标只选中 run_dir= 后的目录文字并复制。资源管理器 Ctrl+L 粘贴该路径，按 Enter。右键 result.json，选“打开方式”→“记事本”，查 completed_steps=20、exit_code=0。updates.jsonl 每行对应一次 Adam 更新，loss 是预测残差和标签的像素平方差均值。一次更新先算梯度，再用 Adam 改参数。validation.jsonl 有第 10、20 步分数。验证模式用固定图块打分，不更新参数。best 取分数最高的一次，last 取最后一次。历史短训曲线在 `{plots / 'teaching_loss.png'}`。

## 练习二 · 新终端重载你的教学 last

1. 留着刚才复制的教学目录。在项目根的资源管理器地址栏再次输入 powershell，按 Enter，打开第二个终端。
2. 复制下方实测命令，先将 --run 后单引号里的历史目录换成练习一的实际目录，再按 Enter。只替换路径文字，保留引号。--role last 选该目录的 last.pt，脚本用保存输入比较重载预测。若想先验证历史结果，可直接执行原命令。

```powershell
{quick_reload}
$LASTEXITCODE
```

等数秒至十余秒。实测关键字段为 step=20、allclose=True、max_abs_error=0.0、exit_code=0。新 F07_infer 目录中的 result.json 记录模型和输入哈希及 stride。复跑时成功以 allclose 和退出码为准，浮点误差只要满足预设容差即可。新进程只做推理。若 --run 找不到目录，核对传入的是 run_dir 后的路径，而不是整条输出或某个文件。

## 练习三 · 用正式 best 处理一张验证图

在第二个终端复制以下命令，按 Enter。--run 指向冻结权重所在正式运行，--role best 选最佳验证节点，--demo 切换到整图演示。等待数秒至十余秒。

```powershell
{demo_command}
$LASTEXITCODE
```

实测 sample_id={demo['sample_id']}，exit_code=0。程序打印新 run_dir。资源管理器 Ctrl+L 输入这个目录并按 Enter，双击 reference.png、noisy.png、restored.png，切换窗口对比原图、加噪与恢复效果。result.json 记录本图噪声和分数，NPY 保留 RGB float32 数值。示例恢复分数 {demo['restored_psnr_db']:.4f} dB，是一张整图演示。正式 best 的选择来自 26 张固定图块。练习时指出一个杂点减弱的区域，再解释为什么 best 可能早于 last。

## 可选 · 完整正式训练复跑

前置条件是 CUDA 可用且项目数据路径和哈希一致。复制下方实测命令并按 Enter。--mode formal 读取冻结正式配置，固定种子创建模型与 Adam，每 500 步报告一次验证结果。实测进程内 {report['seconds']:.3f} 秒，启动另计。每次生成新的 F07_formal 目录，已有冻结清单仍指向本次交付。

```powershell
{formal_command}
$LASTEXITCODE
```

当显示 status=completed completed_steps=10000 且退出码为 0 时，训练阶段完成。进程完成后用资源管理器查看 updates.jsonl、validation.jsonl 与 step_*.pt。完整重跑的 best 仍需要独立审计、重载后才能形成新的冻结记录。

## 常见错误和恢复

- Python 路径找不到时，先用资源管理器确认共享解释器路径。已有环境按审核配置只读使用。
- CUDA 不可用、OOM、文件哈希不符或退出码 1 时，打开该次 result.json 读 error。保留目录，排查 GPU 和源文件后用原命令新建运行。推理权重缺少 Adam 与采样恢复状态，重启会从固定种子随机参数开始。
- 中途关闭终端可能只留下部分 JSONL 和间隔 checkpoint。检查 completed 状态与步数。下次从 Windows+E 重新打开项目再进入 PowerShell，历史输出仍可查阅。
- 读离线 HTML 时，在 step07 文档目录双击复现操作指南.html，用浏览器打开。Markdown 与 HTML 提供同一正文。
"""


def write_walkthrough(path, files):
    notes = {
        "write_json": "用临时文件完整写入 JSON 后替换目标，避免读取半条摘要。",
        "fingerprint": "以文件原始字节计算 SHA-256，配对清单、配置、权重与探针。",
        "validation": "保存训练模式，在无梯度下遍历固定块，逐图计算分数后恢复模式。",
        "save_checkpoint": "记录参数、角色和来源。best 与 last 各自保存一次权重对应的探针输出。",
        "train": "显式选择配置，创建唯一目录并从随机参数执行指定更新预算。",
        "load_model": "限制本地路径，读取受控权重和结构配置，建立 CUDA 推理模型。",
        "restore": "把 RGB HWC 转 NCHW，减去网络预测残差后返回同尺寸 RGB HWC。",
        "reload_probe": "核验权重与探针指纹，再按固定数值容差比对新进程输出。",
        "demonstrate": "选择冻结验证第一张，统一加噪并保存整图推理的数组和 PNG。",
        "audit": "在独立进程重读日志与权重，校验计划步数、来源、打分和角色。",
        "main": "解析命令参数并保存本次调用的机器可读结果。"
    }
    fragments = {
        "import ": "引入本行列出的库或项目模块，训练由显式入口启动。",
        "model.eval()": "切换到推理模式，使评估语义与训练分离。",
        "no_grad()": "关闭梯度记录，固定样本评分无需反向传播。",
        "target = noisy - reference": "监督标签使用实际裁剪后的噪声输入减参考，包含饱和造成的差异。",
        "zero_grad": "清除上一步梯度，保证本步只按当前 batch 更新。",
        "loss =": "对预测残差与目标的差做平方均值。",
        "loss.backward()": "由 PyTorch 自动求导，写入各参数的梯度。",
        "optimizer.step()": "Adam 使用梯度和内部统计执行一次参数更新。",
        "reset_peak_memory_stats": "重置分配器峰值计数，随后在验证之前取峰值。",
        "begin.record()": "在 CUDA 流上记录计算计时起点。",
        "end.record()": "在更新操作排队后记录 CUDA 流终点。",
        "torch.cuda.synchronize()": "等待设备完成已提交操作，为计时和读取提供边界。",
        "mean_restored_psnr_db\"] > best": "只接受严格提高的平均 PSNR，平分时保留更早权重。",
        "allclose": "在有限且同形状的前提下使用预设绝对和相对容差，另记最大误差。",
        "assert ": "审核要求此条件成立，失败时记录异常并退出。",
        "read_bytes": "读取原始文件字节以保留或核验指纹。",
        "finally:": "无论该段是否抛错，都会执行恢复操作。",
    }
    import ast
    out = ["# F-07 核心代码逐行讲解\n\n每行编号与交付源码对应。多行表达式按相邻行一起读。\n"]
    for file in files:
        source = file.read_text(encoding="utf-8")
        functions = [node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.FunctionDef)]
        out.append(f"\n## {file.name}\n")
        for number, line in enumerate(source.splitlines(), 1):
            if not line.strip():
                continue
            owners = [fn for fn in functions if fn.lineno <= number <= fn.end_lineno]
            owner = min(owners, key=lambda fn: fn.end_lineno - fn.lineno).name if owners else ""
            reason = next((value for key, value in fragments.items() if key in line),
                          notes.get(owner, "这里定义模块入口、配置常量或导入所需符号。"))
            out.append(f"\n**第 {number} 行**\n\n```python\n{line}\n```\n{reason}\n")
    path.write_text("".join(out), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal", required=True, type=Path)
    parser.add_argument("--quick", required=True, type=Path)
    parser.add_argument("--evidence", required=True, nargs="+", type=Path)
    main(parser.parse_args())
