"""F-01 合成输入环境短测；不属于正式去噪训练。"""
import argparse  # 读取命令行参数。
import hashlib  # 生成文件指纹。
import json  # 保存结构化证据。
import subprocess  # 启动独立进程并记录退出码。
import sys  # 获取当前解释器的绝对路径。
import time  # 测量实际耗时。
import traceback  # 保留失败堆栈。
import uuid  # 避免复跑目录重名。
from datetime import datetime, timezone, timedelta  # 明确北京时间。
from pathlib import Path  # 处理包含中文的路径。

ROOT = Path(__file__).resolve().parents[1]  # scripts 的父目录就是项目根目录。

def write_json(path, value):  # 将 Python 数据保存为可读 JSON。
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")  # 不转义中文。

def digest(path):  # 对一个文件计算 SHA-256。
    return hashlib.sha256(path.read_bytes()).hexdigest()  # 此处只读取小型证据文件。

def command(args, folder, name):  # 执行命令，保留 stdout、stderr 和退出码。
    started = time.perf_counter()  # 单调时钟用于计时。
    result = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace")  # 等待子进程结束。
    (folder / (name + '.log')).write_text(result.stdout + '\nSTDERR:\n' + result.stderr, encoding='utf-8')  # 失败也写日志。
    record = dict(args=args, cwd=str(Path.cwd()), exit_code=result.returncode, seconds=time.perf_counter()-started)  # 保存精确参数。
    with (folder / 'commands.jsonl').open('a', encoding='utf-8') as handle:  # 一条命令占一行。
        handle.write(json.dumps(record, ensure_ascii=False) + '\n')  # 追加，不覆盖历史命令。
    return result  # 调用者决定失败是否阻断。

def network(torch):  # 仅用于检查梯度和序列化的两层网络。
    return torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3, padding=1), torch.nn.ReLU(), torch.nn.Conv2d(4, 3, 3, padding=1))  # 输入输出均为三通道。

def main():  # 同一脚本可运行主检查或独立加载检查。
    parser = argparse.ArgumentParser(description=__doc__)  # 自动生成 --help。
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/environment_check.json')  # 默认项目配置。
    parser.add_argument('--reload', type=Path)  # 内部子进程模式，指定已有运行目录。
    args = parser.parse_args()  # 解析用户输入。
    if args.reload:  # 子进程只重读模型，不更新参数。
        import torch  # 重新导入证明不依赖父进程内存。
        folder = args.reload.resolve()  # 定位保存的输入、模型和参考输出。
        cfg = json.loads((folder / 'config.json').read_text(encoding='utf-8'))  # 使用运行快照。
        payload = torch.load(folder / 'smoke_model.pt', map_location='cuda:0', weights_only=True)  # 仅加载本脚本生成的张量字典。
        model = network(torch).to('cuda:0').eval()  # 创建相同结构，启用推理模式。
        model.load_state_dict(payload['state'])  # 恢复优化一步后的权重。
        with torch.no_grad():  # 推理不需要梯度。
            actual = model(payload['input'])  # 对保存的相同输入推理。
        difference = (actual - payload['expected']).abs().max().item()  # 最大绝对差。
        passed = torch.allclose(actual, payload['expected'], atol=cfg['atol'], rtol=cfg['rtol'])  # 同时考虑绝对和相对误差。
        write_json(folder / 'reload.json', dict(passed=passed, max_abs_diff=difference, atol=cfg['atol'], rtol=cfg['rtol'], device=str(actual.device), shape=list(actual.shape)))  # 独立加载证据。
        print(json.dumps(dict(reload_passed=passed, max_abs_diff=difference)))  # 父进程捕获真实输出。
        return 0 if passed else 1  # 不一致时必须失败。
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')  # 运行编号使用北京时间。
    folder = ROOT / 'runs' / ('F01_' + stamp + '_' + uuid.uuid4().hex[:8])  # 每次创建新目录。
    folder.mkdir(parents=True, exist_ok=False)  # 即使重名也拒绝覆盖。
    started = time.perf_counter()  # 记录主检查耗时。
    report = dict(status='running', interpreter=sys.executable, python=sys.version, run_dir=str(folder), command=[sys.executable, *sys.argv], cwd=str(Path.cwd()))  # 初始证据。
    try:  # 所有可捕获失败都会落盘。
        cfg = json.loads(args.config.read_text(encoding='utf-8-sig'))  # 兼容 Windows BOM。
        write_json(folder / 'config.json', cfg)  # 冻结实际使用参数。
        (folder / 'check_environment.py').write_bytes(Path(__file__).read_bytes())  # 保留本次源代码快照。
        hardware = "[Console]::OutputEncoding=[Text.Encoding]::UTF8; @{os=(Get-CimInstance Win32_OperatingSystem | Select-Object Caption,Version,TotalVisibleMemorySize,FreePhysicalMemory); cpu=(Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors); disk=(Get-PSDrive E | Select-Object Used,Free); tasks=@(Get-CimInstance Win32_Process | Where-Object {$_.Name -match 'python|pip|uv.exe'} | Select-Object ProcessId,Name,ExecutablePath)} | ConvertTo-Json -Depth 5"  # 不采集环境变量、凭证或进程命令行。
        hw = command(['powershell.exe', '-NoProfile', '-Command', hardware], folder, 'hardware')  # 查询硬件和相关进程。
        if hw.returncode != 0:  # 查询失败不冒充完成。
            raise RuntimeError('Hardware query failed')  # 保留日志并中止。
        write_json(folder / 'hardware.json', json.loads(hw.stdout))  # 结构化硬件信息。
        gpu = command(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv'], folder, 'gpu')  # 记录显卡显存和驱动。
        if gpu.returncode != 0:  # 本步要求明确 GPU 证据。
            raise RuntimeError('nvidia-smi failed')  # 不能只凭历史信息通过。
        freeze = command([sys.executable, '-m', 'pip', 'freeze', '--all'], folder, 'pip_freeze')  # 精确全环境清单。
        if freeze.returncode != 0:  # 清单不可缺失。
            raise RuntimeError('pip freeze failed')  # 进入失败报告。
        (folder / 'requirements-freeze.txt').write_text(freeze.stdout, encoding='utf-8')  # 保存可审计版本。
        checked = command([sys.executable, '-m', 'pip', 'check'], folder, 'pip_check')  # 检查声明依赖冲突。
        if checked.returncode != 0:  # 有冲突时不得宣布环境通过。
            raise RuntimeError('pip check failed')  # 原始输出在日志中。
        import numpy as np  # CPU 数组基础库。
        import cv2  # 图像编解码基础库。
        import torch  # 张量、自动求导与优化基础库。
        report['versions'] = dict(numpy=np.__version__, opencv=cv2.__version__, torch=torch.__version__, cuda_runtime=torch.version.cuda)  # CUDA runtime 不是本机完整 toolkit。
        matrix = np.array([[1., 2.], [3., 4.]])  # 已知的小型 CPU 输入。
        product = matrix @ matrix  # 矩阵乘法。
        np.testing.assert_array_equal(product, [[7., 10.], [15., 22.]])  # 对照手算结果。
        report['cpu'] = dict(shape=list(matrix.shape), result=product.tolist(), passed=True)  # 保存 CPU 结果。
        pixels = np.arange(16*16*3, dtype=np.uint8).reshape(16, 16, 3)  # 合成 BGR 图，不读取数据集。
        ok, encoded = cv2.imencode('.png', pixels)  # PNG 无损编码。
        if not ok:  # 明确编码状态。
            raise RuntimeError('PNG encode failed')  # 拒绝保存无效结果。
        encoded.tofile(str(folder / 'synthetic.png'))  # 兼容中文 Windows 路径。
        decoded = cv2.imdecode(np.fromfile(str(folder / 'synthetic.png'), dtype=np.uint8), cv2.IMREAD_COLOR)  # 从磁盘读取再解码。
        np.testing.assert_array_equal(decoded, pixels)  # 要求每个像素一致。
        report['opencv'] = dict(shape=list(pixels.shape), dtype=str(pixels.dtype), max_abs_diff=0, passed=True)  # 图像读写证据。
        if cfg['require_cuda'] and not torch.cuda.is_available():  # 本配置要求 GPU，不能悄悄改用 CPU。
            raise RuntimeError('CUDA required but unavailable')  # 明确缺失设备。
        torch.manual_seed(cfg['seed'])  # 固定随机数起点。
        torch.cuda.manual_seed_all(cfg['seed'])  # 固定 GPU 随机种子。
        torch.backends.cudnn.benchmark = False  # 小短测不搜索最快卷积算法。
        torch.backends.cudnn.deterministic = True  # 尽量固定卷积路径，不保证跨平台位级一致。
        device = torch.device('cuda:0')  # 使用第一张 GPU。
        tensor = torch.tensor([[1., 2.], [3., 4.]], device=device)  # GPU 小张量。
        torch.testing.assert_close(tensor @ tensor, torch.tensor([[7., 10.], [15., 22.]], device=device))  # 实际执行 CUDA 矩阵乘法。
        model = network(torch).to(device)  # 网络参数搬到 GPU。
        inputs = torch.rand(cfg['shape'], device=device)  # NCHW：2 张、3 通道、16×16。
        target = torch.zeros_like(inputs)  # 合成零目标，无物理单位，不是去噪数据。
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])  # 用 Adam 做一次优化。
        before = [p.detach().clone() for p in model.parameters()]  # 保存更新前参数。
        optimizer.zero_grad()  # 清空梯度。
        loss = torch.nn.functional.mse_loss(model(inputs), target)  # 前向得到均方误差。
        loss.backward()  # 反向计算各参数梯度。
        if not torch.isfinite(loss) or not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):  # 检查损失和梯度。
            raise RuntimeError('Nonfinite loss or gradient')  # 非有限值不允许通过。
        optimizer.step()  # 唯一一次参数更新。
        delta = max((p.detach()-old).abs().max().item() for p, old in zip(model.parameters(), before))  # 最大参数变化。
        if delta <= 0:  # 反向成功不等于优化成功。
            raise RuntimeError('No parameter update')  # 必须观察到变化。
        model.eval()  # 切换推理状态。
        with torch.no_grad():  # 禁止记录推理计算图。
            expected = model(inputs)  # 更新后参考输出。
        if not torch.isfinite(expected).all():  # 输出也必须有限。
            raise RuntimeError('Nonfinite inference')  # 保存前检查。
        torch.save(dict(state=model.state_dict(), input=inputs, expected=expected), folder / 'smoke_model.pt')  # 仅环境短测模型。
        torch.cuda.synchronize()  # 等待异步 GPU 工作完成。
        report['gpu'] = dict(name=torch.cuda.get_device_name(0), device=str(device), total_bytes=torch.cuda.get_device_properties(0).total_memory, shape=cfg['shape'], loss=loss.item(), parameter_max_abs_change=delta, steps=1, passed=True)  # 真实观测。
        child = command([sys.executable, str(Path(__file__).resolve()), '--reload', str(folder)], folder, 'reload_console')  # 新 Python 进程加载模型。
        if child.returncode != 0:  # 子进程失败会使整次检查失败。
            raise RuntimeError('Independent reload failed')  # 不忽略子进程退出码。
        report['reload'] = json.loads((folder / 'reload.json').read_text(encoding='utf-8'))  # 合并独立证据。
        report['status'] = 'passed'  # 所有测试完成才标记通过。
        report['exit_code'] = 0  # 主进程成功码。
    except Exception:  # 捕获依赖、文件、CUDA 等错误。
        report.update(status='failed', exit_code=1, error=traceback.format_exc())  # 保留真实失败。
        (folder / 'failure.log').write_text(report['error'], encoding='utf-8')  # 独立错误文件。
    report['seconds'] = time.perf_counter() - started  # 包含子进程启动与环境查询，非训练基准。
    report['hashes'] = {p.name: digest(p) for p in folder.iterdir() if p.is_file()}  # result 自身不做循环哈希。
    write_json(folder / 'result.json', report)  # 成功和失败都落盘。
    (folder / 'console.log').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')  # 主报告的控制台镜像。
    print(json.dumps(report, ensure_ascii=True, indent=2))  # ASCII 转义避免终端编码报错。
    return report['exit_code']  # 传回 shell。

if __name__ == '__main__':  # 直接运行脚本才执行入口。
    sys.exit(main())  # 让 PowerShell 的 LASTEXITCODE 反映结果。
