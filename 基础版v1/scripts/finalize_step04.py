"""Supplemental checks and a hashed F04 handoff from saved runs."""
import ast
import hashlib
import html
import json
import sys
from pathlib import Path
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.feature_frontend import screen
RUN=ROOT/'runs/F04_20261107_100012_1a22f6'
DOC=ROOT/'docs/steps/step04'
HAND=ROOT/'handoffs/step04'
def load(p):
    return json.loads(p.read_text(encoding='utf-8'))
def save(p,value):
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
zeros=np.zeros((2,32),np.uint8)
targets=np.vstack([zeros[0],np.full(32,255,np.uint8)])
ambiguous=screen(zeros,targets)
assert ambiguous['status']=='no_matches' and ambiguous['counts']['ratio_pass']==2 and ambiguous['counts']['mutual']==0
distance=cv2.norm(zeros[0],targets[1],cv2.NORM_HAMMING)
assert distance==256
save(RUN/'supplemental.json',dict(command=[sys.executable,*sys.argv],exit_code=0,duplicate_sources=ambiguous,opencv_hamming=distance,expected=256))
# Annotate every executable core source line, grouped by its enclosing function.
explanations={
'detect':'先验证RGB浮点格式，再统一量化转灰度。ORB由库生成关键点和32字节描述子；空输出显式返回空数组。按位置、尺度、方向、响应及描述子排序，使后续索引稳定。points和descriptors使用同一排列，不丢失对应关系。',
'screen':'输入两组N×32 uint8描述子。XOR得到不同位，BITS查表逐字节计数求和，uint16可容纳最大256。one_way找两个最近邻，stable排序在相同距离时保留目标索引顺序。只有严格比例通过的候选才进入forward/reverse，随后取相互确认的交集。计数保留失败原因，空列表仍是合法结果。',
'one_way':'逐个源描述子处理一整行到目标的距离；空目标跳过，第二邻居不存在用None，不拿0伪造距离。每源raw最多一条，严格小于比例阈值才accepted。',
'match_images':'分别调用同一参数的检测，再筛选；索引查回每张原图里的浮点像素，给每个阶段加坐标。返回全部点表便于核对索引，不在此处求几何。',
'draw':'新建横向画布；转BGR仅为OpenCV写图。只取阶段前limit条，绘制时给目标加左图宽度、两端加标题高度。标题分别给总数和画出数，坐标偏移不回写记录。',
'save':'将Python结构序列化为UTF-8 JSON，禁止NaN，方便其他步骤读取并做字节比较。',
'main':'创建唯一运行目录，计时并保存命令；只读配置与冻结输入并校验哈希。先缓存每帧各条件，再遍历固定帧对。异常写入result并返回退出1，成功返回0；完整记录文件来自实际计算。核验脚本的main独立比较两次结果，验证坐标、指纹和合成输入。'
}
parts=['# F-04 代码逐行讲解\n\n先读结果报告的接口约定。下面逐行展示新增计算、运行和数值核验代码；行号对应交付文件，解释紧随代码。包装文档脚本只整理已保存证据，不生成匹配算法结果。\n']
for name in ('src/feature_frontend.py','scripts/run_step04.py','scripts/verify_step04.py'):
    text=(ROOT/name).read_text(encoding='utf-8');tree=ast.parse(text)
    functions=[n for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))]
    parts.append('\n## '+name+'\n')
    for i,line in enumerate(text.splitlines(),1):
        if not line.strip():continue
        owners=[f for f in functions if f.lineno<=i<=f.end_lineno]
        owner=min(owners,key=lambda f:f.end_lineno-f.lineno).name if owners else 'module'
        s=line.strip()
        if s.startswith(('import ','from ')): detail='导入所需模块；导入本身不读取数据或运行实验。'
        elif s.startswith('def '):detail=explanations.get(owner,'定义此函数，调用时才执行函数体。')
        elif s.startswith('assert '):detail='要求此条件为真，否则产生AssertionError，当前核验失败并保存错误；不要使用python -O。'
        elif s.startswith('return '):detail='把本函数计算结果交给调用者；退出码由最外层sys.exit传给PowerShell。'
        elif s.startswith(('for ','if ','elif ','else:','try:','except ')):detail='按此条件或循环控制下一段缩进代码；空候选与错误分支不会虚构对应。'
        elif 'sha' in s or 'fingerprint' in s:detail='计算或核对内容指纹；哈希一致验证输入/数组复算，不能证明匹配语义正确。'
        elif 'save(' in s:detail='将当前真实计算结构写入本次唯一目录，保留机器可读证据。'
        elif 'cv2.' in s:detail='调用OpenCV提供的基础能力，参数在本行明确；这部分不是自研ORB。'
        elif 'distances[i]' in s:detail='逐字节XOR后查BITS表，沿32字节求和得到这一源描述子到全部目标的Hamming距离。'
        elif 'reverse_pairs' in s or 'mutual =' in s:detail='将反向通过候选换回源→目标顺序，再要求正向候选也在该集合里。'
        elif 'second' in s:detail='第二近邻用于判断歧义；不存在时为None，有两个同距离最近邻时严格比例不会通过。'
        elif 'rgb' in s or 'noise' in s or 'filtered' in s:detail='按RGB float32 [0,1]处理或保存该条件；噪声与滤波使用F-03接口，彩色预览不作为计算输入。'
        elif owner!='module':detail='这是 '+owner+' 函数的数据准备/字段构造语句；'+explanations.get(owner,'沿当前分支连接输入与输出。')
        else:detail='设置模块常量、入口或说明；ROOT从脚本定位项目，避免依赖启动目录。'
        parts.append(f'\n**第{i}行**\n```python\n{line}\n```\n{detail}\n')
parts.append('\n## 包装脚本\n\npackage_step04.py从matches/result生成对照CSV、报告、同全文HTML、教程和状态备份；finalize_step04.py补充重复源候选与OpenCV距离对照，再生成逐行教学与哈希交接。它们引用本次固定运行，不属于日后快速计算入口。用户只需run_step04.py两次和verify_step04.py一次。HTML使用转义后的完整MD正文，离线没有外部资源。\n')
(DOC/'代码逐行讲解.md').write_text(''.join(parts),encoding='utf-8')
md=(DOC/'复现操作指南.md').read_text(encoding='utf-8')
assert html.unescape((DOC/'复现操作指南.html').read_text(encoding='utf-8').split('<pre>',1)[1].split('</pre>',1)[0])==md
runs=[ROOT/'runs'/n for n in ('F04_20261107_095929_4d4069','F04_20261107_100003_e4a15a','F04_20261107_100012_1a22f6','F04_verify_3ecfbc21')]
paths=[ROOT/p for p in ('src/feature_frontend.py','src/image_baseline.py','scripts/run_step04.py','scripts/verify_step04.py','scripts/package_step04.py','scripts/finalize_step04.py','configs/frontend_step04.json','configs/baseline_step03.json','configs/interpreter.json','data/manifests/step02/dev_pairs.jsonl','data/manifests/step02/roots_and_cameras.json','README.md','项目状态与下一步.md')]
paths+=list(DOC.glob('*'))+[HAND/'返回总控摘要.md']
for run in runs:paths+=list(run.glob('*'))
handoff=dict(step_id='F-04',status='待总控审核',project_root=str(ROOT),interpreter=sys.executable,
 input_manifests=load(RUN/'result.json')['input_hashes'],config=str(ROOT/'configs/frontend_step04.json'),
 runs=[dict(path=str(p),result=load(p/'result.json')) for p in runs],
 outputs=dict(matches=str(RUN/'matches.json'),comparison=str(RUN/'comparison.csv'),supplemental=str(RUN/'supplemental.json')),
 verification=load(runs[-1]/'result.json'),validation_scope='五个开发RGB、四个固定帧对、原图及三档噪声/滤波，28条结果；未读深度/GT/独立测试',
 limitations=['ORB教师许可仍待确认','对应数量不是正确率或轨迹精度','不保证跨版本字节一致','未破坏数据注入缺文件/坏图','日后核验默认比较名称排序最后两次F04运行'],
 user_replay='F-07 后集中进行',tutorials=[str(p) for p in DOC.glob('*')],
 hashes={str(p):sha(p) for p in paths if p.is_file()},unfinished='等待总控审核；F-05及以后未执行')
save(HAND/'handoff.json',handoff)
print(json.dumps(dict(status='packaged',exit_code=0,hashed_files=len(handoff['hashes']),tutorial_html_exact=True),ensure_ascii=True))
