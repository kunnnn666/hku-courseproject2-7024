"""下载官方原始资料；已有文件不覆盖，成功解压后写标记。"""
import hashlib, json, subprocess, sys, tarfile, time, uuid
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    'BSDS300-images.tgz': 'https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/segbench/BSDS300-images.tgz',
    'rgbd_dataset_freiburg3_structure_texture_far.tgz': 'https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_structure_texture_far.tgz',
}
PAGES = {
    'bsds_terms.html': 'https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/segbench/',
    'tum_home.html': 'https://cvg.cit.tum.de/data/datasets/rgbd-dataset',
    'tum_format.html': 'https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats',
    'tum_download.html': 'https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download',
    'dncnn_paper.html': 'https://arxiv.org/html/1608.03981v1',
    'dncnn_readme.md': 'https://raw.githubusercontent.com/cszn/DnCNN/master/README.md',
    'dncnn_commit.json': 'https://api.github.com/repos/cszn/DnCNN/commits/master',
}

def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()

def main():
    run = ROOT / 'runs' / ('F02_download_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6])
    run.mkdir(parents=True)
    records = []
    try:
        for name, url in {**PAGES, **SOURCES}.items():
            target = ROOT / ('data/downloads' if name in SOURCES else 'references/step02_sources') / name
            start = time.perf_counter()
            if target.exists():
                records.append(dict(path=str(target), url=url, action='reuse', sha256=sha(target)))
                continue
            partial = target.with_name(target.name + '.' + uuid.uuid4().hex[:6] + '.partial')
            cmd = ['curl.exe', '--fail', '--location', '--retry', '2', '--connect-timeout', '30', '--max-time', '1800', '--silent', '--show-error', '--dump-header', str(run/(name+'.headers')), '--output', str(partial), url]
            result = subprocess.run(cmd, capture_output=True, text=True)
            (run/(name+'.log')).write_text(result.stdout+result.stderr, encoding='utf-8')
            records.append(dict(command=cmd, exit_code=result.returncode, seconds=time.perf_counter()-start, url=url, path=str(target), partial=str(partial)))
            if result.returncode:
                raise RuntimeError('Download failed: '+name)
            partial.rename(target)
            records[-1].update(bytes=target.stat().st_size, sha256=sha(target))
            print('Downloaded', name, target.stat().st_size, flush=True)
        for name in SOURCES:
            destination = ROOT/'data/raw'/name.removesuffix('.tgz')
            marker = destination/'extraction_complete.json'
            if marker.exists():
                continue
            destination.mkdir(parents=True, exist_ok=True)
            if any(destination.iterdir()):
                raise RuntimeError('Incomplete extraction exists; preserve and inspect: '+str(destination))
            with tarfile.open(ROOT/'data/downloads'/name, 'r:gz') as archive:
                members = archive.getmembers()
                for member in members:
                    path = (destination/member.name).resolve()
                    if not path.is_relative_to(destination.resolve()) or not (member.isfile() or member.isdir()):
                        raise RuntimeError('Unsafe archive member: '+member.name)
                archive.extractall(destination, filter='data')
            marker.write_text(json.dumps(dict(archive_sha256=sha(ROOT/'data/downloads'/name), members=len(members))),encoding='utf-8')
        code = 0
    except Exception as error:
        records.append(dict(error=repr(error)))
        code = 1
    (run/'result.json').write_text(json.dumps(dict(command=[sys.executable,*sys.argv], cwd=str(Path.cwd()), exit_code=code, records=records),ensure_ascii=False,indent=2),encoding='utf-8')
    print('RUN',run,'EXIT',code,flush=True)
    return code

if __name__ == '__main__':
    sys.exit(main())
