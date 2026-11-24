"""Frozen development frames, original/noisy/filtered comparisons."""
import json
from pathlib import Path
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone, timedelta
import cv2
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import image_baseline as b
from src.feature_frontend import match_images

def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")

def draw(path, source, target, record, stage, limit):
    height = max(source.shape[0], target.shape[0])
    width = source.shape[1]
    canvas = np.full((height+60, width+target.shape[1], 3), 255, np.uint8)
    canvas[60:60+source.shape[0], :width] = b.rgb_to_uint8(source)[..., ::-1]
    canvas[60:60+target.shape[0], width:] = b.rgb_to_uint8(target)[..., ::-1]
    shown = record[stage][:limit]
    for number, match in enumerate(shown):
        x,y = match["source_uv"]
        u,v = match["target_uv"]
        color = ((number*71)%220, (number*113)%220, (number*37)%220)
        a,c = (round(x),round(y)+60),(round(u)+width,round(v)+60)
        cv2.line(canvas,a,c,color,1,cv2.LINE_AA)
        cv2.putText(canvas,str(number),a,cv2.FONT_HERSHEY_SIMPLEX,.35,color,1)
    label = f'{record["pair"]} {record["method"]} sigma={record["sigma"]} {stage} total={len(record[stage])} shown={len(shown)}'
    cv2.putText(canvas,label,(8,22),cv2.FONT_HERSHEY_SIMPLEX,.53,(0,0,0),1)
    cv2.putText(canvas,"line labels = zero-based stage list row; source left -> target right",(8,46),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,0),1)
    cv2.imencode(".png",canvas)[1].tofile(str(path))

def main():
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")
    run = ROOT/"runs"/("F04_"+stamp+"_"+uuid.uuid4().hex[:6])
    run.mkdir()
    start = time.perf_counter()
    report = dict(command=[sys.executable,*sys.argv],cwd=str(Path.cwd()),run_dir=str(run))
    try:
        cv2.setNumThreads(1)
        cv2.setRNGSeed(7024)
        config_path = ROOT/"configs/frontend_step04.json"
        config = json.loads(config_path.read_text())
        previous = json.loads((ROOT/"configs/baseline_step03.json").read_text())
        if config["sigmas"] != previous["sigmas_8bit"] or config["base_seed"] != previous["base_seed"]:
            raise ValueError("Noise configuration differs from F-03")
        if config["filter"] != {"size": previous["filter"]["size"], "sigma_px": previous["filter"]["sigma_px"]}:
            raise ValueError("Filter configuration differs from F-03")
        frozen = json.loads((ROOT/"handoffs/step02/handoff.json").read_text(encoding="utf-8"))["manifests"]
        files = [ROOT/"data/manifests/step02"/n for n in ("dev_pairs.jsonl","roots_and_cameras.json")]
        for file in files:
            if b.sha256(file.read_bytes()) != frozen[str(file)]:
                raise ValueError("Frozen manifest differs: "+str(file))
        report["input_hashes"] = {str(f):b.sha256(f.read_bytes()) for f in files}
        root = Path(json.loads(files[1].read_text(encoding="utf-8"))["sequences"]["dev"]["root"]).resolve()
        rows = [json.loads(line) for line in files[0].read_text().splitlines()]
        frames, metadata = {}, {}
        for index in sorted({i for pair in config["pairs"] for i in pair}):
            row = rows[index]
            path = (root/row["rgb_path"]).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError("Missing/outside RGB path: "+str(path))
            data = path.read_bytes()
            if b.sha256(data) != row["rgb_sha256"]:
                raise ValueError("RGB hash differs")
            raw = cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_UNCHANGED)
            if raw is None or raw.dtype != np.uint8 or raw.shape != (480,640,3):
                raise ValueError("Expected 480x640x3 uint8 RGB image")
            rgb = raw[...,::-1].astype(np.float32)/np.float32(255)
            frames[index] = {("original",0):rgb}
            metadata[index] = dict(pair_id=row["pair_id"],timestamp=row["rgb_timestamp"],manifest_row=index,
                                   rgb_index=row["rgb_index"],shape=list(rgb.shape),path=str(path),sha256=row["rgb_sha256"])
            for sigma in config["sigmas"]:
                noisy, seed = b.add_gaussian(rgb,row["pair_id"],sigma,config["base_seed"])
                frames[index][("noisy",sigma)] = noisy
                frames[index][("filtered",sigma)] = b.gaussian_filter(noisy,config["filter"]["size"],config["filter"]["sigma_px"])
        save(run/"frames.json",metadata)
        records, timings, fingerprints = [], [], {}
        for index, variants in frames.items():
            fingerprints[index] = {f"{method}_{sigma}":b.array_sha(array) for (method,sigma),array in variants.items()}
        save(run/"input_arrays.json",fingerprints)
        for i,j in config["pairs"]:
            if float(metadata[i]["timestamp"]) >= float(metadata[j]["timestamp"]):
                raise ValueError("Source must precede target")
            for method,sigma in frames[i]:
                source,target = frames[i][method,sigma],frames[j][method,sigma]
                tick = time.perf_counter()
                record = match_images(source,target,config["orb"],config["ratio"])
                timings.append(dict(pair=[i,j],method=method,sigma=sigma,seconds=time.perf_counter()-tick))
                record.update(pair=[i,j],source=metadata[i],target=metadata[j],method=method,sigma=sigma,
                              rgb_dt_seconds=float(metadata[j]["timestamp"])-float(metadata[i]["timestamp"]),
                              source_input_hash=b.array_sha(source),target_input_hash=b.array_sha(target),
                              source_noise_hash=fingerprints[i].get(f"noisy_{sigma}"),
                              target_noise_hash=fingerprints[j].get(f"noisy_{sigma}"))
                records.append(record)
                for stage in ("raw","mutual"):
                    draw(run/f"{i}_{j}_{method}_{sigma}_{stage}.png",source,target,record,stage,config["draw_limit"])
        save(run/"matches.json",records)
        save(run/"timings.json",timings)
        save(run/"config.json",config)
        report.update(status="passed",exit_code=0,record_count=len(records),numpy=np.__version__,opencv=cv2.__version__,
                      hashes={str(f):b.sha256(f.read_bytes()) for f in [config_path,Path(__file__),ROOT/"src/feature_frontend.py",ROOT/"src/image_baseline.py"]})
    except Exception:
        report.update(status="failed",exit_code=1,error=traceback.format_exc())
    report["seconds"] = time.perf_counter()-start
    save(run/"result.json",report)
    print(json.dumps(report,ensure_ascii=True,indent=2))
    return report["exit_code"]
if __name__ == "__main__":
    sys.exit(main())
