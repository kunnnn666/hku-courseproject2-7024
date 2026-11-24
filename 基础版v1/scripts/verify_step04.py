"""Independent numeric and replay checks. Run after two F04 runs."""
import json
import sys
import time
import uuid
from pathlib import Path
import cv2
import numpy as np
from run_step04 import ROOT, save
from src import image_baseline as b
from src.feature_frontend import screen, match_images

def main():
    run = ROOT/"runs"/("F04_verify_"+uuid.uuid4().hex[:8])
    run.mkdir()
    start = time.perf_counter()
    result = dict(command=[sys.executable,*sys.argv],cwd=str(Path.cwd()))
    try:
        first,second = sorted(ROOT.glob("runs/F04_20*"))[-2:]
        result["compared_runs"] = [str(first),str(second)]
        for name in ("matches.json","input_arrays.json","frames.json","config.json"):
            assert (first/name).read_bytes() == (second/name).read_bytes(),name
        config = json.loads((first/"config.json").read_text())
        records = json.loads((first/"matches.json").read_text())
        for item in records:
            assert item["counts"]["mutual"] <= item["counts"]["ratio_pass"] <= item["counts"]["raw"]
            for stage in ("raw","ratio_pass","mutual"):
                assert len(item[stage]) == item["counts"][stage]
                for m in item[stage]:
                    for side in ("source","target"):
                        assert 0 <= m[side+"_index"] < item[side+"_features"]
                        u,v = m[side+"_uv"]
                        height,width,_ = item[side]["shape"]
                        assert np.isfinite([u,v]).all() and 0 <= u < width and 0 <= v < height
            for side in ("source","target"):
                assert len({m[side+"_index"] for m in item["mutual"]}) == len(item["mutual"])
        fingerprints = json.loads((first/"input_arrays.json").read_text())
        frames = json.loads((first/"frames.json").read_text())
        for index,row in frames.items():
            raw = cv2.imdecode(np.fromfile(row["path"],np.uint8),cv2.IMREAD_UNCHANGED)
            rgb = raw[...,::-1].astype(np.float32)/np.float32(255)
            for sigma in config["sigmas"]:
                noise,_ = b.add_gaussian(rgb,row["pair_id"],sigma,7024)
                filtered = b.gaussian_filter(noise,7,1)
                assert b.array_sha(noise) == fingerprints[index][f"noisy_{sigma}"]
                assert b.array_sha(filtered) == fingerprints[index][f"filtered_{sigma}"]
        descriptors = np.zeros((3,32),np.uint8)
        descriptors[1] = 255
        descriptors[2] = 85
        exact = screen(descriptors,descriptors)
        assert [(m["source_index"],m["target_index"],m["distance"]) for m in exact["mutual"]] == [(0,0,0),(1,1,0),(2,2,0)]
        assert exact["raw"][0]["second_distance"] == 128
        tied = screen(descriptors[:1],np.zeros((2,32),np.uint8))
        assert tied["raw"][0]["target_index"] == 0 and not tied["ratio_pass"]
        scarce = screen(descriptors,descriptors[:1])
        assert scarce["status"] == "insufficient_neighbors" and scarce["counts"]["forward_missing_second"] == 3
        empty = match_images(np.zeros((120,140,3),np.float32),np.zeros((150,160,3),np.float32),config["orb"],.75)
        assert empty["status"] == "no_descriptors"
        rng = np.random.default_rng(7024)
        gray = rng.integers(0,256,(300,400),dtype=np.uint8)
        gray = cv2.GaussianBlur(gray,(3,3),.6)
        rgb = np.repeat(gray[:,:,None],3,axis=2).astype(np.float32)/255
        translated = cv2.warpAffine(rgb,np.float32([[1,0,9],[0,1,6]]),(400,300))
        shift = match_images(rgb,translated,config["orb"],.75)
        offsets = np.array([np.subtract(m["target_uv"],m["source_uv"]) for m in shift["mutual"]])
        errors = np.linalg.norm(offsets-[9,6],axis=1)
        assert len(errors)>=20 and np.median(errors)<1 and (errors<2).mean()>.8
        result.update(status="passed",exit_code=0,records=len(records),byte_identical=["matches","frames","input_arrays","config"],
                      timing_fields="excluded",all_coordinates_and_unique_indices=True,
                      regenerated_all_frame_noise_and_filter_hashes=True,synthetic_exact=exact["counts"],
                      tied_candidate=tied["counts"],insufficient=scarce["counts"],blank_status=empty["status"],
                      translation=dict(expected_uv=[9,6],matches=len(errors),median_offset=np.median(offsets,axis=0).tolist(),
                                       median_error=float(np.median(errors)),fraction_below_2px=float((errors<2).mean()),
                                       tolerance="median <1 px and >80% below2px; >=20 matches"))
    except Exception:
        import traceback
        result.update(status="failed",exit_code=1,error=traceback.format_exc())
    result["seconds"] = time.perf_counter()-start
    save(run/"result.json",result)
    print(json.dumps(result,ensure_ascii=True,indent=2))
    return result["exit_code"]
if __name__=="__main__":
    sys.exit(main())
