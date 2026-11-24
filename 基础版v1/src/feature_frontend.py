"""Deterministic descriptor screening; coordinates remain in each original image."""
import cv2
import numpy as np
from .image_baseline import check_rgb, rgb_to_uint8

BITS = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(axis=1)

def detect(rgb, parameters):
    check_rgb(rgb)
    if parameters["WTA_K"] != 2:
        raise ValueError("Ordinary Hamming requires WTA_K=2")
    gray = cv2.cvtColor(rgb_to_uint8(rgb), cv2.COLOR_RGB2GRAY)
    keypoints, descriptors = cv2.ORB_create(**parameters).detectAndCompute(gray, None)
    if descriptors is None:
        return [], np.empty((0, 32), np.uint8)
    order = sorted(range(len(keypoints)), key=lambda i: (
        keypoints[i].pt[0], keypoints[i].pt[1], keypoints[i].octave,
        keypoints[i].angle, keypoints[i].response, descriptors[i].tobytes(), i))
    points = [list(map(float, keypoints[i].pt)) for i in order]
    return points, descriptors[order]

def screen(source, target, ratio=0.75):
    for descriptors in (source, target):
        if descriptors.ndim != 2 or descriptors.shape[1] != 32 or descriptors.dtype != np.uint8:
            raise ValueError("Descriptors must be N x 32 uint8")
    if not 0 < ratio < 1:
        raise ValueError("Ratio must lie strictly between zero and one")
    distances = np.empty((len(source), len(target)), np.uint16)
    for i, descriptor in enumerate(source):
        distances[i] = BITS[np.bitwise_xor(descriptor, target)].sum(axis=1)
    def one_way(matrix):
        raw, accepted = [], []
        for i, row in enumerate(matrix):
            if not len(row):
                continue
            order = np.argsort(row, kind="stable")
            j = int(order[0])
            second = int(row[order[1]]) if len(row) >= 2 else None
            match = dict(source_index=i, target_index=j, distance=int(row[j]), second_distance=second)
            raw.append(match)
            if second is not None and row[j] < ratio * second:
                accepted.append(match)
        return raw, accepted
    raw, forward = one_way(distances)
    reverse_raw, reverse = one_way(distances.T)
    reverse_pairs = {(m["target_index"], m["source_index"]) for m in reverse}
    mutual = [m for m in forward if (m["source_index"], m["target_index"]) in reverse_pairs]
    status = "ok" if mutual else "no_matches"
    if not len(source) or not len(target):
        status = "no_descriptors"
    elif len(source) < 2 or len(target) < 2:
        status = "insufficient_neighbors"
    return dict(status=status, source_features=len(source), target_features=len(target),
                raw=raw, ratio_pass=forward, mutual=mutual,
                counts=dict(raw=len(raw), ratio_pass=len(forward), reverse_ratio_pass=len(reverse),
                            mutual=len(mutual), forward_missing_second=len(raw) if len(target)<2 else 0,
                            reverse_missing_second=len(reverse_raw) if len(source)<2 else 0))

def match_images(source, target, parameters, ratio):
    source_points, source_descriptors = detect(source, parameters)
    target_points, target_descriptors = detect(target, parameters)
    result = screen(source_descriptors, target_descriptors, ratio)
    for stage in ("raw", "ratio_pass", "mutual"):
        for match in result[stage]:
            i, j = match["source_index"], match["target_index"]
            match["source_uv"] = source_points[i]
            match["target_uv"] = target_points[j]
    result["source_points"] = source_points
    result["target_points"] = target_points
    return result
