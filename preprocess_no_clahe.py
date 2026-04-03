"""
Preprocess VinDr-Mammo without CLAHE: DICOM → Windowing → Crop black border → PNG
Reads from source dir first, falls back to user dir for missing/corrupted DICOMs.
Saves to images_png/.
"""

import os
import pydicom
import numpy as np
import cv2
from pathlib import Path
from multiprocessing import Pool, cpu_count

SRC_IMG = "/mnt/ocean_storage/data/VinDr-Mammo/download/1.0.0/images"
USR_IMG = "/mnt/ocean_storage/users/zzhao/VinDr-Mammo/images"
DST_IMG = "/mnt/ocean_storage/users/zzhao/VinDr-Mammo/images_png"


def process_one(args):
    study_id, image_id = args
    dst_path = os.path.join(DST_IMG, study_id, image_id + ".png")
    if os.path.exists(dst_path):
        return "skip"

    # Try source first, then user dir
    for base in [SRC_IMG, USR_IMG]:
        dcm_path = os.path.join(base, study_id, image_id + ".dicom")
        if not os.path.exists(dcm_path):
            continue
        try:
            dcm = pydicom.dcmread(dcm_path)
            img = dcm.pixel_array.astype(np.float32)
            photo = str(dcm.PhotometricInterpretation)

            wc, ww = dcm.WindowCenter, dcm.WindowWidth
            wc = float(wc[0]) if isinstance(wc, pydicom.multival.MultiValue) else float(wc)
            ww = float(ww[0]) if isinstance(ww, pydicom.multival.MultiValue) else float(ww)
            lower, upper = wc - ww / 2, wc + ww / 2
            img = np.clip(img, lower, upper)
            img = ((img - lower) / (upper - lower) * 255).astype(np.uint8)

            if photo == "MONOCHROME1":
                img = 255 - img

            # Crop black border
            _, mask = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            largest = max(contours, key=cv2.contourArea)
            cx, cy, cw, ch = cv2.boundingRect(largest)
            img_cropped = img[cy:cy+ch, cx:cx+cw]

            # No CLAHE — save directly
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            cv2.imwrite(dst_path, img_cropped)
            return "ok"
        except Exception as e:
            continue  # try next base

    return "fail"


def main():
    # Collect all (study_id, image_id) pairs from both directories
    pairs = set()
    for base in [SRC_IMG, USR_IMG]:
        for study_dir in os.listdir(base):
            study_path = os.path.join(base, study_dir)
            if not os.path.isdir(study_path):
                continue
            for fname in os.listdir(study_path):
                if fname.endswith(".dicom"):
                    pairs.add((study_dir, fname.replace(".dicom", "")))

    pairs = sorted(pairs)
    print(f"Total images: {len(pairs)}. Processing with {cpu_count()} workers...")

    with Pool(processes=cpu_count()) as pool:
        results = pool.map(process_one, pairs)

    ok = results.count("ok")
    skip = results.count("skip")
    fail = results.count("fail")
    print(f"Done: {ok} processed, {skip} skipped (existing), {fail} failed.")


if __name__ == "__main__":
    main()
