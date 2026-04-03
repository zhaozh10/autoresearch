"""
Preprocess the 131 missing DICOMs that were downloaded to DST_ROOT.
DICOM → Windowing → Crop black border → CLAHE → PNG (saved alongside the DICOM).
Also appends crop info to crop_info.csv and updates annotation CSVs.
"""

import os
import pydicom
import numpy as np
import cv2
import pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count

DST_ROOT = "/mnt/ocean_storage/users/zzhao/VinDr-Mammo"
IMG_DIR = os.path.join(DST_ROOT, "images")


def process_one(dcm_path):
    """Process a single DICOM file. Save PNG in the same directory."""
    try:
        dcm = pydicom.dcmread(dcm_path)
        img = dcm.pixel_array.astype(np.float32)
        photo = str(dcm.PhotometricInterpretation)

        # Windowing
        wc = dcm.WindowCenter
        ww = dcm.WindowWidth
        wc = float(wc[0]) if isinstance(wc, pydicom.multival.MultiValue) else float(wc)
        ww = float(ww[0]) if isinstance(ww, pydicom.multival.MultiValue) else float(ww)
        lower = wc - ww / 2
        upper = wc + ww / 2
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

        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_final = clahe.apply(img_cropped)

        # Save PNG alongside the DICOM
        png_path = os.path.splitext(dcm_path)[0] + ".png"
        cv2.imwrite(png_path, img_final)

        image_id = Path(dcm_path).stem
        return {
            "image_id": image_id,
            "crop_x": cx,
            "crop_y": cy,
            "new_width": cw,
            "new_height": ch,
        }
    except Exception as e:
        print(f"ERROR processing {dcm_path}: {e}")
        return None


def main():
    # Collect DICOMs that don't have a corresponding PNG yet
    dcm_paths = []
    for study_dir in sorted(os.listdir(IMG_DIR)):
        study_path = os.path.join(IMG_DIR, study_dir)
        if not os.path.isdir(study_path):
            continue
        for fname in os.listdir(study_path):
            if fname.endswith(".dicom"):
                dcm_path = os.path.join(study_path, fname)
                png_path = os.path.splitext(dcm_path)[0] + ".png"
                if not os.path.exists(png_path):
                    dcm_paths.append(dcm_path)

    print(f"Found {len(dcm_paths)} DICOMs without PNG. Processing with {cpu_count()} workers...")

    if not dcm_paths:
        print("Nothing to do.")
        return

    with Pool(processes=cpu_count()) as pool:
        results = pool.map(process_one, dcm_paths)

    # Merge new crop info into existing crop_info.csv
    crop_csv = os.path.join(DST_ROOT, "crop_info.csv")
    crop_info = {}
    if os.path.exists(crop_csv):
        old = pd.read_csv(crop_csv)
        for _, row in old.iterrows():
            crop_info[row["image_id"]] = row.to_dict()

    failed = 0
    for r in results:
        if r is None:
            failed += 1
            continue
        crop_info[r["image_id"]] = r

    print(f"Processed {len(results) - failed} images, {failed} failures. Total crop info: {len(crop_info)}.")

    crop_df = pd.DataFrame(list(crop_info.values()))
    crop_df.to_csv(crop_csv, index=False)

    # Update annotation CSVs
    SRC_ROOT = "/mnt/ocean_storage/data/VinDr-Mammo/download/1.0.0"

    breast = pd.read_csv(os.path.join(SRC_ROOT, "breast-level_annotations.csv"))
    breast["height"] = breast["image_id"].map(lambda x: crop_info[x]["new_height"] if x in crop_info else None)
    breast["width"] = breast["image_id"].map(lambda x: crop_info[x]["new_width"] if x in crop_info else None)
    breast.to_csv(os.path.join(DST_ROOT, "breast-level_annotations.csv"), index=False)

    finding = pd.read_csv(os.path.join(SRC_ROOT, "finding_annotations.csv"))
    finding["height"] = finding["image_id"].map(lambda x: crop_info[x]["new_height"] if x in crop_info else None)
    finding["width"] = finding["image_id"].map(lambda x: crop_info[x]["new_width"] if x in crop_info else None)

    has_bbox = finding["xmin"].notna()
    for idx in finding[has_bbox].index:
        iid = finding.loc[idx, "image_id"]
        if iid not in crop_info:
            continue
        ci = crop_info[iid]
        finding.loc[idx, "xmin"] = max(0, finding.loc[idx, "xmin"] - ci["crop_x"])
        finding.loc[idx, "ymin"] = max(0, finding.loc[idx, "ymin"] - ci["crop_y"])
        finding.loc[idx, "xmax"] = min(ci["new_width"], finding.loc[idx, "xmax"] - ci["crop_x"])
        finding.loc[idx, "ymax"] = min(ci["new_height"], finding.loc[idx, "ymax"] - ci["crop_y"])

    finding.to_csv(os.path.join(DST_ROOT, "finding_annotations.csv"), index=False)
    print("Updated annotations. Done!")


if __name__ == "__main__":
    main()
