"""
Draw bbox annotations on VinDr-Mammo preprocessed images.
Saves to images_bbox/ with the same study/image directory structure.
"""

import os
import cv2
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count

DST_ROOT = "/mnt/ocean_storage/users/zzhao/VinDr-Mammo"
IMG_SRC = os.path.join(DST_ROOT, "images")
IMG_DST = os.path.join(DST_ROOT, "images_bbox")

# Color palette for different finding categories
COLORS = [
    (0, 255, 0),    # green
    (0, 0, 255),    # red
    (255, 0, 0),    # blue
    (0, 255, 255),  # yellow
    (255, 0, 255),  # magenta
    (255, 255, 0),  # cyan
    (128, 255, 0),  # lime
    (0, 128, 255),  # orange
    (255, 0, 128),  # pink
    (128, 0, 255),  # purple
]


def draw_one(args):
    """Draw bboxes on a single image."""
    study_id, image_id, bboxes = args
    try:
        src_path = os.path.join(IMG_SRC, study_id, image_id + ".png")
        if not os.path.exists(src_path):
            print(f"MISSING {src_path}")
            return False

        img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        for bbox in bboxes:
            cat, xmin, ymin, xmax, ymax, color = bbox
            pt1 = (int(round(xmin)), int(round(ymin)))
            pt2 = (int(round(xmax)), int(round(ymax)))
            cv2.rectangle(img, pt1, pt2, color, 2)
            # Label
            label = cat.strip("[]'\"")
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (pt1[0], pt1[1] - th - 6), (pt1[0] + tw + 4, pt1[1]), color, -1)
            cv2.putText(img, label, (pt1[0] + 2, pt1[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        dst_path = os.path.join(IMG_DST, study_id, image_id + ".png")
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        cv2.imwrite(dst_path, img)
        return True
    except Exception as e:
        print(f"ERROR {study_id}/{image_id}: {e}")
        return False


def main():
    finding = pd.read_csv(os.path.join(DST_ROOT, "finding_annotations.csv"))
    has_bbox = finding["xmin"].notna()
    df = finding[has_bbox].copy()

    # Assign colors by category
    categories = df["finding_categories"].unique().tolist()
    cat_color = {c: COLORS[i % len(COLORS)] for i, c in enumerate(categories)}

    # Group bboxes by (study_id, image_id)
    tasks = []
    for (study_id, image_id), grp in df.groupby(["study_id", "image_id"]):
        bboxes = []
        for _, row in grp.iterrows():
            color = cat_color[row["finding_categories"]]
            bboxes.append((row["finding_categories"], row["xmin"], row["ymin"], row["xmax"], row["ymax"], color))
        tasks.append((study_id, image_id, bboxes))

    print(f"Drawing bboxes on {len(tasks)} images with {cpu_count()} workers...")

    with Pool(processes=cpu_count()) as pool:
        results = pool.map(draw_one, tasks)

    ok = sum(r for r in results if r)
    fail = sum(1 for r in results if not r)
    print(f"Done: {ok} success, {fail} failures.")


if __name__ == "__main__":
    main()
