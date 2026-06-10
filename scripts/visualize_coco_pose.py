#!/usr/bin/env python3
import os
import json
import cv2
import numpy as np
import argparse

# ================= CLI =================
parser = argparse.ArgumentParser(description="Visualize COCO keypoints with skeleton")
parser.add_argument("--img-dir", type=str, required=True, help="Image folder")
parser.add_argument("--coco-json", type=str, required=True, help="COCO JSON file")
parser.add_argument("--out-dir", type=str, default="outputs_debug_vis", help="Output folder")
args = parser.parse_args()

IMG_DIR = args.img_dir
COCO_JSON = args.coco_json
OUT_DIR = args.out_dir

os.makedirs(OUT_DIR, exist_ok=True)

# ================= Load COCO =================
with open(COCO_JSON) as f:
    coco = json.load(f)

if "images" not in coco or "annotations" not in coco:
    raise ValueError("Invalid COCO JSON")

img_map = {img["id"]: img for img in coco["images"]}

# ================= Skeleton (FIX) =================
if "categories" in coco and len(coco["categories"]) > 0 and "skeleton" in coco["categories"][0]:
    skeleton_1_based = coco["categories"][0]["skeleton"]
else:
    print("[WARN] No skeleton in JSON → using default COCO skeleton")
    skeleton_1_based = [
        [16,14],[14,12],[17,15],[15,13],[12,13],
        [6,12],[7,13],[6,7],[6,8],[7,9],
        [8,10],[9,11],[2,3],[1,2],[1,3],
        [2,4],[3,5],[4,6],[5,7]
    ]

SKELETON = [(i-1, j-1) for i,j in skeleton_1_based]

# ================= Colors =================
COLOR_KPT = (0, 255, 0)      # 綠：keypoints
COLOR_SKE = (255, 0, 0)      # 藍：skeleton
COLOR_BOX = (0, 0, 255)      # 紅：bbox

# ================= Main Loop =================
for ann in coco["annotations"]:
    img_id = ann["image_id"]

    if img_id not in img_map:
        continue

    img_info = img_map[img_id]
    img_path = os.path.join(IMG_DIR, img_info["file_name"])

    img = cv2.imread(img_path)
    if img is None:
        print(f"[WARN] Image not found: {img_path}")
        continue

    # -------- Keypoints --------
    if "keypoints" in ann:
        kpts = np.array(ann["keypoints"]).reshape(-1, 3)

        for i, (x, y, v) in enumerate(kpts):
            if v > 0:
                cv2.circle(img, (int(x), int(y)), 4, COLOR_KPT, -1)
                cv2.putText(img, str(i),
                            (int(x)+3, int(y)-3),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, COLOR_KPT, 1)

        # -------- Skeleton --------
        for i, j in SKELETON:
            if i < len(kpts) and j < len(kpts):
                if kpts[i][2] > 0 and kpts[j][2] > 0:
                    pt1 = (int(kpts[i][0]), int(kpts[i][1]))
                    pt2 = (int(kpts[j][0]), int(kpts[j][1]))
                    cv2.line(img, pt1, pt2, COLOR_SKE, 2)

    # -------- Bounding Box --------
    if "bbox" in ann:
        x, y, w, h = ann["bbox"]
        x1, y1 = int(x), int(y)
        x2, y2 = int(x + w), int(y + h)

        cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_BOX, 2)

        # center
        cx = int(x + w / 2)
        cy = int(y + h / 2)
        cv2.circle(img, (cx, cy), 4, (0, 255, 255), -1)

    # -------- Save --------
    out_path = os.path.join(OUT_DIR, img_info["file_name"])
    cv2.imwrite(out_path, img)

print(f"[OK] Visualization saved to {OUT_DIR}")