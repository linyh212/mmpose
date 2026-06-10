#!/usr/bin/env python3
import os
import json
import random
import cv2
import numpy as np
import mmcv
from tqdm import tqdm
from mmdet.apis import DetInferencer

# ================= Config =================
IMG_DIR = "data/dataset/images"
ANN_DIR = "data/dataset/annotations"
os.makedirs(ANN_DIR, exist_ok=True)

TRAIN_JSON = os.path.join(ANN_DIR, "train.json")
VAL_JSON = os.path.join(ANN_DIR, "val.json")

SCORE_THR = 0.5
TRAIN_RATIO = 0.9

# ================= Init Detector =================
inferencer = DetInferencer(
    model="configs/faster-rcnn_r50_fpn_1x_coco.py",
    weights="weights/faster_rcnn_r50_fpn_coco.pth",
    device="cuda:0",
)

# ================= Collect images =================
img_list = sorted(
    [f for f in os.listdir(IMG_DIR) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
)

random.shuffle(img_list)
split = int(len(img_list) * TRAIN_RATIO)
train_imgs = img_list[:split]
val_imgs = img_list[split:]


# ================= Helper =================
def build_coco(imgs, out_file):
    coco = {
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": 1,
                "name": "person",
                "keypoints": [
                    "nose",
                    "left_eye",
                    "right_eye",
                    "left_ear",
                    "right_ear",
                    "left_shoulder",
                    "right_shoulder",
                    "left_elbow",
                    "right_elbow",
                    "left_wrist",
                    "right_wrist",
                    "left_hip",
                    "right_hip",
                    "left_knee",
                    "right_knee",
                    "left_ankle",
                    "right_ankle",
                ],
            }
        ],
    }

    ann_id = 1

    for img_id, fname in enumerate(tqdm(imgs, desc=f"Processing {out_file}"), 1):
        img_path = os.path.join(IMG_DIR, fname)
        img = mmcv.imread(img_path)
        if img is None:
            continue

        h, w = img.shape[:2]
        img_center = np.array([w / 2, h / 2])

        coco["images"].append(
            {"id": img_id, "file_name": fname, "width": w, "height": h}
        )

        # -------- Detection --------
        result = inferencer(img_path)
        pred = result["predictions"][0]

        bboxes = np.array(pred["bboxes"])
        scores = np.array(pred["scores"])
        labels = np.array(pred["labels"])

        # person only
        mask = labels == 0
        bboxes = bboxes[mask]
        scores = scores[mask]

        if len(bboxes) == 0:
            continue

        # score filter
        conf_mask = scores > SCORE_THR
        if conf_mask.sum() == 0:
            conf_mask = scores >= scores.max()
        bboxes = bboxes[conf_mask]

        # -------- pick center person --------
        def center_dist(box):
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2
            return np.linalg.norm(np.array([cx, cy]) - img_center)

        idx = np.argmin([center_dist(b) for b in bboxes])
        box = bboxes[idx]

        x1, y1, x2, y2 = box
        w_box = x2 - x1
        h_box = y2 - y1

        # -------- COCO annotation --------
        coco["annotations"].append(
            {
                "id": ann_id,
                "image_id": img_id,
                "category_id": 1,
                "bbox": [float(x1), float(y1), float(w_box), float(h_box)],
                "area": float(w_box * h_box),
                "iscrowd": 0,
                "keypoints": [0.0, 0.0, 0] * 17,
                "num_keypoints": 0,
            }
        )
        ann_id += 1

    with open(out_file, "w") as f:
        json.dump(coco, f, indent=2)

    print(f"[OK] Saved {out_file}")


# ================= Run =================
build_coco(train_imgs, TRAIN_JSON)
build_coco(val_imgs, VAL_JSON)
