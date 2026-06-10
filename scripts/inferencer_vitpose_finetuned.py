#!/usr/bin/env python3
import os
import json
import argparse
import numpy as np
from mmpose.apis import init_model, inference_topdown

# ================= CLI arguments =================
parser = argparse.ArgumentParser()
parser.add_argument("--img-dir", required=True)
parser.add_argument("--coco-json", required=True)
parser.add_argument("--out-dir", required=True)
parser.add_argument("--out-json", default="results_coco.json")
parser.add_argument("--pose-config", default=None)
parser.add_argument("--pose-checkpoint", default=None)

args = parser.parse_args()

IMG_DIR = args.img_dir
JSON_FILE = args.coco_json
OUT_DIR = args.out_dir
OUT_JSON = os.path.join(OUT_DIR, args.out_json)

os.makedirs(OUT_DIR, exist_ok=True)

# ================= Load COCO bbox JSON =================
with open(JSON_FILE, "r") as f:
    coco_bbox = json.load(f)

img_bboxes = {}
for ann in coco_bbox["annotations"]:
    img_id = ann["image_id"]
    x, y, w, h = ann["bbox"]
    bbox_xyxy = [x, y, x + w, y + h]
    img_bboxes.setdefault(img_id, []).append(bbox_xyxy)

img_id2file = {
    img["id"]: os.path.join(IMG_DIR, img["file_name"]) for img in coco_bbox["images"]
}

# ================= Init pose model =================
POSE_CFG = args.pose_config or "configs/vitpose_custom.py"
POSE_CKPT = args.pose_checkpoint or "work_dirs/vitpose_custom/best_coco_AP.pth"

if not os.path.exists(POSE_CKPT):
    print(f"[WARN] checkpoint not found: {POSE_CKPT}")
    print("[WARN] fallback to latest epoch")

    work_dir = os.path.dirname(POSE_CKPT)
    ckpts = sorted(
        [f for f in os.listdir(work_dir) if f.startswith("epoch_")],
        reverse=True,
    )
    if len(ckpts) == 0:
        raise RuntimeError("No checkpoint found!")
    POSE_CKPT = os.path.join(work_dir, ckpts[0])

print(f"[INFO] Using config: {POSE_CFG}")
print(f"[INFO] Using checkpoint: {POSE_CKPT}")

pose_model = init_model(POSE_CFG, POSE_CKPT, device="cuda:0")

# ================= Run inference =================
ann_id = 1

coco_out = {
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

added_images = set()

for img_id, bboxes in img_bboxes.items():
    img_file = img_id2file[img_id]

    results = inference_topdown(pose_model, img_file, bboxes=bboxes)

    if img_id not in added_images:
        coco_out["images"].append(
            {"id": img_id, "file_name": os.path.basename(img_file)}
        )
        added_images.add(img_id)

    for pred_idx, pred in enumerate(results):
        bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bboxes[pred_idx]

        instances = pred.pred_instances
        kpts = np.squeeze(instances.keypoints)

        if kpts.size == 0:
            continue

        if kpts.shape[1] == 3:
            coords = kpts[:, :2]
            scores = kpts[:, 2]
        else:
            coords = kpts
            scores = getattr(instances, "keypoint_scores", None)
            if scores is None:
                scores = np.ones(coords.shape[0])

        coords = coords.astype(float)
        scores = np.array(scores).flatten()

        if scores.shape[0] != coords.shape[0]:
            scores = np.ones(coords.shape[0])

        kpts_flat = []
        xs, ys = [], []

        for i in range(coords.shape[0]):
            x = float(np.clip(coords[i, 0], bbox_x1, bbox_x2))
            y = float(np.clip(coords[i, 1], bbox_y1, bbox_y2))
            s = scores[i]
            v = 2 if s > 0.1 else 0

            kpts_flat += [x, y, v]
            xs.append(x)
            ys.append(y)

        x_min, y_min = min(xs), min(ys)
        w_box, h_box = max(xs) - x_min, max(ys) - y_min

        coco_out["annotations"].append(
            {
                "id": ann_id,
                "image_id": img_id,
                "category_id": 1,
                "bbox": [x_min, y_min, w_box, h_box],
                "area": w_box * h_box,
                "iscrowd": 0,
                "keypoints": kpts_flat,
                "num_keypoints": sum(v > 0 for v in kpts_flat[2::3]),
            }
        )
        ann_id += 1

# ================= Save =================
with open(OUT_JSON, "w") as f:
    json.dump(coco_out, f, indent=2)

print(f"[OK] Inference finished → {OUT_JSON}")
