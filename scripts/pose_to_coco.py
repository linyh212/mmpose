#!/usr/bin/env python3
import os
import json
import mmcv
import numpy as np
from mmpose.apis import init_model, inference_topdown
import argparse

# ================= CLI =================
parser = argparse.ArgumentParser()
parser.add_argument("--img-dir", required=True)
parser.add_argument("--bbox-json", required=True)
parser.add_argument("--pose-config", required=True)
parser.add_argument("--pose-checkpoint", required=True)
parser.add_argument("--out-train", required=True)
parser.add_argument("--out-val", required=True)
parser.add_argument("--out-test", required=True)
parser.add_argument("--smooth-alpha", type=float, default=0.7)
parser.add_argument("--train-ratio", type=float, default=0.8)
parser.add_argument("--val-ratio", type=float, default=0.1)
args = parser.parse_args()

IMG_DIR = args.img_dir

# ================= COCO =================
COCO_KEYPOINTS = [
    "nose","left_eye","right_eye","left_ear","right_ear",
    "left_shoulder","right_shoulder","left_elbow","right_elbow",
    "left_wrist","right_wrist","left_hip","right_hip",
    "left_knee","right_knee","left_ankle","right_ankle"
]

def new_coco():
    return {
        "images": [],
        "annotations": [],
        "categories": [{
            "id": 1,
            "name": "person",
            "keypoints": COCO_KEYPOINTS
        }]
    }

# ================= Load =================
with open(args.bbox_json) as f:
    center_bboxes = json.load(f)

img_list = sorted(center_bboxes.keys())

# ================= Split =================
n = len(img_list)
n_train = int(n * args.train_ratio)
n_val = int(n * args.val_ratio)

train_list = img_list[:n_train]
val_list = img_list[n_train:n_train+n_val]
test_list = img_list[n_train+n_val:]

# ================= Init =================
pose_model = init_model(args.pose_config, args.pose_checkpoint, device="cuda:0")
pose_model.eval()

coco_train, coco_val, coco_test = new_coco(), new_coco(), new_coco()

# ================= FUNCTIONS =================

def smooth(curr, prev, scores, alpha):
    if prev is None:
        return curr

    curr = curr.copy()

    for i in range(len(scores)):
        if scores[i] < 0.5:
            curr[i] = prev[i]
        else:
            curr[i] = alpha * curr[i] + (1 - alpha) * prev[i]

    return curr


def fix_feet(curr, prev, scores):
    if prev is None:
        return curr

    for i in [15, 16]:  # ankles
        if scores[i] < 0.5:
            curr[i] = prev[i]

    return curr


def fix_lr(curr, prev, scores, state):
    if prev is None:
        return curr

    L_KNEE, R_KNEE = 13, 14
    L_ANK, R_ANK = 15, 16

    # 低信心 → 不判斷
    if scores[L_ANK] < 0.5 or scores[R_ANK] < 0.5:
        return curr

    d_same = (
        np.linalg.norm(curr[L_ANK] - prev[L_ANK]) +
        np.linalg.norm(curr[R_ANK] - prev[R_ANK])
    )
    d_swap = (
        np.linalg.norm(curr[L_ANK] - prev[R_ANK]) +
        np.linalg.norm(curr[R_ANK] - prev[L_ANK])
    )

    # ===== hysteresis 防抖 =====
    if not state["swapped"]:
        if d_swap + 5 < d_same:
            curr[[L_KNEE, R_KNEE]] = curr[[R_KNEE, L_KNEE]]
            curr[[L_ANK, R_ANK]] = curr[[R_ANK, L_ANK]]
            state["swapped"] = True
    else:
        if d_same + 5 < d_swap:
            state["swapped"] = False

    return curr


def visibility(scores):
    v = np.zeros_like(scores, dtype=int)
    v[scores > 0.5] = 2
    v[(scores > 0.2) & (scores <= 0.5)] = 1
    return v


# ================= CORE =================
def process(img_list, coco):
    img_id = 1
    ann_id = 1
    prev = None
    state = {"swapped": False}  # 🔥 關鍵

    for name in img_list:
        path = os.path.join(IMG_DIR, name)
        img = mmcv.imread(path)
        h, w = img.shape[:2]

        x1, y1, x2, y2 = center_bboxes[name]
        bbox = np.array([[x1, y1, x2, y2]], dtype=np.float32)

        result = inference_topdown(pose_model, path, bboxes=bbox)

        if len(result) == 0:
            if prev is not None:
                kpts = prev.copy()
                scores = np.ones(17) * 0.3
            else:
                kpts = np.zeros((17, 2))
                scores = np.zeros(17)
        else:
            kpts = result[0].pred_instances.keypoints
            scores = result[0].pred_instances.keypoint_scores

            if hasattr(kpts, "cpu"):
                kpts = kpts.cpu().numpy()
                scores = scores.cpu().numpy()

            if kpts.ndim == 3:
                kpts = kpts[0]
                scores = scores[0]

        # ===== 🔥 正確 temporal pipeline =====
        kpts = fix_lr(kpts, prev, scores, state)
        kpts = fix_feet(kpts, prev, scores)
        kpts = smooth(kpts, prev, scores, args.smooth_alpha)

        prev = kpts.copy()

        vis = visibility(scores)

        flat = []
        visible = 0

        for (x, y), v in zip(kpts, vis):
            flat.extend([float(x), float(y), int(v)])
            if v > 0:
                visible += 1

        coco["images"].append({
            "id": img_id,
            "file_name": name,
            "width": w,
            "height": h
        })

        coco["annotations"].append({
            "id": ann_id,
            "image_id": img_id,
            "category_id": 1,
            "bbox": [float(x1), float(y1), float(x2-x1), float(y2-y1)],
            "area": float((x2-x1)*(y2-y1)),
            "iscrowd": 0,
            "keypoints": flat,
            "num_keypoints": visible
        })

        img_id += 1
        ann_id += 1


# ================= RUN =================
process(train_list, coco_train)
process(val_list, coco_val)
process(test_list, coco_test)

# ================= SAVE =================
os.makedirs(os.path.dirname(args.out_train), exist_ok=True)

json.dump(coco_train, open(args.out_train, "w"), indent=2)
json.dump(coco_val, open(args.out_val, "w"), indent=2)
json.dump(coco_test, open(args.out_test, "w"), indent=2)

print("DONE (stable temporal pose)")