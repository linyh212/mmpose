#!/usr/bin/env python3
import os
import json
import argparse
import numpy as np
from tqdm import tqdm
import mmcv
from mmdet.apis import DetInferencer


# ---------------- IoU ----------------
def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def main(args):
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)

    # ---------------- Detector ----------------
    inferencer = DetInferencer(
        model=args.det_cfg,
        weights=args.det_weights,
        device=args.device
    )

    img_list = sorted([
        f for f in os.listdir(args.img_dir)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ])

    center_bboxes = {}
    prev_box = None

    for fname in tqdm(img_list, desc="Detect center person"):
        img_path = os.path.join(args.img_dir, fname)
        img = mmcv.imread(img_path)

        if img is None:
            continue

        h, w = img.shape[:2]
        img_center = np.array([w / 2, h / 2])

        # ---------------- Detection ----------------
        result = inferencer(img_path)
        pred = result["predictions"][0]

        bboxes = np.array(pred["bboxes"])
        scores = np.array(pred["scores"])
        labels = np.array(pred["labels"])

        # person only
        mask = labels == 0
        person_boxes = bboxes[mask]
        person_scores = scores[mask]

        if len(person_boxes) == 0:
            continue

        # score filter
        conf_mask = person_scores > args.score_thr
        if conf_mask.sum() == 0:
            conf_mask = person_scores >= person_scores.max()

        person_boxes = person_boxes[conf_mask]

        if len(person_boxes) == 0:
            continue

        # area filter
        areas = (person_boxes[:, 2] - person_boxes[:, 0]) * \
                (person_boxes[:, 3] - person_boxes[:, 1])

        large_mask = areas >= args.min_area
        person_boxes = person_boxes[large_mask]
        areas = areas[large_mask]

        if len(person_boxes) == 0:
            continue

        # distance to center
        centers = np.stack([
            (person_boxes[:, 0] + person_boxes[:, 2]) / 2,
            (person_boxes[:, 1] + person_boxes[:, 3]) / 2
        ], axis=1)

        distances = np.linalg.norm(centers - img_center, axis=1)

        # ---------------- Center + Max ----------------
        center_threshold = 0.25 * max(w, h)
        center_mask = distances < center_threshold

        if center_mask.sum() > 0:
            candidate_idx = np.where(center_mask)[0]
            selected_idx = candidate_idx[np.argmax(areas[center_mask])]
        else:
            selected_idx = np.argmax(areas)

        current_box = person_boxes[selected_idx]

        # ---------------- Temporal Tracking (IoU) ----------------
        if prev_box is not None:
            ious = np.array([compute_iou(b, prev_box) for b in person_boxes])
            best_iou_idx = np.argmax(ious)

            if ious[best_iou_idx] > 0.3:
                current_box = person_boxes[best_iou_idx]

        # ---------------- Smoothing ----------------
        if prev_box is not None:
            alpha = 0.7
            current_box = alpha * prev_box + (1 - alpha) * current_box

        prev_box = current_box.copy()

        center_bboxes[fname] = current_box.tolist()

    # ---------------- Save ----------------
    with open(args.out_json, "w") as f:
        json.dump(center_bboxes, f, indent=2)

    print(f"[INFO] Center bboxes saved at {args.out_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--img-dir", type=str, default="data/dataset/images")
    parser.add_argument("--det_cfg", type=str, default="configs/faster-rcnn_r50_fpn_1x_coco.py")
    parser.add_argument("--det_weights", type=str, default="weights/faster_rcnn_r50_fpn_coco.pth")

    parser.add_argument("--out-json", type=str,
                        default="data/dataset/annotations/center_bboxes.json")

    parser.add_argument("--score-thr", type=float, default=0.5)
    parser.add_argument("--min-area", type=float, default=5000)
    parser.add_argument("--device", type=str, default="cuda:0")

    args = parser.parse_args()
    main(args)