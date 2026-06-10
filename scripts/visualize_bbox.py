#!/usr/bin/env python3
import os
import json
import cv2
import numpy as np
import argparse

def main(args):
    os.makedirs(args.vis_dir, exist_ok=True)

    with open(args.coco_json, "r") as f:
        data = json.load(f)

    print("[INFO] Using center bbox visualization (compatible mode)")

    for fname, box in data.items():
        img_path = os.path.join(args.img_dir, fname)
        img = cv2.imread(img_path)

        if img is None:
            print(f"[WARN] Missing {fname}")
            continue

        vis_img = img.copy()

        # ===== current_box =====
        x1, y1, x2, y2 = map(int, box)
        current_box = np.array([x1, y1, x2, y2])

        # ===== person_boxes（模擬）=====
        person_boxes = np.array([current_box])  # 只有一個人

        # ===== image center =====
        H, W = img.shape[:2]
        img_center = np.array([W // 2, H // 2])

        # ===== 原本你的畫法 =====
        for i, box in enumerate(person_boxes.astype(int)):
            color = (0, 255, 0) if np.all(box == current_box.astype(int)) else (0, 0, 255)
            cv2.rectangle(vis_img, (box[0], box[1]), (box[2], box[3]), color, 2)

        cv2.circle(vis_img, tuple(img_center.astype(int)), 5, (255, 0, 0), -1)

        # ===== save =====
        out_path = os.path.join(args.vis_dir, fname)
        cv2.imwrite(out_path, vis_img)

    print(f"[OK] Saved to {args.vis_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-dir", required=True)
    parser.add_argument("--coco-json", required=True)
    parser.add_argument("--vis-dir", required=True)
    args = parser.parse_args()

    main(args)