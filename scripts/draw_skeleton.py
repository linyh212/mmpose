#!/usr/bin/env python3
import os
import json
import cv2
import argparse

def load_bboxes(bbox_json):
    with open(bbox_json) as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data
    elif isinstance(data, list):
        bbox_dict = {}
        for item in data:
            if "file_name" in item:
                bbox_dict[item["file_name"]] = item["bbox"]
            else:
                raise ValueError("Unsupported bbox format: missing 'file_name'")
        return bbox_dict
    else:
        raise ValueError("Unsupported bbox file format")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("img_dir")
    parser.add_argument("coco_json")
    parser.add_argument("out_dir")
    parser.add_argument("--bbox-json", default=None, help="JSON file with center bboxes")
    args = parser.parse_args()

    IMG_DIR = args.img_dir
    COCO_JSON = args.coco_json
    OUT_DIR = args.out_dir

    bbox_dict = {}
    if args.bbox_json:
        bbox_dict = load_bboxes(args.bbox_json)

    os.makedirs(OUT_DIR, exist_ok=True)

    with open(COCO_JSON) as f:
        data = json.load(f)

    img_map = {img["id"]: img["file_name"] for img in data["images"]}

    kp_map = {}
    for ann in data["annotations"]:
        kp_map.setdefault(ann["image_id"], []).append(ann["keypoints"])

    skeleton = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # head
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # arms
        (5, 11), (6, 12), (11, 12),  # torso
        (11, 13), (13, 15), (12, 14), (14, 16),  # legs
    ]

    for img_id, file_name in img_map.items():
        img_path = os.path.join(IMG_DIR, file_name)
        im = cv2.imread(img_path)
        if im is None:
            continue

        # -------- Draw bounding box --------
        if file_name in bbox_dict:
            x1, y1, x2, y2 = map(int, bbox_dict[file_name])
            cv2.rectangle(im, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # -------- Drawing key points and skeleton --------
        for kp in kp_map.get(img_id, []):
            pts, vis = [], []
            for i in range(len(kp) // 3):
                x = int(kp[i * 3])
                y = int(kp[i * 3 + 1])
                v = kp[i * 3 + 2]
                pts.append((x, y))
                vis.append(v)

            for a, b in skeleton:
                if vis[a] > 0 and vis[b] > 0:
                    cv2.line(im, pts[a], pts[b], (0, 255, 0), 2)

            for (x, y), v in zip(pts, vis):
                if v > 0:
                    cv2.circle(im, (x, y), 3, (0, 0, 255), -1)

        idx = int(file_name.split("_")[-1].split(".")[0])
        out_path = os.path.join(OUT_DIR, f"frame_{idx:06d}.jpg")
        cv2.imwrite(out_path, im)


if __name__ == "__main__":
    main()
