import os, json, cv2, random
import numpy as np
from tqdm import tqdm

from mmdet.apis import init_detector, inference_detector
from mmpose.apis import init_model as init_pose_model
from mmpose.apis import inference_topdown
from mmpose.structures import merge_data_samples
from mmpose.utils import adapt_mmdet_pipeline

from mmengine.registry import TRANSFORMS

# ----------------------------------------------------------------------
# Ensure PackDetInputs exists (mmdet / mmpose version safe-guard)
# ----------------------------------------------------------------------
_registered = False
try:
    from mmpose.datasets.transforms import PackDetInputs
    TRANSFORMS.register_module(module=PackDetInputs)
    _registered = True
except Exception:
    pass

if not _registered:
    try:
        from mmdet.datasets.transforms import PackDetInputs
        TRANSFORMS.register_module(module=PackDetInputs)
        _registered = True
    except Exception:
        @TRANSFORMS.register_module()
        class PackDetInputs:
            def __init__(self, *args, **kwargs):
                pass
            def __call__(self, results):
                return results

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
IMG_DIR = 'data/dataset/images'
ANN_DIR = 'data/dataset/annotations'
os.makedirs(ANN_DIR, exist_ok=True)

DET_CFG  = 'demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py'
DET_W    = 'weights/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
POSE_CFG = 'configs/vitpose_custom.py'
POSE_W   = 'weights/vitpose-b-coco.pth'

# ----------------------------------------------------------------------
# Init models
# ----------------------------------------------------------------------
det = init_detector(DET_CFG, DET_W, device='cuda:0')
det.cfg = adapt_mmdet_pipeline(det.cfg)

pose = init_pose_model(POSE_CFG, POSE_W, device='cuda:0')

# ----------------------------------------------------------------------
# Dataset split
# ----------------------------------------------------------------------
images = sorted(os.listdir(IMG_DIR))
random.shuffle(images)
split = int(len(images) * 0.9)

def build_json(imgs, out_file, device='cuda:0'):
    coco = {
        "images": [],
        "annotations": [],
        "categories": [{
            "id": 1,
            "name": "person",
            "keypoints": [
                "nose","left_eye","right_eye","left_ear","right_ear",
                "left_shoulder","right_shoulder","left_elbow","right_elbow",
                "left_wrist","right_wrist","left_hip","right_hip",
                "left_knee","right_knee","left_ankle","right_ankle"
            ]
        }]
    }

    ann_id = 1

    for img_id, name in enumerate(tqdm(imgs), 1):
        path = os.path.join(IMG_DIR, name)
        img = cv2.imread(path)
        h, w = img.shape[:2]

        coco["images"].append({
            "id": img_id,
            "file_name": name,
            "width": w,
            "height": h
        })

        # ------------------------------
        # GPU 上的人物檢測
        # ------------------------------
        det_res = inference_detector(det, path)
        pred = det_res.pred_instances
        person_mask = pred.labels == 0

        if person_mask.sum() == 0:
            continue

        bboxes = pred.bboxes[person_mask]
        bboxes_cpu = bboxes.cpu().numpy() if hasattr(bboxes, 'cpu') else bboxes

        # ------------------------------
        # GPU 上的姿態估計
        # ------------------------------
        pose_results = inference_topdown(pose, path, bboxes_cpu)
        data_samples = merge_data_samples(pose_results)
        inst = data_samples.pred_instances

        # keypoints, scores, bboxes → CPU + numpy
        kpts = inst.keypoints
        scores = inst.keypoint_scores
        out_boxes = inst.bboxes

        kpts = kpts.cpu().numpy() if hasattr(kpts, 'cpu') else kpts
        scores = scores.cpu().numpy() if hasattr(scores, 'cpu') else scores
        out_boxes = out_boxes.cpu().numpy() if hasattr(out_boxes, 'cpu') else out_boxes

        for i in range(kpts.shape[0]):
            keypoints = []
            visible = 0
            for j in range(17):
                x, y = kpts[i, j]
                v = int(scores[i, j] > 0.3)
                visible += v
                keypoints += [float(x), float(y), v]

            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": 1,
                "keypoints": keypoints,
                "num_keypoints": visible,
                "bbox": out_boxes[i].tolist(),
                "iscrowd": 0,
                "area": w * h
            })
            ann_id += 1

    with open(out_file, 'w') as f:
        json.dump(coco, f)

build_json(images[:split], f"{ANN_DIR}/train.json", device='cuda:0')
build_json(images[split:], f"{ANN_DIR}/val.json", device='cuda:0')