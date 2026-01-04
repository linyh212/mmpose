import os, json, cv2, random
from tqdm import tqdm
from mmdet.apis import init_detector, inference_detector
try:
    from mmpose.apis import init_pose_model
except Exception:
    from mmpose.apis import init_model as init_pose_model
from mmpose.apis import inference_topdown
from mmpose.structures import merge_data_samples
# ensure detection pipeline transforms like PackDetInputs are available across scopes
from mmengine.registry import TRANSFORMS
_registered = False
try:
    from mmpose.datasets.pipelines import PackDetInputs
    TRANSFORMS.register_module(module=PackDetInputs)
    _registered = True
except Exception:
    pass
if not _registered:
    try:
        from mmdet.datasets.pipelines import PackDetInputs as MDPack
        TRANSFORMS.register_module(module=MDPack)
        _registered = True
    except Exception:
        # Fallback: register a minimal no-op PackDetInputs so Compose can build the
        # pipeline even if mmdet/mmpose don't expose it in this environment.
        @TRANSFORMS.register_module()
        class PackDetInputs:
            def __init__(self, *args, **kwargs):
                pass

            def __call__(self, results):
                # Expected to take a dict-like `results` and return it or a
                # modified version suitable for detectors. For our use-case
                # (running inference), leaving it unchanged is sufficient.
                return results

        _registered = True

IMG_DIR = 'data/dataset/images'
ANN_DIR = 'data/dataset/annotations'
os.makedirs(ANN_DIR, exist_ok=True)

DET_CFG = 'demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py'
DET_W   = 'weights/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
POSE_CFG = 'configs/vitpose_custom.py'
POSE_W   = 'weights/vitpose-b-coco.pth'

det = init_detector(DET_CFG, DET_W, device='cuda:0')
# adapt the detector config pipeline for inference (remove LoadAnnotations etc.)
try:
    from mmpose.utils import adapt_mmdet_pipeline
    det.cfg = adapt_mmdet_pipeline(det.cfg)
except Exception:
    # If we can't adapt (older/newer mmpose), continue; fallback PackDetInputs should
    # ensure pipeline building doesn't crash, but results may be incomplete.
    pass
pose = init_pose_model(POSE_CFG, POSE_W, device='cuda:0')

images = sorted(os.listdir(IMG_DIR))
random.shuffle(images)
split = int(len(images) * 0.9)

def build_json(imgs, out):
    coco = {"images":[], "annotations":[], "categories":[{
        "id":1,"name":"person",
        "keypoints":[
            "nose","left_eye","right_eye","left_ear","right_ear",
            "left_shoulder","right_shoulder","left_elbow","right_elbow",
            "left_wrist","right_wrist","left_hip","right_hip",
            "left_knee","right_knee","left_ankle","right_ankle"
        ]
    }]}
    aid = 1
    for iid, name in enumerate(tqdm(imgs), 1):
        p = os.path.join(IMG_DIR, name)
        img = cv2.imread(p)
        h,w = img.shape[:2]
        coco["images"].append({"id":iid,"file_name":name,"width":w,"height":h})

        det_res = inference_detector(det, p)
        bboxes = det_res.pred_instances.bboxes[
            det_res.pred_instances.labels == 0]

        if len(bboxes)==0:
            continue
        # Convert detector bboxes to numpy array (N, 4) as expected by inference_topdown
        import numpy as _np
        if hasattr(bboxes, 'cpu'):
            bboxes_np = bboxes.cpu().numpy()
        else:
            bboxes_np = _np.asarray(bboxes)

        poses_res = inference_topdown(pose, p, bboxes_np)
        # Normalize to a single PoseDataSample-like object
        try:
            if isinstance(poses_res, list):
                data_samples = merge_data_samples(poses_res)
            else:
                data_samples = poses_res
        except Exception:
            data_samples = poses_res

        pred_instances = getattr(data_samples, 'pred_instances', None)
        if pred_instances is None:
            # nothing to add
            continue

        keypoints = getattr(pred_instances, 'keypoints', None)
        keypoint_scores = getattr(pred_instances, 'keypoint_scores', None)
        bboxes_out = getattr(pred_instances, 'bboxes', None)

        if keypoints is None or keypoint_scores is None:
            continue

        import numpy as _np
        def _to_numpy(x):
            if hasattr(x, 'cpu'):
                return x.cpu().numpy()
            return _np.asarray(x)

        num_persons = _to_numpy(keypoints).shape[0]
        for pi in range(num_persons):
            k = _to_numpy(keypoints[pi])
            s = _to_numpy(keypoint_scores[pi])
            kp = []
            for i in range(17):
                kp += [float(k[i][0]), float(k[i][1]), int(s[i] > 0.3)]
            bbox = None
            if bboxes_out is not None:
                bbox = _to_numpy(bboxes_out[pi]).tolist()
            else:
                # fallback: use detector bbox
                try:
                    bbox = bboxes_np[pi].tolist()
                except Exception:
                    bbox = [0, 0, 0, 0]
            coco["annotations"].append({
                "id": aid,
                "image_id": iid,
                "category_id": 1,
                "keypoints": kp,
                "num_keypoints": int((_np.asarray(s) > 0.3).sum()),
                "bbox": bbox,
                "iscrowd": 0,
                "area": w * h
            })
            aid += 1

    with open(out,'w') as f: json.dump(coco,f)

build_json(images[:split], f"{ANN_DIR}/train.json")
build_json(images[split:], f"{ANN_DIR}/val.json")
