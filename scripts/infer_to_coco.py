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
try:
    from mmpose.datasets.pipelines import PackDetInputs  # noqa: F401
except Exception:
    pass
try:
    from mmdet.datasets.pipelines import PackDetInputs as MDPack
    from mmengine.registry import TRANSFORMS
    TRANSFORMS.register_module(module=MDPack)
except Exception:
    pass

IMG_DIR = 'data/dataset/images'
ANN_DIR = 'data/dataset/annotations'
os.makedirs(ANN_DIR, exist_ok=True)

DET_CFG = 'demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py'
DET_W   = 'weights/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
POSE_CFG = 'configs/vitpose_custom.py'
POSE_W   = 'weights/vitpose-b-coco.pth'

det = init_detector(DET_CFG, DET_W, device='cuda:0')
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

        if len(bboxes)==0: continue
        persons=[{"bbox":b.cpu().numpy()} for b in bboxes]

        poses = inference_topdown(pose, p, persons)
        poses = merge_data_samples(poses)

        for ds in poses:
            k = ds.pred_instances.keypoints[0].cpu().numpy()
            s = ds.pred_instances.keypoint_scores[0].cpu().numpy()
            kp=[]
            for i in range(17):
                kp += [float(k[i][0]), float(k[i][1]), int(s[i]>0.3)]
            coco["annotations"].append({
                "id":aid,"image_id":iid,"category_id":1,
                "keypoints":kp,"num_keypoints":sum(s>0.3),
                "bbox":ds.pred_instances.bboxes[0].cpu().numpy().tolist(),
                "iscrowd":0,"area":w*h
            })
            aid+=1

    with open(out,'w') as f: json.dump(coco,f)

build_json(images[:split], f"{ANN_DIR}/train.json")
build_json(images[split:], f"{ANN_DIR}/val.json")