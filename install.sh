#!/bin/bash
set -e

#####################################
# Config
#####################################
FPS=30
DET_CFG="configs/faster-rcnn_r50_fpn_1x_coco.py"                     # 物件檢測 config
DET_WEIGHTS="weights/faster_rcnn_r50_fpn_coco.pth"                   # 物件檢測 checkpoint
POSE_CFG="configs/vitpose_custom.py"                                 # 姿態估計 config
POSE_PRETRAIN=weights/vitpose-b-coco.pth
TRAIN_CFG=configs/vitpose_custom.py
WORK_DIR="work_dirs/vitpose_custom"                                  # 訓練工作目錄

#####################################
# 0. Prepare dirs
#####################################
mkdir -p frames data/dataset/images data/dataset/annotations
mkdir -p skeleton_vis outputs_finetuned work_dirs

#####################################
# 1. Video → Frames
#####################################
echo "=== [1] Extract frames ==="
for v in videos/*.{mp4,MP4}; do
  [ -f "$v" ] || continue
  name=$(basename "$v" | sed 's/\.[mM][pP]4//')
  mkdir -p frames/$name
  ffmpeg -y -i "$v" -vf fps=$FPS frames/$name/frame_%06d.jpg
done

#####################################
# 2. Collect frames
#####################################
echo "=== [2] Collect frames ==="
rm -f data/dataset/images/*.jpg
c=1
for d in frames/*; do
  for f in $d/*.jpg; do
    printf -v n "frame_%06d.jpg" "$c"
    cp "$f" data/dataset/images/$n
    ((c++))
  done
done

#####################################
# 3. Frames → COCO keypoints
#####################################
echo "=== [3] Generate COCO annotations ==="
python3 scripts/infer_to_coco.py

#####################################
# 4. Train ViTPose
#####################################
echo "=== [4] Train ViTPose ==="
python scripts/train.py \
  $TRAIN_CFG \
  --work-dir $WORK_DIR

#####################################
# 5. Inference (finetuned)
#####################################
echo "=== [5] Inference finetuned ==="
python demo/topdown_demo_with_mmdet.py \
  --det-config $DET_CFG \
  --det-checkpoint $DET_WEIGHTS \
  --pose-config $POSE_CFG \
  --pose-checkpoint $WORK_DIR/best_AP.pth \
  --input data/dataset/images \
  --output-root outputs_finetuned \
  --save-predictions

#####################################
# 6. Draw skeleton
#####################################
echo "=== [6] Draw skeleton ==="
python scripts/draw_skeleton.py \
  data/dataset/images \
  outputs_finetuned/results.json \
  skeleton_vis

#####################################
# 7. Skeleton → Video
#####################################
echo "=== [7] Skeleton video ==="
ffmpeg -y -r $FPS -i skeleton_vis/frame_%06d.jpg \
  -c:v libx264 -pix_fmt yuv420p skeleton.mp4

echo "=== PIPELINE DONE ==="