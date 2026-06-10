#!/bin/bash
set -e

#####################################
# Config
#####################################
FPS=20
# ===== Step 4: OFFICIAL ViTPose (pseudo label) =====
POSE_CFG_PSEUDO="mmpose/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-base-simple_8xb64-210e_coco-256x192.py"
POSE_PRETRAIN_PSEUDO="mmpose/weights/td-hm_ViTPose-base-simple_8xb64-210e_coco-256x192-0b8234ea_20230407.pth"
# ===== Step 5+: CUSTOM ViTPose =====
POSE_CFG_CUSTOM="configs/vitpose_custom.py"
WORK_DIR="work_dirs/vitpose_custom"

IMG_DIR="data/dataset/images"
ANN_DIR="data/dataset/annotations"

export PYTHONPATH=$PYTHONPATH:$(pwd)/mmpose

#####################################
# 0. Prepare dirs
#####################################
mkdir -p frames "$IMG_DIR" "$ANN_DIR"
mkdir -p skeleton_vis outputs_finetuned "$WORK_DIR"

#####################################
# 1. Video → Frames (except A01)
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
rm -f "$IMG_DIR"/*.jpg
c=1
for d in frames/*; do
  for f in "$d"/*.jpg; do
    printf -v n "frame_%06d.jpg" "$c"
    cp "$f" "$IMG_DIR/$n"
    ((c++))
  done
done

#####################################
# 3. Detect center person bbox
#####################################
if [ ! -f "$ANN_DIR/center_bboxes.json" ]; then
    echo "=== [3] Detect center person bbox ==="
    python scripts/detect_center_bbox.py \
      --img-dir "$IMG_DIR" \
      --out-json "$ANN_DIR/center_bboxes.json"
else
    echo "[SKIP] center_bboxes.json exists"
fi

: '
#####################################
# 3.5 Visualize center person bbox
#####################################
echo "=== [3.5] Visualize center person bbox ==="
python scripts/visualize_bbox.py \
  --img-dir "$IMG_DIR" \
  --coco-json "$ANN_DIR/center_bboxes.json" \
  --vis-dir "data/bbox_vis"
'

#####################################
# 4. Generate pseudo keypoints (COCO)
#####################################
echo "=== [4] Generate pseudo keypoints ==="
python scripts/pose_to_coco.py \
  --img-dir "$IMG_DIR" \
  --bbox-json "$ANN_DIR/center_bboxes.json" \
  --pose-config "$POSE_CFG_PSEUDO" \
  --pose-checkpoint "$POSE_PRETRAIN_PSEUDO" \
  --out-train "$ANN_DIR/train.json" \
  --out-val "$ANN_DIR/val.json" \
  --out-test "$ANN_DIR/test.json" \
  --train-ratio 0.8 \
  --val-ratio 0.1

: '
#####################################
# 4.5 Visualize pseudo keypoints
#####################################
echo "=== [4.5] Visualize pseudo keypoints ==="
python scripts/visualize_coco_pose.py \
  --img-dir "$IMG_DIR" \
  --coco-json "$ANN_DIR/train.json" \
  --out-dir data/outputs_debug_vis/train

python scripts/visualize_coco_pose.py \
  --img-dir "$IMG_DIR" \
  --coco-json "$ANN_DIR/val.json" \
  --out-dir data/outputs_debug_vis/val

python scripts/visualize_coco_pose.py \
  --img-dir "$IMG_DIR" \
  --coco-json "$ANN_DIR/test.json" \
  --out-dir data/outputs_debug_vis/test
'

#####################################
# 5. Train ViTPose
#####################################
echo "=== [5] Train ViTPose ==="
python scripts/train.py \
  "$POSE_CFG_CUSTOM" \
  --work-dir "$WORK_DIR"

#####################################
# 6. Inference (finetuned)
#####################################
echo "=== [6] Inference ==="
python scripts/inferencer_vitpose_finetuned.py \
    --img-dir "$IMG_DIR" \
    --coco-json "$ANN_DIR/train.json" \
    --pose-config "$POSE_CFG_CUSTOM" \
    --pose-checkpoint "$BEST_CKPT" \
    --out-dir data/outputs_finetuned/ \
    --out-json results_train.json

python scripts/inferencer_vitpose_finetuned.py \
    --img-dir "$IMG_DIR" \
    --coco-json "$ANN_DIR/val.json" \
    --pose-config "$POSE_CFG_CUSTOM" \
    --pose-checkpoint "$BEST_CKPT" \
    --out-dir data/outputs_finetuned/ \
    --out-json results_val.json

python scripts/inferencer_vitpose_finetuned.py \
    --img-dir "$IMG_DIR" \
    --coco-json "$ANN_DIR/test.json" \
    --pose-config "$POSE_CFG_CUSTOM" \
    --pose-checkpoint "$BEST_CKPT" \
    --out-dir data/outputs_finetuned/ \
    --out-json results_test.json

####################################
# 7. Draw skeleton(train)
####################################
echo "=== [7] Draw skeleton ==="
OUT_SKEL="data/skeleton_vis/train"
mkdir -p "$OUT_SKEL"
python scripts/draw_skeleton.py \
    "$IMG_DIR" \
    "data/outputs_finetuned/results_train.json" \
    "$OUT_SKEL" \
    --bbox-json "$ANN_DIR/center_bboxes.json"

#####################################
# 8. Skeleton → Video
#####################################
echo "=== [8] Generate video ==="
ffmpeg -y \
    -framerate 25 \
    -pattern_type glob \
    -i "data/skeleton_vis/train/frame_*.jpg" \
    -c:v libx264 \
    -pix_fmt yuv420p \
    skeleton_train.mp4

echo "=== PIPELINE DONE ==="
