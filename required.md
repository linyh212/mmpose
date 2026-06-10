# MMPose 1.3.2 + CUDA 11.8 Installation and Setup Guide

This document outlines the complete and reproducible installation process of **MMPose 1.3.2** under **Python 3.10 + CUDA 11.8**, and corrects common version conflicts and repetitive steps.

---

## 1. Set up the Conda environment (Python 3.10)

```bash
conda create -n mmpose(*yourname) python=3.10 -y
conda activate mmpose(*yourname)
```

## 2. Install MMPose (fixed version)

```bash
pip install mmpose==1.3.2
```

## 3. Upgrade basic tools

```bash
pip install --upgrade pip setuptools wheel
```

## 4. Install PyTorch (CUDA 11.8)

```bash
pip install torch==2.0.1+cu118 \
torchvision==0.15.2+cu118 \
torchaudio==2.0.2 \
--extra-index-url https://download.pytorch.org/whl/cu118
```

## 5. Install the OpenMMLab core suite

```bash
pip install -U openmim
mim install mmcv==2.0.0
pip install mmdet==3.1.0
mim install mmpretrain
```

## 6. Other necessary packages (avoid numpy 2.x)

```bash
pip install "numpy<2.0" opencv-python matplotlib tqdm
pip install build cmake pybind11
pip install chumpy==0.70 --no-build-isolation
```

## 7. Download the MMPose source code

```bash
git clone https://github.com/open-mmlab/mmpose.git
```
## 8. Download the Detection Model weights (Faster R-CNN)

```bash
mmpose/
├── weights/
│ └── faster_rcnn_r50_fpn_coco.pth
```
```bash
mkdir -p weights

wget -O weights/faster_rcnn_r50_fpn_coco.pth \
https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth

wget https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
```

## 9. Download Data Set

- [DropBox All videos](https://www.dropbox.com/scl/fo/12hfh5dxb8d6tw7ge170m/AGkPzdb-sgANJzbdiyv6R-M?rlkey=k8mclge4bhvt31i528ak9dvyg&st=peqtwp4e&dl=0)

```bash
mmpose/
├── videos/
│ └── A01何耀榮.mp4
│ └── A02 田豐源.mp4
│  ...
│  ...
│  ...
```

## 10. Final environment check
```bash
python - << EOF
import torch
import mmcv
import mmdet
import mmpose
print("torch:", torch.__version__)
print("mmcv:", mmcv.__version__)
print("mmdet:", mmdet.__version__)
print("mmpose:", mmpose.__version__)
print("cuda:", torch.cuda.is_available())
EOF
```
```bash
torch: 2.0.1+cu118
mmcv: 2.0.0
mmdet: 3.1.0
mmpose: 1.3.2
cuda: True
```
