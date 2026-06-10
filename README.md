# Dragon Boat Motion Analysis Project

## Project Overview

This capstone project focuses on **motion analysis for dragon boat athletes** using  
**computer vision** and **deep learning** techniques.

The primary objective is to build an end-to-end pipeline for:

> **Video → 2D Pose → 3D Pose → Motion Metrics Analysis**

by leveraging **video-based human detection, pose estimation, and tracking**.

---

## Project Goals

- Automatically analyze paddling techniques from race or training videos
- Provide **quantitative motion metrics** for coaches, athletes, and researchers
- Reduce reliance on manual annotation and subjective observation

---

## System Pipeline

The final workflow consists of the following stages:

1. **Paddle Detection**
   - Detect and track the paddle in video frames
   - Support stroke phase analysis (entry / pull / exit)

2. **2D Human Pose Estimation**
   - Detect human body keypoints using **ViTPose**
   - Extract consistent skeletons across video frames

3. **3D Human Pose Reconstruction**
   - Lift 2D keypoints into 3D joint coordinates
   - Enable spatial and kinematic analysis

4. **Motion Data Analysis**
   - Stroke frequency
   - Stroke distance
   - Paddle entry / exit / maximum angles
   - Joint trajectories and velocities

---

## Repository Structure
```bash
mmpose/
├── configs/
│   └── vitpose_custom.py
├── scripts/
│   ├── detect_center.py
│   ├── train.py
│   └── draw_skeleton.py
├── install.sh
└──  required.md
```
- **configs/**  
  Custom ViTPose configuration files

- **scripts/**  
  Training, inference, visualization, and utility scripts

- **install.sh**  
  Full environment setup and installation pipeline

- **required.md**  
  Additional dataset and environment requirements

---
## Model Development

### 1. Human Detection

- **Purpose:**  
  Identify human bounding boxes in each video frame

- **Method:**  
  ViTPose **top-down pipeline** with a pretrained detector  
  (e.g., **Faster R-CNN**)

- **Output:**  
  Bounding boxes used to crop human regions for pose estimation

---

### 2. 2D Human Pose Estimation

- **Purpose:**  
  Extract human body keypoints and form a skeleton representation

- **Framework:**  
  **ViTPose** within the **MMPose** framework

- **Keypoints Definition (17 joints)**

- **Processing Pipeline:**
  1. Extract frames from input videos
  2. Store frames in `data/dataset/images/` with **sequential filenames**
  3. Find the center person's bbox and Train ViTPose using a **custom config**
  4. Run inference on all frames
  5. Export keypoints as JSON files
  6. Visualize results using `draw_keypoints.py`
  7. Reassemble frames into a video using `ffmpeg`

---

### 3. 3D Human Pose Reconstruction

- **Purpose:**  
  Convert 2D keypoints into a **3D skeletal model**

- **Approach:**  
  Temporal lifting methods such as **VideoPose3D**

- **Benefits:**
  - Accurate joint trajectories
  - Stroke angle and range-of-motion analysis
  - Velocity and acceleration estimation

---

## Setup Instructions

### 1. Clone Repository
```bash
git clone https://github.com/linyh212/mmpose.git
cd ~/mmpose
```

### 2. Prepare dataset
```bash
mkdir videos
```
* Place videos into videos/
* Frames will be automatically extracted and copied to data/dataset/images/.
* Additional requirements are documented in [required.md](https://github.com/linyh212/mmpose/blob/main/required.md)

### 3. Install dependencies and start `video to 2D` process
```bash
bash install.sh
```

# Research Background
Reference material from the **Sports Science & Technology Center (運動科學與科技中心)** presentation outlines real-world metrics in dragon boat racing, including:

* Stroke frequency (spm)
* Stroke distance (cm)
* Entry, exit, and maximum angles (degrees)
* Stroke duration and recovery time (ms)

The model aims to extract similar measurements **automatically from video footage**.
