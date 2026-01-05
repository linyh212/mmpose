conda create -n mmpose python=3.10 -y
conda activate mmpose

pip install --upgrade pip setuptools wheel
pip install numpy<2.0 opencv-python matplotlib tqdm

pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2 --extra-index-url https://download.pytorch.org/whl/cu118
pip install mmdet==3.1.0

pip install -U openmim
mim install mmcv==2.0.0
mim install mmpretrain

pip install --upgrade pip setuptools wheel
pip install build cmake pybind11
pip install "numpy<2" opencv-python matplotlib tqdm
pip install chumpy==0.70 --no-build-isolation
pip install mmpose==1.3.2