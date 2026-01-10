# configs/vitpose_custom.py

# ================= pipeline =================
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=(192, 256)),
    dict(
        type='GenerateTarget',
        encoder=dict(
            type='UDPHeatmap',
            input_size=(192, 256),
            heatmap_size=(48, 64),
            sigma=2,
        ),
    ),
    dict(type='PackPoseInputs'),
]

# ================= basic =================
default_scope = 'mmpose'
log_level = 'INFO'
load_from = None
resume = False

# ================= hooks (VERY IMPORTANT) =================
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=10),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=2,
        save_best='AP',
        rule='greater',
        save_best_filename='best_AP.pth',
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)

# ================= env =================
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

# ================= keypoints =================
keypoint_info = {
    'nose': dict(id=0, name='nose', swap='nose'),
    'left_eye': dict(id=1, name='left_eye', swap='right_eye'),
    'right_eye': dict(id=2, name='right_eye', swap='left_eye'),
    'left_ear': dict(id=3, name='left_ear', swap='right_ear'),
    'right_ear': dict(id=4, name='right_ear', swap='left_ear'),
    'left_shoulder': dict(id=5, name='left_shoulder', swap='right_shoulder'),
    'right_shoulder': dict(id=6, name='right_shoulder', swap='left_shoulder'),
    'left_elbow': dict(id=7, name='left_elbow', swap='right_elbow'),
    'right_elbow': dict(id=8, name='right_elbow', swap='left_elbow'),
    'left_wrist': dict(id=9, name='left_wrist', swap='right_wrist'),
    'right_wrist': dict(id=10, name='right_wrist', swap='left_wrist'),
    'left_hip': dict(id=11, name='left_hip', swap='right_hip'),
    'right_hip': dict(id=12, name='right_hip', swap='left_hip'),
    'left_knee': dict(id=13, name='left_knee', swap='right_knee'),
    'right_knee': dict(id=14, name='right_knee', swap='left_knee'),
    'left_ankle': dict(id=15, name='left_ankle', swap='right_ankle'),
    'right_ankle': dict(id=16, name='right_ankle', swap='left_ankle'),
}

keypoint_names = list(keypoint_info.keys())
num_keypoints = len(keypoint_names)

# ================= skeleton_info =================
skeleton_info = {
    0: dict(link=('left_eye', 'right_eye'), id=0),
    1: dict(link=('left_shoulder', 'right_shoulder'), id=1),
    2: dict(link=('left_hip', 'right_hip'), id=2),
}

# ================= metainfo =================
metainfo = dict(
    dataset_name='custom_pose17',
    keypoint_info=keypoint_info,
    skeleton_info=skeleton_info,
    keypoint_names=keypoint_names,
    num_keypoints=num_keypoints,
    joint_weights=[1.0] * num_keypoints,
    sigmas=[0.026] * num_keypoints,
)

# ================= model =================
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
    ),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        init_cfg=None,
    ),
    head=dict(
        type='HeatmapHead',
        in_channels=2048,
        out_channels=len(metainfo['keypoint_names']),
        deconv_out_channels=(256, 256, 256),
        deconv_kernel_sizes=(4, 4, 4),
        conv_out_channels=(256, 256, len(metainfo['keypoint_names'])),
        conv_kernel_sizes=(1, 1, 1),
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        decoder=dict(
            type='UDPHeatmap',
            input_size=(192, 256),
            heatmap_size=(48, 64),
            sigma=2,
        ),
    ),
    test_cfg=dict(
        flip_test=False,
        output_heatmaps=True,
    ),
)

# dataloaders + evaluators...
train_dataloader = dict(
    batch_size=64,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CocoDataset',
        data_root='data/dataset/',
        ann_file='annotations/train.json',
        data_prefix=dict(img='images/'),
        pipeline=train_pipeline,
        metainfo=metainfo,
    ),
)

val_dataloader = dict(
    batch_size=32,
    num_workers=4,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root='data/dataset/',
        ann_file='annotations/val.json',
        data_prefix=dict(img='images/'),
        pipeline=train_pipeline,
        test_mode=True,
        metainfo=metainfo,
    ),
)

test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file='data/dataset/annotations/val.json',
)
test_evaluator = val_evaluator

train_cfg = dict(by_epoch=True, max_epochs=5, val_interval=1)
val_cfg = dict()
test_cfg = dict()

optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=0.01),
    clip_grad=dict(max_norm=1.0, norm_type=2),
)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=True,
        begin=0,
        end=1,
    )
]

auto_scale_lr = dict(base_batch_size=64)
gpu_ids = [0]

visualizer = dict(type='PoseLocalVisualizer', vis_backends=[], name='visualizer')

log_processor = dict(type='LogProcessor', window_size=10, by_epoch=True, num_digits=6)