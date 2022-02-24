_base_ = ['../../../../_base_/datasets/gaze.py']
log_level = 'INFO'
load_from = None
resume_from = None
dist_params = dict(backend='nccl')
workflow = [('train', 1)]
checkpoint_config = dict(interval=10)
evaluation = dict(interval=5, metric=['NME'], save_best='NME')

optimizer = dict(
    type='Adam',
    lr=2e-3,
)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[40, 55])
total_epochs = 100
log_config = dict(
    interval=5,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])

channel_cfg = dict(
    num_output_channels=17,
    dataset_joints=17,
    dataset_channel=[
        list(range(17)),
    ],
    inference_channel=list(range(17)))

# model settings
model = dict(
    type='RMTopDown',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='RMResNet', 
        depth=50,
        frozen_parameters=False),
    loss_cfg=dict(
        keypoint = True,
        gaze = False,
    ),
    keypoint_head=dict(
        type='RMTopdownHeatmapSimpleHead',
        in_channels=2048,
        out_channels=channel_cfg['num_output_channels'],
        frozen_parameters=False,
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
    gaze_head=dict(
        type='RMFcSimpleHead',
        in_channels=2048,
        out_channels=2,
        frozen_parameters=True,
        loss_gaze=dict(type='RMGazeMSELoss')),
    train_cfg=dict(),
    test_cfg=dict(
        flip_test=True,
        post_process='default',
        shift_heatmap=True,
        modulate_kernel=11))

data_cfg = dict(
    image_size=[256, 256],
    heatmap_size=[64, 64],
    num_output_channels=channel_cfg['num_output_channels'],
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'])

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownRandomFlip', flip_prob=0.5),
    dict(
        type='TopDownGetRandomScaleRotation', rot_factor=30,
        scale_factor=0.25),
    dict(type='TopDownAffine'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(type='TopDownGenerateTarget', sigma=1.5),
    dict(
        type='Collect',
        keys=['img', 'target', 'target_weight', 'gaze'],
        meta_keys=[
            'image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale',
            'rotation', 'flip_pairs'
        ]),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownAffine'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=['image_file', 'center', 'scale', 'rotation', 'flip_pairs']),
]

test_pipeline = val_pipeline

data_train_root = '/jiangtao2/dataset/train/gazePoints/'
data_test_root = '/jiangtao2/dataset/train/gazePoints/'
data = dict(
    samples_per_gpu=128,
    workers_per_gpu=2,
    val_dataloader=dict(samples_per_gpu=32),
    test_dataloader=dict(samples_per_gpu=32),
    train=dict(
        type='FaceGazeDataset',
        ann_file=f'{data_train_root}/gazePoints_train.json',
        img_prefix=f'{data_train_root}',
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        dataset_info={{_base_.dataset_info}}),
    val=dict(
        type='FaceGazeDataset',
        ann_file=f'{data_test_root}/gazePoints_val.json',
        img_prefix=f'{data_test_root}',
        data_cfg=data_cfg,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}}),
    test=dict(
        type='FaceGazeDataset',
        ann_file=f'{data_test_root}/gazePoints_val.json',
        img_prefix=f'{data_test_root}',
        data_cfg=data_cfg,
        pipeline=test_pipeline,
        dataset_info={{_base_.dataset_info}}),
)
