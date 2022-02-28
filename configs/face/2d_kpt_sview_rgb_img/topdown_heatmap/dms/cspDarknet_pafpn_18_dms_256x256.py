_base_ = ['../../../../_base_/datasets/dms.py']
log_level = 'INFO'
load_from = None
resume_from = None
dist_params = dict(backend='nccl')
workflow = [('train', 1)]
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metric=['NME'], save_best='NME')

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
    num_output_channels=21,
    dataset_joints=21,
    dataset_channel=[
        list(range(21)),
    ],
    inference_channel=list(range(21)))

# model settings
model = dict(
    type='TopDown',
    pretrained=None,
    backbone=dict(
        type='RMCSPDarknet', 
        frozen_backbone=False, 
        use_depthwise=False,
        deepen_factor=0.33, 
        widen_factor=0.25,
        act_cfg=dict(type='ReLU')),
    neck=dict(
        type='YOLOXPAFPN',
        in_channels=[64, 128, 256],
        out_channels=64,
        num_csp_blocks=1,
        use_depthwise=False,
        act_cfg=dict(type='ReLU')),
    keypoint_head=dict(
        type='TopdownHeatmapSimpleHead',
        in_channels=64,
        out_channels=channel_cfg['num_output_channels'],
        num_deconv_layers=1,
        num_deconv_filters=(64,),
        num_deconv_kernels=(2,),
        extra=dict(final_conv_kernel=1,
                    num_conv_layers=1,
                    num_conv_kernels=[3,]),
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
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
        keys=['img', 'target', 'target_weight'],
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

data_train_root = '/jiangtao2/dataset/train/alignment/train/'
data_test_root = '/jiangtao2/dataset/train/alignment/test/'
data = dict(
    samples_per_gpu=128,
    workers_per_gpu=16,
    val_dataloader=dict(samples_per_gpu=32),
    test_dataloader=dict(samples_per_gpu=32),
    train=dict(
        type='FaceDMSDataset',
        ann_file=f'{data_train_root}/face_landmarks_dms_train.json',
        img_prefix=f'{data_train_root}',
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        dataset_info={{_base_.dataset_info}}),
    val=dict(
        type='FaceDMSDataset',
        ann_file=f'{data_test_root}/face_landmarks_dms_test.json',
        img_prefix=f'{data_test_root}',
        data_cfg=data_cfg,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}}),
    test=dict(
        type='FaceDMSDataset',
        ann_file=f'{data_test_root}/face_landmarks_dms_test.json',
        img_prefix=f'{data_test_root}',
        data_cfg=data_cfg,
        pipeline=test_pipeline,
        dataset_info={{_base_.dataset_info}}),
)
