log_level = 'INFO'
load_from = None
resume_from = None
dist_params = dict(backend='nccl')
workflow = [('train', 1)]
checkpoint_config = dict(interval=10)
evaluation = dict(interval=10, metric='mAP')

optimizer = dict(
    type='Adam',
    lr=2e-3,
)
optimizer_config = dict(
    type="Fp16OptimizerHook",
    loss_scale="dynamic",
    grad_clip=None,
)
# learning policy
lr_config = dict(
    policy='step',
    # warmup=None,
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[170, 200],
    # step=[40, 47]
)
total_epochs = 210
# total_epochs = 50
log_config = dict(
    interval=10,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])

channel_cfg = dict(
    num_output_channels=76,
    dataset_joints=76,
    dataset_channel=[
       list(range(76))
    ],
    inference_channel=list(range(76)))

# model settings
model = dict(
    type='TopDown',
    pretrained=None,
    backbone=dict(
        type='myLiteHRNet',
        in_channels=3,
        extra=dict(
            stem=dict(stem_channels=32, out_channels=32, expand_ratio=1),
            num_stages=3,
            stages_spec=dict(
                num_modules=(2, 4, 2),
                num_branches=(2, 3, 4),
                num_blocks=(2, 2, 2),
                module_type=('LITE', 'LITE', 'LITE'),
                with_fuse=(True, True, True),
                reduce_ratios=(8, 8, 8),
                num_channels=(
                    (40, 80),
                    (40, 80, 160),
                    (40, 80, 160, 320),
                )),
            with_head=True,
        )),
    keypoint_head=dict(
        type='TopDownSimpleHead',
        in_channels=40,
        out_channels=channel_cfg['num_output_channels'],
        num_deconv_layers=0,
        extra=dict(final_conv_kernel=1, ),
    ),
    train_cfg=dict(),
    test_cfg=dict(
        flip_test=True,
        post_process=True,
        shift_heatmap=True,
        unbiased_decoding=False,
        modulate_kernel=11),
    loss_pose=dict(type='JointsMSELoss', use_target_weight=True))

data_cfg = dict(
    # image_size=[192, 256],  # TODO: inverse image size since profile foot usually has width > height
    # heatmap_size=[48, 64],
    # image_size=[768, 1024],
    # heatmap_size=[192, 256],
    image_size=[1152, 1536],
    heatmap_size=[288, 384],
    # image_size=[1536, 2048],
    # heatmap_size=[384, 512],
    num_output_channels=channel_cfg['num_output_channels'],
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'],
    soft_nms=False,
    nms_thr=1.0,
    oks_thr=0.9,
    vis_thr=0.2,
    bbox_thr=1.0,
    use_gt_bbox=False,
    image_thr=0.0,
    bbox_file='',
)

val_data_cfg = dict(
    # image_size=[192, 256],
    # heatmap_size=[48, 64],
    # image_size=[768, 1024],
    # heatmap_size=[192, 256],
    image_size=[1152, 1536],
    heatmap_size=[288, 384],
    # image_size=[1536, 2048],
    # heatmap_size=[384, 512],
    num_output_channels=channel_cfg['num_output_channels'],
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'],
    soft_nms=False,
    nms_thr=1.0,
    oks_thr=0.9,
    vis_thr=0.2,
    bbox_thr=1.0,
    use_gt_bbox=True,
    image_thr=0.0,
    bbox_file='',
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownRandomFlip', flip_prob=0.5),
    dict(
        type='TopDownGetRandomScaleRotation', rot_factor=30,
        scale_factor=0.25),
    # dict(type='TopDownGetRandomRotation90'),
    dict(type='TopDownAffine'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(type='TopDownGenerateTarget', sigma=2),
    dict(
        type='Collect',
        keys=['img', 'target', 'target_weight'],
        meta_keys=[
            'image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale',
            'rotation', 'bbox_score', 'flip_pairs'
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
        keys=[
            'img',
        ],
        meta_keys=[
            'image_file', 'center', 'scale', 'rotation', 'bbox_score',
            'flip_pairs'
        ]),
]
test_pipeline = val_pipeline
data_root = 'data/gleamer'
data = dict(
    samples_per_gpu=6,
    workers_per_gpu=4,
    pin_memory=True,
    train_dataloader=dict(
      prefetch_factor=4,    # 2 * num_workers samples prefetched across all workers
    ),
    train=dict(
        type='TopDownGleamerDataset',
        task='spine_frontal',
        ann_file=f'{data_root}/annotations/2022-04-26T08h11m31s_frontal_spine_train_coco.json',
        img_prefix=f'{data_root}/train/',
        data_cfg=data_cfg,
        pipeline=train_pipeline),
    val=dict(
        type='TopDownGleamerDataset',
        task='spine_frontal',
        ann_file=f'{data_root}/annotations/2022-04-26T08h11m32s_frontal_spine_test_coco.json',
        img_prefix=f'{data_root}/test/',
        data_cfg=val_data_cfg,
        pipeline=val_pipeline),
    test=dict(
        type='TopDownGleamerDataset',
        task='spine_frontal',
        ann_file=f'{data_root}/annotations/2022-04-26T08h11m32s_frontal_spine_test_coco.json',
        img_prefix=f'{data_root}/test/',
        data_cfg=val_data_cfg,
        pipeline=test_pipeline),
)
