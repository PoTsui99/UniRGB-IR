dataset_type = 'MFNetRGBADataset'
data_root = '/path/to/Datasets/MFNet_mmseg'
ori_size = (
    480,
    640,
)
crop_size = (
    768,
    768,
)
train_pipeline = [
    dict(type='LoadImageFromFile', color_type='unchanged'),
    dict(type='LoadAnnotations'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Resize', scale=(
        crop_size[0],
        crop_size[1],
    ), keep_ratio=False),
    dict(type='PackSegInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile', color_type='unchanged'),
    dict(type='Resize', scale=(
        crop_size[0],
        crop_size[1],
    ), keep_ratio=False),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]
img_ratios = [
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
    1.75,
]
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=0.5, keep_ratio=True),
                dict(type='Resize', scale_factor=0.75, keep_ratio=True),
                dict(type='Resize', scale_factor=1.0, keep_ratio=True),
                dict(type='Resize', scale_factor=1.25, keep_ratio=True),
                dict(type='Resize', scale_factor=1.5, keep_ratio=True),
                dict(type='Resize', scale_factor=1.75, keep_ratio=True),
            ],
            [
                dict(type='RandomFlip', prob=0.0, direction='horizontal'),
                dict(type='RandomFlip', prob=1.0, direction='horizontal'),
            ],
            [
                dict(type='LoadAnnotations'),
            ],
            [
                dict(type='PackSegInputs'),
            ],
        ]),
]
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='MFNetRGBADataset',
        data_root='/path/to/Datasets/MFNet_mmseg',
        reduce_zero_label=False,
        data_prefix=dict(
            img_path='images/train', seg_map_path='annotations/train'),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='MFNetRGBADataset',
        data_root='/path/to/Datasets/MFNet_mmseg',
        data_prefix=dict(
            img_path='images/val', seg_map_path='annotations/val'),
        pipeline=test_pipeline))

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='MFNetRGBADataset',
        data_root='/path/to/Datasets/MFNet_mmseg',
        data_prefix=dict(
            img_path='images/val', seg_map_path='annotations/val'),
        pipeline=test_pipeline))

val_evaluator = dict(
    type='IoUMetric', iou_metrics=[
        'mIoU',
    ])
test_evaluator = dict(
    type='IoUMetric', iou_metrics=[
        'mIoU',
    ])