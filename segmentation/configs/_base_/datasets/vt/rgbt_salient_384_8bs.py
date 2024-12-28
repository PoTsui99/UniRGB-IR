dataset_type = 'VTDataset'
data_root = '/path/to/Datasets/VT5000'
ori_size = (
    480,
    640,
)
crop_size = (
    384,
    384,
)
train_pipeline = [
    dict(type='LoadDualModalStackedFromFile'),
    dict(type='LoadSODGroundTruth'),
    dict(type='RandomFlipSOD', prob=0.5),
    dict(type='ResizeSOD', scale=(
        crop_size[0],
        crop_size[1],
    ), keep_ratio=False),
    dict(type='SODPackSegInputs'),
]

test_pipeline = train_pipeline  # NOTE: dummy

train_dataloader = dict(
    batch_size=8,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='VTDataset',
        data_root='/path/to/Datasets/VT5000',
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
        data_root='/path/to/Datasets/VT5000',
        data_prefix=dict(
            img_path='images/val', seg_map_path='annotations/val'),
        pipeline=test_pipeline))

val_evaluator = dict(  # NOTE: threshold influence the PR
    type='IoUMetric', iou_metrics=[
        'mIoU',
    ])
