# For Dataset only, i.e. dataset_type, classes, data_root, image_size, backend_args, pipeline, data_loader
_base_ = [
    '../../../../configs/_base_/default_runtime.py',
]

dataset_type = 'DualSpectralDataset'
classes = ('car', 'person', 'bicycle')  # part of classes listed in DualSpectralDataset.Metainfo
data_root = '/path/to/Datasets/FLIR_align/'  # separator needed
image_size = (1024, 1024)  # Resize to a square, bs=1/gpu, 768~832

backend_args = None

train_pipeline = [
    dict(type='LoadAlignedImagesFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True), 
    dict(type='AlignedImagesRandomFlip', prob=0.5, direction='horizontal'),
    dict(
        type='AlignedImagesRandomResize',
        scale=image_size,
        ratio_range=(0.1, 2.0),
        keep_ratio=True),
    dict(
        type='AlignedImagesRandomCrop',
        crop_type='absolute_range',
        crop_size=image_size,
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type='AlignedImagesPad', size=image_size, pad_val=dict(img=(114, 114, 114))),
    dict(type='PackAlignedImagesDetInputs')
]

test_pipeline = [
    dict(type='LoadAlignedImagesFromFile', backend_args=backend_args),
    dict(type='AlignedImagesResize', scale=image_size, keep_ratio=False),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='PackAlignedImagesDetInputs',
        meta_keys=('img_id', 'img_path', 'img_ir_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=1,  
    num_workers=4,
    persistent_workers=True,
    # sampler=dict(type='DefaultSampler', shuffle=True),
    sampler=dict(type='InfiniteSampler', shuffle=True), 
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=dict(classes=classes),
        ann_file='Annotation_train.json',
        data_prefix=dict(img='train/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=dict(classes=classes),
        ann_file='Annotation_test.json',
        data_prefix=dict(img='test/'),
        test_mode=True,
        pipeline=test_pipeline))
test_dataloader = val_dataloader

