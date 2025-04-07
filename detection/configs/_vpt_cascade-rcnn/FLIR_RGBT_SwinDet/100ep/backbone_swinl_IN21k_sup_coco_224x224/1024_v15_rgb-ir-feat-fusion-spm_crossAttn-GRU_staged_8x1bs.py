# default: 4 stages
_base_ = [
    '../../../../_base_/datasets/flir(aligned)/flir_dual_LSJ_1024_1bs.py',
]

custom_imports = dict(imports=['projects.Swin.swindet'], allow_failed_imports=False)

backbone_norm_cfg = dict(type='LN', requires_grad=True)
norm_cfg = dict(type='LN2d', requires_grad=True)
image_size = (1024, 1024)
batch_augments = [
    dict(type='BatchFixedSizePad', size=image_size, pad_mask=True)
]

# model settings, setting mostly inherits from cascade-rcnn_r50_fpn
model = dict(
    type='TwoStreamTwoStageDetectorFuseBeforeFPN',
    data_preprocessor=dict(
        type='DualSteramDetDataPreprocessor',
        mean=[159.8808906080302, 162.22057018543336, 160.28301196773916],
        mean_ir=[136.63746562356317, 136.63746562356317, 136.63746562356317],  # IR as 3-channel input
        std=[56.96897676312916, 59.57937492901139, 63.11906486423505],
        std_ir=[64.97730349740912, 64.97730349740912, 64.97730349740912],  
        bgr_to_rgb=True,
        pad_size_divisor=32,
        batch_augments=batch_augments),  # NOTE: batch augmentation
    backbone=dict(
        type='SwinRGBTv15',
        pretrain_img_size=224,
        patch_size=4,
        in_chans=3,
        embed_dim=192,  # Swin-L
        depths=(2, 2, 18, 2),  # Swin-L
        num_heads=(6, 12, 24, 48),  # Swin-L
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.3,
        norm_layer='LN',
        ape=False, # TSUIPO: not using absolute position embedding
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        use_checkpoint=False,
        # For RGBT support
        conv_inplane=64,
        n_points=4,
        deform_num_heads=12,
        deform_ls_init_values=0.0,
        deform_ratio=0.5,
        adapter_dim=768,  # 明确设置adapter维度为768，与stage3维度相同（对应Swin-L）
        init_cfg=dict(
            type='Pretrained', checkpoint="/path/to/swinl_IN21k_sup_coco_cascade-mask-rcnn_224x224.pth")
    ), 
    neck=dict(  # Modified to use SwinFPN instead of SimpleFPN
        type='SwinFPN',
        in_channels=[192, 384, 768, 1536],  # Swin-L channels
        out_channels=256,
        num_outs=5,
        norm_cfg=norm_cfg,
        add_ln_norm=True,
        use_residual=True,
        fuse_type='sum',
        top_block=dict(type='LastLevelMaxPool')),
    rpn_head=dict(
        type='RPNHead',
        num_convs=2,
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    roi_head=dict(
        type='CascadeRoIHead',
        num_stages=3,
        stage_loss_weights=[1, 0.5, 0.25],  # weight for loss from 3 stage
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            # output fixed featmap size: (7x7)
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=[
            dict(
                # with 4 shared convolution layer, plus a fully connected layer
                type='Shared4Conv1FCBBoxHead',  # modified
                in_channels=256,
                conv_out_channels=256,  # added
                norm_cfg=norm_cfg,  # added
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=3,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared4Conv1FCBBoxHead',  # modified
                in_channels=256,
                conv_out_channels=256,  # added
                norm_cfg=norm_cfg,  # added
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=3,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared4Conv1FCBBoxHead',  # modified
                in_channels=256,
                conv_out_channels=256,  # added
                norm_cfg=norm_cfg,  # added
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=3,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
        ]),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=[
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,  # matcher
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.6,
                    min_pos_iou=0.6,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.7,
                    min_pos_iou=0.7,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False)
        ]),
    test_cfg=dict(
        rpn=dict(
            nms_pre=2000,  # before: 1000
            max_per_img=2000,  # before: 1000
            nms=dict(type='nms', iou_threshold=0.8),  # before: 0.7
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=2000)))  # before: 100

data_root = '/path/to/Datasets/FLIR_align/'  # with separator '/'
# TODO: add MR^{-1} metric.
val_evaluator = dict(  
    type='CocoMetric',
    ann_file=data_root + 'Annotation_test.json',
    metric=['bbox'], 
    format_only=False
)

test_evaluator = val_evaluator

optim_wrapper = dict(
    type='AmpOptimWrapper',
    constructor='LayerDecayOptimizerConstructor',
    paramwise_cfg={
        'decay_rate': 0.9,  # Adjusted for Swin-L (larger model)
        'decay_type': 'layer_wise',
        'num_layers': 24,  # Adjusted for Swin-L (sum of depths)
    },
    optimizer=dict(
        type='AdamW',
        lr=0.0001,  # basic lr
        betas=(0.9, 0.999),
        weight_decay=0.1,
    ),
)

# iters = 100ep * 4129 imges / (8gpu * 1)
# FLIR: train - 4129, val - 1013
max_iters = 51613
interval = 500
dynamic_intervals = [(max_iters // interval * interval + 1, max_iters)]
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=250),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_iters,
        by_epoch=False,
        # 88%, 96% as milestones
        milestones=[45419, 49548],
        gamma=0.1)
]

train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=max_iters,
    val_interval=interval,
    dynamic_intervals=dynamic_intervals)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        save_last=True,
        save_best='coco/bbox_mAP',
        interval=interval,
        max_keep_ckpts=2,
    )
)

custom_hooks = [dict(type='Fp16CompresssionHook')]

log_processor = dict(type='LogProcessor', window_size=50, by_epoch=False)

auto_scale_lr = dict(base_batch_size=64, enable=True) 