# TODO: add SSIM / IoU loss
_base_ = [
    '../../_base_/datasets/vt/rgbt_salient_416_8bs.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_20k_sod.py'
]
# img_size / 16 % 2 == 0
backbone_norm_cfg = dict(type='LN', eps=1e-6, requires_grad=True)
norm_cfg = dict(type='SyncBN', requires_grad=True)
# NOTE: Normalize when preprocessing data (not played by dataloader)
# key: img(H, W, C)
data_preprocessor = dict(
    type='SODSegDataPreProcessor',
    mean=[135.73938737109376, 157.8530516015625, 140.7335156080729],
    mean_ir=[192.20718705729166, 92.03320703776042, 87.13236662239584],
    std=[62.22105260241252, 60.33881855612459, 61.503459813819916],
    std_ir=[61.554071856663306, 68.75750769340034, 53.08267291838843],
    bgr_to_rgb=True,
    size_divisor=16,
    pad_val=0,
    seg_pad_val=255
)
checkpoint = '/path/to/vitb_coco_IN1k_mae_coco_cascade-mask-rcnn_224x224_withClsToken_noRel.pth'
test_data_root = '/path/to/Datasets/VT5000/Test'

crop_size = (416, 416)
image_size = crop_size

model = dict(
    type='SODEncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='ViTRGBTv15_SOD',
        img_size=crop_size[0],  # HACK: modify input images' size here
        stage_ranges=[[0, 2], [3, 5], [6, 8], [9, 11]],
        conv_inplane=64,
        n_points=4,
        deform_num_heads=12,
        # cffn_ratio=0.25,
        deform_ratio=0.5,
        patch_size=16,
        in_chans=3, # refer to the channnel of rgb modal
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        # layer_scale_init_values=None,
        # scale_factor=12,
        # drop_path_rate=0.1,
        drop_path_rate=0.3, 
        # deform_norm_layer=partial(nn.LayerNorm, eps=1e-6),
        norm_cfg=backbone_norm_cfg,
        window_size=14,  # windowed attention
        pretrain_img_size=224,
        pretrain_use_cls_token=True,
        window_block_indexes=[
            0,
            1,
            3,
            4,
            6,
            7,
            9,
            10,
        ],
        # use_rel_pos=False, 
        use_rel_pos=True,
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint)), 
    neck=dict(
        type='MLANeck',
        in_channels=[768, 768, 768, 768],
        # in_channels=[1024, 1024, 1024, 1024],
        out_channels=256,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='ReLU'),
    ),
    decode_head=dict(
        type='SODSETRMLAHead',
        in_channels=(256, 256, 256, 256),
        channels=512,
        in_index=(0, 1, 2, 3),
        dropout_ratio=0,
        mla_channels=128,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
            dict(type='SODIoULoss', loss_weight=1.0),
            dict(type='SODSSIMLoss', loss_weight=1.0)
            ]),
    auxiliary_head=[
        dict(
            type='SODFCNHead',
            in_channels=256,
            channels=256,
            in_index=0,
            dropout_ratio=0,
            num_convs=0,
            kernel_size=1,
            concat_input=False,
            align_corners=False,
            loss_decode=[
                dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.4),
                dict(type='SODIoULoss', loss_weight=0.4),
                dict(type='SODSSIMLoss', loss_weight=0.4)
            ]),
        dict(
            type='SODFCNHead',
            in_channels=256,
            channels=256,
            in_index=1,
            dropout_ratio=0,
            num_convs=0,
            kernel_size=1,
            concat_input=False,
            align_corners=False,
            loss_decode=[
                dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.4),
                dict(type='SODIoULoss', loss_weight=0.4),
                dict(type='SODSSIMLoss', loss_weight=0.4)
            ]),
        dict(
            type='SODFCNHead',
            in_channels=256,
            channels=256,
            in_index=2,
            dropout_ratio=0,
            num_convs=0,
            kernel_size=1,
            concat_input=False,
            align_corners=False,
            loss_decode=[
                dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.4),
                dict(type='SODIoULoss', loss_weight=0.4),
                dict(type='SODSSIMLoss', loss_weight=0.4)
            ]),
        dict(
            type='SODFCNHead',
            in_channels=256,
            channels=256,
            in_index=3,
            dropout_ratio=0,
            num_convs=0,
            kernel_size=1,
            concat_input=False,
            align_corners=False,
            loss_decode=[
                dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.4),
                dict(type='SODIoULoss', loss_weight=0.4),
                dict(type='SODSSIMLoss', loss_weight=0.4)
            ]),
    ],
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(100, 100)),
)

# val_dataloader=None
# val_evaluator=None
# val_cfg=None

# for VT1000
# vis_suffix='.jpg',
# ir_suffix='.jpg',
# gt_suffix='.jpg'  
# for VT821
# vis_suffix='.jpg',
# ir_suffix='.jpg',
# gt_suffix='.jpg'  
# for VT723
# vis_suffix='.png',
# ir_suffix='.png',
# gt_suffix='.png'  
# for VT5000
# vis_suffix = '.jpg',
# ir_suffix = '.jpg',
# gt_suffix = '.png'  

test_dataloader = dict(
    batch_size=1,
    num_workers=12,
    persistent_workers=True,
    dataset=dict(
        type='VTDataset',
        # data_root='/path/to/Datasets/VT723',
        # data_root='/path/to/Datasets/VT821',
        # data_root='/path/to/Datasets/VT1000',
        data_root=test_data_root,
        data_prefix=dict(  # root of images
            vis_path='RGB', 
            ir_path='T', 
            gt_path='GT'
        ),
        vis_suffix='.jpg',
        ir_suffix='.jpg',
        gt_suffix='.png',
        pipeline=[
            # load vis and ir, then stack them as **6-channel** picture
            # NOTE: vis + ir stacked to be img heare
            dict(type='LoadDualModalStackedFromFile'),  
            dict(type='LoadSODGroundTruth'),  # float32
            # Flip is not required
            # dict(type='RandomFlipSOD', prob=0.5),
            dict(type='ResizeSOD', scale=image_size),
            # dict(type='RandomCropSOD', crop_size=image_size),
            dict(
                type='SODPackSegInputs',
                meta_keys=('vis_path', 'ir_path', 'ori_shape', 'img_shape',
                        'flip', 'flip_direction')
            )
        ]
    )
)

test_cfg = dict(type='TestLoop')
test_evaluator = dict(
    type='MAEMetric',
)
