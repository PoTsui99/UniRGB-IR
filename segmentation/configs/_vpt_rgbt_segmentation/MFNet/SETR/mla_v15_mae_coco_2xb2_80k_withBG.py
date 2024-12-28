_base_ = [
    # '../_base_/models/setr_pup.py', 
    '../../../_base_/datasets/mfnet/mfnet_768x768_2bs.py', 
    '../../../_base_/default_runtime.py', 
    '../../../_base_/schedules/schedule_80k.py'
]

crop_size = (768, 768)
backbone_norm_cfg = dict(type='LN', eps=1e-6, requires_grad=True)
norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    type='RGBTSegDataPreProcessor',
    size=(768, 768), 
    mean_rgbt=[56.49899560214711, 65.97552241111288, 58.657336894963066, 100.82957525759328],  # channel-sequence: RGBT
    std_rgbt=[58.101488179481095, 59.048258917390804, 60.30034803081342, 47.1247675350245],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)
checkpoint = "/path/to/vitb_coco_IN1k_mae_coco_cascade-mask-rcnn_224x224_withClsToken_noRel.pth" 

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    # pretrained='pretrain/jx_vit_large_p16_384-b3be5167.pth',
    backbone=dict(
        # _delete_=True,
        type='ViTRGBTv15',
        method=None,
        img_size=crop_size[0],
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
        type='SETRMLAHead',
        in_channels=(256, 256, 256, 256),
        channels=512,
        in_index=(0, 1, 2, 3),
        dropout_ratio=0,
        mla_channels=128,
        num_classes=9,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0,
            class_weight=[1.51059098, 15.81630837, 30.08046375, 41.70549916, 
                          38.33253736, 39.78383054, 46.98874852, 46.46856094, 44.23738209])),
    auxiliary_head=[
        dict(
            type='FCNHead',
            in_channels=256,
            channels=256,
            in_index=0,
            dropout_ratio=0,
            num_convs=0,
            kernel_size=1,
            concat_input=False,
            num_classes=9,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4,
                class_weight=[1.51059098, 15.81630837, 30.08046375, 41.70549916, 
                          38.33253736, 39.78383054, 46.98874852, 46.46856094, 44.23738209])),
        dict(
            type='FCNHead',
            in_channels=256,
            channels=256,
            in_index=1,
            dropout_ratio=0,
            num_convs=0,
            kernel_size=1,
            concat_input=False,
            num_classes=9,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4,
                class_weight=[1.51059098, 15.81630837, 30.08046375, 41.70549916, 
                          38.33253736, 39.78383054, 46.98874852, 46.46856094, 44.23738209])),
        dict(
            type='FCNHead',
            in_channels=256,
            channels=256,
            in_index=2,
            dropout_ratio=0,
            num_convs=0,
            kernel_size=1,
            concat_input=False,
            num_classes=9,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4,
                class_weight=[1.51059098, 15.81630837, 30.08046375, 41.70549916, 
                          38.33253736, 39.78383054, 46.98874852, 46.46856094, 44.23738209])),
        dict(
            type='FCNHead',
            in_channels=256,
            channels=256,
            in_index=3,
            dropout_ratio=0,
            num_convs=0,
            kernel_size=1,
            concat_input=False,
            num_classes=9,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4,
                class_weight=[1.51059098, 15.81630837, 30.08046375, 41.70549916, 
                          38.33253736, 39.78383054, 46.98874852, 46.46856094, 44.23738209])),
    ],
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

default_hooks = dict(
    visualization = dict(
        interval=1
    )
)

visualizer = dict(
    alpha=0.5
)
