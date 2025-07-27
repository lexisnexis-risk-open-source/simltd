_base_ = [
    "../../../configs/_base_/default_runtime.py",
    "../base/supervised_lvis_detection.py"
]

CLASSES_FILE = "annotations/lvis_v1_tail_classes337.txt"
# To train with ConvNeXt v1 backbone, use the appropriate pretrained checkpoint below.
# And set ``use_grn=False`` with ``layer_scale_init_value=1.0``.
pretrained = "https://download.openmmlab.com/mmclassification/v0/convnext-v2/convnext-v2-large_fcmae-in21k-pre_3rdparty_in1k-384px_20230104-9139a1f3.pth" # ConvNeXt v2
# pretrained = "https://download.openmmlab.com/mmclassification/v0/convnext/convnext-large_in21k-pre-3rdparty_in1k-384px_20221219-6d38dd66.pth" # ConvNeXt v1

num_levels = 5
model = dict(
    type="DINO",
    freeze_exceptions=[
        "bbox_head.cls_branches",
        "dn_query_generator.label_embedding"
    ],
    num_queries=900,
    with_box_refine=True,
    as_two_stage=True,
    num_feature_levels=num_levels,
    data_preprocessor=dict(
        type="DetDataPreprocessor",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=1),
    backbone=dict(
        type="ConvNeXt",
        arch="large",
        frozen_stages=4,
        out_indices=[0, 1, 2, 3],
        drop_path_rate=0.2,
        use_grn=True, # ConvNeXt v2
        layer_scale_init_value=0.0, # ConvNeXt v2
        gap_before_final_norm=False,
        with_cp=False,
        init_cfg=dict(
            type="Pretrained", checkpoint=pretrained,
            prefix="backbone.")),
    neck=dict(
        type="ChannelMapper",
        in_channels=[192, 384, 768, 1536],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type="GN", num_groups=32),
        num_outs=num_levels),
    encoder=dict(
        num_layers=6,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_levels=num_levels,
                               dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,
                ffn_drop=0.0))),
    decoder=dict(
        num_layers=6,
        return_intermediate=True,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_heads=8,
                               dropout=0.0),
            cross_attn_cfg=dict(embed_dims=256, num_levels=num_levels,
                                dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,
                ffn_drop=0.0)),
        post_norm_cfg=None),
    positional_encoding=dict(
        num_feats=128,
        normalize=True,
        offset=0.0,
        temperature=20),
    bbox_head=dict(
        type="DINOHead",
        num_classes=337,
        sync_cls_avg_factor=True,
        loss_cls=dict(
            type="FocalLoss",
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type="L1Loss", loss_weight=5.0),
        loss_iou=dict(type="GIoULoss", loss_weight=2.0)),
    dn_cfg=dict(
        label_noise_scale=0.5,
        box_noise_scale=1.0,
        group_cfg=dict(dynamic=True, num_groups=None,
                       num_dn_queries=100)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type="HungarianAssigner",
            match_costs=[
                dict(type="FocalLossCost", weight=2.0),
                dict(type="BBoxL1Cost", weight=5.0, box_format="xywh"),
                dict(type="IoUCost", iou_mode="giou", weight=2.0)
            ])),
    test_cfg=dict(max_per_img=300)) # LVIS allows up to 300

labeled_dataset = _base_.labeled_dataset
data_root = labeled_dataset.dataset.dataset.data_root
METAINFO = dict(classes=data_root + CLASSES_FILE)
labeled_dataset.dataset.dataset.metainfo = METAINFO

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=labeled_dataset)
val_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(metainfo=METAINFO),
)
test_dataloader = val_dataloader

# training schedule
num_iters = 40000
train_cfg = dict(
    type="IterBasedTrainLoop", max_iters=num_iters, val_interval=5000)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

# optimizer
optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(
        type="AdamW",
        lr=0.0001,
        weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
)
log_processor = dict(by_epoch=False)
default_hooks = dict(
    logger=dict(type="LoggerHook", interval=100, log_metric_by_epoch=False),
    checkpoint=dict(
        type="CheckpointHook",
        interval=5000,
        max_keep_ckpts=10,
        by_epoch=False,
    ),
)
resume = False
load_from = "work_dirs/dino-convnext/dino-5scale_convnext-l_lvis_v1_head866/model_reset_remove.pth"
work_dir = "work_dirs/dino-convnext/dino-5scale_convnext-l_lvis_v1_tail337/"
