_base_ = [
    "../../../configs/_base_/default_runtime.py",
    "../base/supervised_lvis_detection.py"
]

CLASSES_FILE = "annotations/lvis_v1_tail_classes337.txt"
pretrained = "https://download.pytorch.org/models/resnet101-cd907fc2.pth"

model = dict(
    type="DeformableDETR",
    freeze_exceptions=["bbox_head.cls_branches"],
    num_queries=900,
    num_feature_levels=4,
    with_box_refine=True,
    as_two_stage=True,
    data_preprocessor=dict(
        type="DetDataPreprocessor",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=1),
    backbone=dict(
        type="ResNet",
        depth=101,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=4,
        norm_cfg=dict(type="BN", requires_grad=False),
        norm_eval=True,
        style="pytorch",
        init_cfg=dict(type="Pretrained", checkpoint=pretrained)),
    neck=dict(
        type="ChannelMapper",
        in_channels=[512, 1024, 2048],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type="GN", num_groups=32),
        num_outs=4),
    encoder=dict(  # DeformableDetrTransformerEncoder
        num_layers=6,
        layer_cfg=dict(  # DeformableDetrTransformerEncoderLayer
            self_attn_cfg=dict(  # MultiScaleDeformableAttention
                embed_dims=256,
                batch_first=True),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=1024, ffn_drop=0.1))),
    decoder=dict(  # DeformableDetrTransformerDecoder
        num_layers=6,
        return_intermediate=True,
        layer_cfg=dict(  # DeformableDetrTransformerDecoderLayer
            self_attn_cfg=dict(  # MultiheadAttention
                embed_dims=256,
                num_heads=8,
                dropout=0.1,
                batch_first=True),
            cross_attn_cfg=dict(  # MultiScaleDeformableAttention
                embed_dims=256,
                batch_first=True),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=1024, ffn_drop=0.1)),
        post_norm_cfg=None),
    positional_encoding=dict(num_feats=128, normalize=True, offset=-0.5),
    bbox_head=dict(
        type="DeformableDETRHead",
        num_classes=337,
        sync_cls_avg_factor=True,
        loss_cls=dict(
            type="FocalLoss",
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type="L1Loss", loss_weight=5.0),
        loss_iou=dict(type="GIoULoss", loss_weight=2.0)),
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

train_cfg = dict(
    type="IterBasedTrainLoop", max_iters=40000, val_interval=5000)
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
load_from = "results/deformable-detr-resnet/deformable-detr-refine-twostage_r101_lvis_v1_head866/model_reset_remove.pth"
work_dir = "work_dirs/deformable-detr-resnet/deformable-detr-refine-twostage_r101_lvis_v1_tail337/"
