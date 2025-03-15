custom_imports = dict(
    imports=[
        "projects.SimLTD.simltd",
        "mmdet.datasets.lvis_v1",
        "mmdet.evaluation.metrics.lvis_v1_metric",
        "mmdet.datasets.transforms.simple_copy_paste",
    ], allow_failed_imports=False)

# dataset settings
dataset_type = "LVISV1Dataset"
data_root = "data/coco/"

color_space = [
    [dict(type="ColorTransform")],
    [dict(type="AutoContrast")],
    [dict(type="Equalize")],
    [dict(type="Sharpness")],
    [dict(type="Posterize")],
    [dict(type="Solarize")],
    [dict(type="Color")],
    [dict(type="Contrast")],
    [dict(type="Brightness")],
]

scale = [(1333, 400), (1333, 1200)]

load_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
    dict(type="FilterAnnotations", min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type="RandomFlip", prob=0.5),
]
train_pipeline = [
    dict(type="CopyPaste", max_num_pasted=100),
    dict(type="RandomResize", scale=scale, keep_ratio=True),
    dict(type="RandAugment", aug_space=color_space, aug_num=1),
    dict(type="PackDetInputs")
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(1333, 800), keep_ratio=True),
    dict(
        type="PackDetInputs",
        meta_keys=("img_id", "img_path", "ori_shape", "img_shape",
                   "scale_factor"))
]

batch_size = 2
num_workers = 2

labeled_dataset = dict(
    type="MultiImageMixDataset",
    dataset=dict(
        type="ClassBalancedDataset",
        oversample_thr=1e-3,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file="annotations/lvis_v1_train.json",
            data_prefix=dict(img=""),
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            pipeline=load_pipeline,
        )),
    pipeline=train_pipeline)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    sampler=dict(type="InfiniteSampler", shuffle=True),
    batch_sampler=dict(type="AspectRatioBatchSampler"),
    dataset=labeled_dataset)

val_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="annotations/lvis_v1_val.json",
        data_prefix=dict(img=""),
        test_mode=True,
        pipeline=test_pipeline))

test_dataloader = val_dataloader

val_evaluator = dict(
    type="LVISV1Metric",
    ann_file=data_root + "annotations/lvis_v1_val.json",
    metric="bbox",
    format_only=False)
test_evaluator = val_evaluator
