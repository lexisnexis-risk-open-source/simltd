custom_imports = dict(
    imports=[
        "projects.MixPL.mixpl",
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

geometric = [
    [dict(type="Rotate")],
    [dict(type="ShearX")],
    [dict(type="ShearY")],
    [dict(type="TranslateX")],
    [dict(type="TranslateY")],
]

scale = [(1333, 400), (1333, 1000)]

branch_field = ["sup", "unsup_teacher", "unsup_student"]
# pipeline used to augment labeled data,
# which will be sent to student model for supervised training.
load_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
    dict(type="FilterAnnotations", min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type="RandomFlip", prob=0.5),
]
sup_pipeline = [
    dict(type="CopyPaste", max_num_pasted=100),
    dict(type="RandomResize", scale=scale, keep_ratio=True),
    dict(type="RandAugment", aug_space=color_space, aug_num=1),
    dict(
        type="MultiBranch",
        branch_field=branch_field,
        sup=dict(type="PackDetInputs"))
]

# pipeline used to augment unlabeled data weakly,
# which will be sent to teacher model for predicting pseudo instances.
weak_pipeline = [
    dict(type="RandomResize", scale=scale, keep_ratio=True),
    dict(type="RandomFlip", prob=0.5),
    dict(
        type="PackDetInputs",
        meta_keys=("img_id", "img_path", "ori_shape", "img_shape",
                   "scale_factor", "flip", "flip_direction",
                   "homography_matrix")),
]

# pipeline used to augment unlabeled data strongly,
# which will be sent to student model for unsupervised training.
strong_pipeline = [
    dict(type="RandomResize", scale=scale, keep_ratio=True),
    dict(type="RandomFlip", prob=0.5),
    dict(
        type="RandomOrder",
        transforms=[
            dict(type="RandAugment", aug_space=color_space, aug_num=1),
            dict(type="RandAugment", aug_space=geometric, aug_num=1),
        ]),
    dict(
        type="PackDetInputs",
        meta_keys=("img_id", "img_path", "ori_shape", "img_shape",
                   "scale_factor", "flip", "flip_direction",
                   "homography_matrix")),
]

# pipeline used to augment unlabeled data into different views
unsup_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadEmptyAnnotations"),
    dict(
        type="MultiBranch",
        branch_field=branch_field,
        unsup_teacher=weak_pipeline,
        unsup_student=strong_pipeline,
    )
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(1333, 800), keep_ratio=True),
    dict(
        type="PackDetInputs",
        meta_keys=("img_id", "img_path", "ori_shape", "img_shape",
                   "scale_factor"))
]

batch_size = 6
num_workers = 6

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
    pipeline=sup_pipeline)

unlabeled_dataset = dict(
    type="CocoDataset",
    data_root=data_root,
    ann_file="annotations/instances_unlabeled2017.json",
    data_prefix=dict(img="unlabeled2017/"),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=unsup_pipeline)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    sampler=dict(
        type="GroupMultiSourceSampler",
        batch_size=batch_size,
        source_ratio=[2, 4]),
    dataset=dict(
        type="ConcatDataset", datasets=[labeled_dataset, unlabeled_dataset]))

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
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
