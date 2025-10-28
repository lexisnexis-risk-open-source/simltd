# Modified from
# https://github.com/ucbdrive/few-shot-object-detection/blob/master/tools/ckpt_surgery.py
import argparse
import os
import torch


# Module names for the supervised Deformable DETR architecture.
DDETR_MODULE_NAMES = [
    "bbox_head.cls_branches.0",
    "bbox_head.cls_branches.1",
    "bbox_head.cls_branches.2",
    "bbox_head.cls_branches.3",
    "bbox_head.cls_branches.4",
    "bbox_head.cls_branches.5",
    "bbox_head.cls_branches.6",
]

# Module names for the supervised DINO architecture.
DINO_MODULE_NAMES = [
    "bbox_head.cls_branches.0",
    "bbox_head.cls_branches.1",
    "bbox_head.cls_branches.2",
    "bbox_head.cls_branches.3",
    "bbox_head.cls_branches.4",
    "bbox_head.cls_branches.5",
    "bbox_head.cls_branches.6",
    "dn_query_generator.label_embedding",
]

# Module names for the semi-supervised DINO architecture.
SEMI_DINO_MODULE_NAMES = [
    "student.bbox_head.cls_branches.0",
    "student.bbox_head.cls_branches.1",
    "student.bbox_head.cls_branches.2",
    "student.bbox_head.cls_branches.3",
    "student.bbox_head.cls_branches.4",
    "student.bbox_head.cls_branches.5",
    "student.bbox_head.cls_branches.6",
    "teacher.bbox_head.cls_branches.0",
    "teacher.bbox_head.cls_branches.1",
    "teacher.bbox_head.cls_branches.2",
    "teacher.bbox_head.cls_branches.3",
    "teacher.bbox_head.cls_branches.4",
    "teacher.bbox_head.cls_branches.5",
    "teacher.bbox_head.cls_branches.6",
    "student.dn_query_generator.label_embedding",
    "teacher.dn_query_generator.label_embedding",
]

# Default module names.
MODULE_NAMES_DICT = {
    "ddetr"    : DDETR_MODULE_NAMES,
    "dino"     : DINO_MODULE_NAMES,
    "semi_dino": SEMI_DINO_MODULE_NAMES,
}


def parse_args():
    parser = argparse.ArgumentParser()
    # Paths
    parser.add_argument(
        "--head-ckpt",
        type=str,
        default="",
        help="Path to the head model checkpoint.")
    parser.add_argument(
        "--tail-ckpt",
        type=str,
        default="",
        help="Path to the tail model checkpoint (for merging).")
    parser.add_argument(
        "--head-ids",
        type=str,
        default="data/coco/annotations/lvis_v1_head_ids866.txt",
        help="Path to the head class IDs.")
    parser.add_argument(
        "--tail-ids",
        type=str,
        default="data/coco/annotations/lvis_v1_tail_ids337.txt",
        help="Path to the tail class IDs.")
    parser.add_argument(
        "--save-dir", type=str, default="", help="Save directory.")
    # Surgery methods
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["combine", "remove", "randinit"],
        help="Surgery method. `combine`: "
        "merge checkpoints. `remove`: for fine-tuning on "
        "tail dataset, remove the final layer of the "
        "head detector. `randinit`: randomly initialize "
        "weights for the tail detector.")
    parser.add_argument(
        "--keep-student-teacher",
        action="store_true",
        help="Perform student-teacher semi-supervised multi-branch surgery.")
    parser.add_argument(
        "--with-bg",
        action="store_true",
        help="Whether the detector has the background class. Default to False.")
    # Targets
    parser.add_argument(
        "--target-name",
        type=str,
        default="model_reset",
        help="Name of the new checkpoint.")
    parser.add_argument(
        "--module-names",
        type=str,
        required=True,
        choices=["ddetr", "dino", "semi_dino"],
        help="Module names for surgery.")
    args = parser.parse_args()
    return args


def checkpoint_surgery(args):
    """Either remove the final layer weights for fine-tuning on tail dataset or
    append randomly initialized weights for the tail classes."""

    def surgery_func(module_name, is_weight, target_size, head_ckpt, tail_ckpt):
        weight_name = module_name + (".weight" if is_weight else ".bias")
        head_weight = head_ckpt["state_dict"][weight_name]
        if is_weight:
            feat_size = head_weight.size(1)
            new_weight = torch.rand((target_size, feat_size))
            torch.nn.init.normal_(new_weight, 0, 0.01)
        else:
            new_weight = torch.zeros(target_size)
        for label, ID in enumerate(HEAD_IDS):
            new_weight[ID2LABEL[ID]] = head_weight[label]
        if "cls" in module_name and args.with_bg:
            new_weight[-1] = head_weight[-1]  # background class
        head_ckpt["state_dict"][weight_name] = new_weight

    surgery_loop(args, surgery_func)


def combine_checkpoints(args):
    """Combine head detector with tail detector. Feature extractor weights are
    from the head detector. Only the final layer weights are combined."""

    def surgery_func(module_name, is_weight, target_size, head_ckpt, tail_ckpt):
        if not is_weight and module_name + ".bias" not in head_ckpt["state_dict"]:
            return
        weight_name = module_name + (".weight" if is_weight else ".bias")
        head_weight = head_ckpt["state_dict"][weight_name]
        if is_weight:
            feat_size = head_weight.size(1)
            new_weight = torch.rand((target_size, feat_size))
        else:
            new_weight = torch.zeros(target_size)
        for label, ID in enumerate(HEAD_IDS):
            new_weight[ID2LABEL[ID]] = head_weight[label]
        tail_weight = tail_ckpt["state_dict"][weight_name]
        for label, ID in enumerate(TAIL_IDS):
            new_weight[ID2LABEL[ID]] = tail_weight[label]
        if "cls" in module_name and args.with_bg:
            new_weight[-1] = head_weight[-1]  # background class
        head_ckpt["state_dict"][weight_name] = new_weight

    surgery_loop(args, surgery_func)


def surgery_loop(args, surgery_func):
    # Load checkpoints.
    head_ckpt = torch.load(args.head_ckpt, map_location=torch.device("cpu"))
    if not args.keep_student_teacher:
        head_ckpt = extract_teacher(head_ckpt)
    if args.method == "combine":
        tail_ckpt = torch.load(args.tail_ckpt, map_location=torch.device("cpu"))
        if not args.keep_student_teacher:
            tail_ckpt = extract_teacher(tail_ckpt)
        save_name = args.target_name + "_combine.pth"
    else:
        tail_ckpt = None
        save_name = (
            args.target_name
            + "_"
            + ("remove" if args.method == "remove" else "surgery")
            + ".pth"
        )
    if args.save_dir == "":
        # By default, save to directory of `head_ckpt`.
        save_dir = os.path.dirname(args.head_ckpt)
    else:
        save_dir = args.save_dir
    save_path = os.path.join(save_dir, save_name)
    os.makedirs(save_dir, exist_ok=True)
    reset_checkpoint(head_ckpt)

    # Remove parameters.
    if args.method == "remove":
        for module_name in args.module_names:
            del head_ckpt["state_dict"][module_name + ".weight"]
            if module_name + ".bias" in head_ckpt["state_dict"]:
                del head_ckpt["state_dict"][module_name + ".bias"]
        save_checkpoint(head_ckpt, save_path)
        return

    target_size = TARGET_SIZE
    if args.with_bg:
        target_size = TARGET_SIZE + 1
    
    for module_name in args.module_names:
        if "dn_query" in module_name:
            # Modify weights only with `is_weight=True`.
            surgery_func(
                module_name=module_name,
                is_weight=True,
                target_size=target_size,
                head_ckpt=head_ckpt,
                tail_ckpt=tail_ckpt
            )
        else:
            # Modify weights with `is_weight=True`.
            surgery_func(
                module_name=module_name,
                is_weight=True,
                target_size=target_size,
                head_ckpt=head_ckpt,
                tail_ckpt=tail_ckpt
            )
            # Modify biases with `is_weight=False`.
            surgery_func(
                module_name=module_name,
                is_weight=False,
                target_size=target_size,
                head_ckpt=head_ckpt,
                tail_ckpt=tail_ckpt
            )
    
    # Save to file.
    save_checkpoint(head_ckpt, save_path)


def extract_teacher(ckpt):
    keys = list(ckpt["state_dict"].keys())
    for key in keys:
        if "student" in key:
            ckpt["state_dict"].pop(key)
            continue
        new_key = key.replace("teacher.", "")
        ckpt["state_dict"][new_key] = ckpt["state_dict"].pop(key)
    return ckpt


def save_checkpoint(ckpt, save_path):
    torch.save(ckpt, save_path)
    print("New checkpoint saved to {}".format(save_path))


def reset_checkpoint(ckpt):
    if "scheduler" in ckpt:
        del ckpt["scheduler"]
    if "optimizer" in ckpt:
        del ckpt["optimizer"]
    if "iteration" in ckpt:
        ckpt["iteration"] = 0


if __name__ == "__main__":
    args = parse_args()
    args.module_names = MODULE_NAMES_DICT[args.module_names]
    TAIL_IDS = sorted([int(line.strip("\n")) for line in open(args.tail_ids, "r").readlines()])
    HEAD_IDS = sorted([int(line.strip("\n")) for line in open(args.head_ids, "r").readlines()])
    ALL_IDS = sorted(HEAD_IDS + TAIL_IDS)
    ID2LABEL = {ID: label for label, ID in enumerate(ALL_IDS)}
    TARGET_SIZE = len(ID2LABEL)

    if args.method == "combine":
        combine_checkpoints(args)
    else:
        checkpoint_surgery(args)
