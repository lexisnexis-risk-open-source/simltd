import copy

from mmdet.registry import DATASETS
from mmdet.datasets import LVISV1Dataset as BaseLVISV1Dataset
from .api_wrappers.lvis_api import LVIS


@DATASETS.register_module(force=True)
class LVISV1Dataset(BaseLVISV1Dataset):
    """Updated LVIS v1 object detection dataset with filterable class names."""

    def load_data_list(self):
        """Load annotations from an annotation file ``self.ann_file``.

        Returns:
            List[dict]: A list of annotation.
        """
        self.lvis = LVIS(self.ann_file)
        self.cat_ids = self.lvis.get_cat_ids(cat_names=self.metainfo["classes"])
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.cat_img_map = copy.deepcopy(self.lvis.cat_img_map)
        
        img_ids = self.lvis.get_img_ids()
        data_list = []
        total_ann_ids = []
        for img_id in img_ids:
            raw_img_info = self.lvis.load_imgs([img_id])[0]
            raw_img_info["img_id"] = img_id
            # Extract the ``filename`` from ``coco_url``.
            # e.g. http://images.cocodataset.org/train2017/000000391895.jpg
            raw_img_info["file_name"] = raw_img_info["coco_url"].replace(
                "http://images.cocodataset.org/", "")
            ann_ids = self.lvis.get_ann_ids(img_ids=[img_id])
            raw_ann_info = self.lvis.load_anns(ann_ids)
            total_ann_ids.extend(ann_ids)
            parsed_data_info = self.parse_data_info({
                "raw_ann_info":
                raw_ann_info,
                "raw_img_info":
                raw_img_info
            })
            data_list.append(parsed_data_info)
        if self.ANN_ID_UNIQUE:
            assert len(set(total_ann_ids)) == len(total_ann_ids),
            f"Annotation ids in ``{self.ann_file}`` are not unique!"
        
        del self.lvis

        return data_list
