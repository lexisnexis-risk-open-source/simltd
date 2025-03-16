from mmdet.models.detectors import DINO as BaseDINO
from mmdet.models.utils.freeze import freeze_model_except

from mmdet.registry import MODELS


@MODELS.register_module(force=True)
class DINO(BaseDINO):
    r"""Implementation of `DINO: DETR with Improved DeNoising Anchor Boxes
    for End-to-End Object Detection <https://arxiv.org/abs/2203.03605>`.

    A subclass of `mmdet.models.detectors.DINO` to allow for
    targeted freezing of select modules in the model.

    Args:
        freeze_exceptions (list or tuple, optional): List of module names
            to update the gradient during transfer learning and fine-tuning.
            Defaults to None, meaning no module is frozen.
    """

    def __init__(self,
                 *args,
                 freeze_exceptions=None,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        freeze_model_except(self, freeze_exceptions)
