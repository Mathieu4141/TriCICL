from typing import List

from avalanche.evaluation.metric_utils import default_cm_image_creator
from PIL.Image import Image
from torch import Tensor


class SortedCMImageCreator:
    def __init__(self, class_order: List[int], use_labels: bool = True):
        self.labels = class_order if use_labels else None
        self.class_order = class_order

    def __call__(self, tensor: Tensor) -> Image:
        return default_cm_image_creator(tensor[self.class_order][:, self.class_order], display_labels=self.labels)
