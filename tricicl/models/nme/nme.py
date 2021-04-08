from copy import deepcopy

import torch
from torch import Tensor, stack, zeros

from tricicl.models.feature_based_module import FeatureBasedModule


class NME(FeatureBasedModule):
    def __init__(self, base_module: FeatureBasedModule):
        super().__init__(base_module.features_size, base_module.n_classes)
        self.base_module = base_module
        self.class_ids: Tensor = None
        self.centers: Tensor = None

    def featurize(self, x: Tensor) -> Tensor:
        return self.base_module.featurize(x)

    def classify(self, features: Tensor) -> Tensor:
        if self.training:
            return self.base_module.classify(features)

        p = stack([1 / torch.pow(feature - self.centers, 2).sum(dim=1) for feature in features])
        proba = zeros(len(p), self.n_classes)
        proba[:, self.class_ids] = p
        return proba

    def __deepcopy__(self, memodict={}):
        return deepcopy(self.base_module)
