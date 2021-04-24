from abc import ABC, abstractmethod
from typing import Tuple

from torch import Tensor
from torch.nn import Module


class FeatureBasedModule(Module, ABC):
    def __init__(self, n_classes: int):
        super().__init__()
        self.n_classes = n_classes
        self.keep_features = False
        self.last_features_: Tensor = None

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_with_features(x)[1]

    def forward_with_features(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        features = self.featurize(x)
        logits = self.classify(features)
        return features, logits

    @abstractmethod
    def featurize(self, x: Tensor) -> Tensor:
        pass

    @abstractmethod
    def classify(self, features: Tensor) -> Tensor:
        pass
