from abc import ABC, abstractmethod

from torch import Tensor
from torch.nn import Module


class FeatureBasedModule(Module, ABC):
    def __init__(self, features_size: int, n_classes: int):
        super().__init__()
        self.n_classes = n_classes
        self.features_size = features_size

    def forward(self, x: Tensor) -> Tensor:
        x = self.featurize(x)
        x = self.classify(x)
        return x

    @abstractmethod
    def featurize(self, x: Tensor) -> Tensor:
        pass

    @abstractmethod
    def classify(self, features: Tensor) -> Tensor:
        pass
