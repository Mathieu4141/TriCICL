from torch import Tensor
from torch.nn import AdaptiveMaxPool2d, Conv2d, Dropout, Flatten, Linear, ReLU, Sequential

from tricicl.models.feature_based_module import FeatureBasedModule


class SimpleCNN(FeatureBasedModule):
    def __init__(self, *, n_classes: int, n_channels: int = 3):
        super().__init__(n_classes)

        self.featurizer = Sequential(
            Conv2d(n_channels, 32, kernel_size=3, stride=1, padding=1),
            ReLU(inplace=True),
            Conv2d(32, 32, kernel_size=3, padding=0),
            ReLU(inplace=True),
            # MaxPool2d(kernel_size=2, stride=2),
            # Dropout(p=0.25),
            # Conv2d(32, 64, kernel_size=3, padding=1),
            # ReLU(inplace=True),
            # Conv2d(64, 64, kernel_size=3, padding=0),
            # ReLU(inplace=True),
            # MaxPool2d(kernel_size=2, stride=2),
            # Dropout(p=0.25),
            # Conv2d(64, 64, kernel_size=1, padding=0),
            # ReLU(inplace=True),
            AdaptiveMaxPool2d(1),
            Dropout(p=0.25),
            Flatten(),
        )
        self.classifier = Sequential(Linear(32, n_classes))

    def featurize(self, x: Tensor) -> Tensor:
        return self.featurizer(x)

    def classify(self, features: Tensor) -> Tensor:
        return self.classifier(features)
