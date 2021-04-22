from torch import Tensor
from torch.nn import Dropout, Linear, ReLU, Sequential

from tricicl.models.feature_based_module import FeatureBasedModule


class SimpleMLP(FeatureBasedModule):
    def __init__(self, n_classes: int, input_size: int, hidden_size=512):
        super().__init__(features_size=hidden_size, n_classes=n_classes)
        self.featurizer = Sequential(Linear(input_size, hidden_size), ReLU(inplace=True), Dropout())
        self.classifier = Linear(hidden_size, n_classes)
        self.input_size = input_size

    def featurize(self, x: Tensor) -> Tensor:
        x = x.contiguous()
        x = x.view(x.size(0), self.input_size)
        x = self.featurizer(x)
        return x

    def classify(self, features: Tensor) -> Tensor:
        return self.classifier(features)
