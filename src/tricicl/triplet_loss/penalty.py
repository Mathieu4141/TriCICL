from typing import Dict

from avalanche.benchmarks.utils import AvalancheSubset
from numpy import array, ndarray
from numpy.random import randint
from torch import Tensor, stack
from torch.nn import TripletMarginLoss

from tricicl.constants import device
from tricicl.models.feature_based_module import FeatureBasedModule


class TripletLossPenalty:
    def __init__(
        self,
        *,
        criterion: TripletMarginLoss,
        class_id2dataset: Dict[int, AvalancheSubset],
        model: FeatureBasedModule,
    ):
        self.model = model
        self.criterion = criterion
        self.class_id2dataset = class_id2dataset
        self.criterion: TripletMarginLoss

    def __call__(self, features: Tensor, classes: Tensor) -> Tensor:
        classes = classes.detach().cpu().numpy()
        positives_features = self.generate_positives(classes)
        negatives_features = self.generate_negatives(classes)
        return self.criterion(features, positives_features, negatives_features)

    def generate_positives(self, classes: ndarray) -> Tensor:
        datasets = (self.class_id2dataset[class_id] for class_id in classes)
        images = stack([dataset[randint(len(dataset))][0] for dataset in datasets])
        return self.model.featurize(images.to(device))

    def generate_negatives(self, classes: ndarray) -> Tensor:
        negative_classes = _generate_negative_classes(classes, array(list(self.class_id2dataset.keys())))
        return self.generate_positives(negative_classes)


def _generate_negative_classes(classes: ndarray, possible_classes: ndarray) -> ndarray:
    indices = randint(len(possible_classes) - 1, size=len(classes))
    negative_candidates = possible_classes[indices]
    indices += negative_candidates == classes
    return possible_classes[indices]
