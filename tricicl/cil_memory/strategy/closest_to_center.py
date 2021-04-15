from avalanche.benchmarks.utils import AvalancheSubset
from torch import cat
from torch.utils.data import DataLoader

from tricicl.cil_memory.strategy.strategy import CILMemoryStrategyABC
from tricicl.models.feature_based_module import FeatureBasedModule


class ClosestToCenterMemoryStrategy(CILMemoryStrategyABC):
    def select(self, dataset: AvalancheSubset, model: FeatureBasedModule, m: int) -> AvalancheSubset:
        features = cat([model.featurize(images) for images, *_ in DataLoader(dataset, batch_size=32)])
        center = features.mean(dim=0)
        distances = pow(features - center, 2).sum(dim=1)
        return AvalancheSubset(dataset, distances.argsort()[:m])
