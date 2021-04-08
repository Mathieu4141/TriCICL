from math import inf

from avalanche.benchmarks.utils import AvalancheSubset
from torch import cat
from torch.utils.data import DataLoader

from tricicl.cil_memory.strategy.strategy import CILMemoryStrategyABC
from tricicl.models.feature_based_module import FeatureBasedModule


class HerdingMemoryStrategy(CILMemoryStrategyABC):
    def select(self, dataset: AvalancheSubset, model: FeatureBasedModule, m: int) -> AvalancheSubset:
        features = cat([model.featurize(images) for images, *_ in DataLoader(dataset, batch_size=32)])
        center = features.mean(dim=0)
        current_center = center * 0
        indices = []
        for i in range(m):
            candidate_centers = current_center * i / (i + 1) + features / (i + 1)
            distances = pow(candidate_centers - center, 2).sum(dim=1)
            distances[indices] = inf
            indices.append(distances.argmin().tolist())
            current_center = candidate_centers[indices[-1]]
        return AvalancheSubset(dataset, indices)
