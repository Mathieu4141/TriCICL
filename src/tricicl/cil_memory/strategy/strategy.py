from abc import ABC, abstractmethod

from avalanche.benchmarks.utils import AvalancheSubset
from torch.nn import Module


class CILMemoryStrategyABC(ABC):
    @abstractmethod
    def select(self, dataset: AvalancheSubset, model: Module, m: int) -> AvalancheSubset:
        pass
