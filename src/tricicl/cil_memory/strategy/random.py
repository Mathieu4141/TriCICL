from avalanche.benchmarks.utils import AvalancheSubset
from numpy.random import choice
from torch.nn import Module

from tricicl.cil_memory.strategy.strategy import CILMemoryStrategyABC


class RandomMemoryStrategy(CILMemoryStrategyABC):
    def select(self, dataset: AvalancheSubset, model: Module, m: int) -> AvalancheSubset:
        return AvalancheSubset(dataset, choice(len(dataset), m, replace=False))
