from typing import Iterable, Tuple

from avalanche.benchmarks import Experience
from avalanche.benchmarks.utils import AvalancheDataset, AvalancheSubset
from avalanche.training.plugins import StrategyPlugin
from avalanche.training.strategies import BaseStrategy
from numpy import array, where
from torch.nn import Module

from tricicl.cil_memory.memory import CILMemory
from tricicl.cil_memory.strategy.random import RandomMemoryStrategy
from tricicl.cil_memory.strategy.strategy import CILMemoryStrategyABC


class CILMemoryPlugin(StrategyPlugin):
    def __init__(self, memory: CILMemory, memory_strategy: CILMemoryStrategyABC = RandomMemoryStrategy()):
        super().__init__()

        self.memory = memory
        self.memory_strategy = memory_strategy

    def after_training_exp(self, strategy: BaseStrategy, **kwargs):
        self._add_experience_to_memory(strategy.experience, strategy.model)
        self.memory.prune_surplus()

    def _add_experience_to_memory(self, experience: Experience, model: Module):
        for class_id, subset in make_per_class_subset(experience.dataset):
            self.memory.class_id2dataset[class_id] = self.memory_strategy.select(
                subset,
                model,
                min(self.memory.m, len(experience.dataset)),
            )


def make_per_class_subset(dataset: AvalancheDataset) -> Iterable[Tuple[int, AvalancheSubset]]:
    class_ids = array(dataset.targets)
    for class_id in set(class_ids):
        idx = where(class_ids == class_id)[0]
        yield class_id, AvalancheSubset(dataset, idx)
