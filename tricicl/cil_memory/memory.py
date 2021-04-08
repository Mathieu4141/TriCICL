from dataclasses import dataclass, field
from typing import Dict

from avalanche.benchmarks.utils import AvalancheConcatDataset, AvalancheDataset, AvalancheSubset
from numpy import arange


@dataclass
class CILMemory:
    size: int
    class_id2dataset: Dict[int, AvalancheSubset] = field(default_factory=dict)

    @property
    def dataset(self) -> AvalancheDataset:
        return AvalancheConcatDataset(list(self.class_id2dataset.values()))

    def prune_surplus(self):
        self.class_id2dataset = {
            class_id: AvalancheSubset(dataset, arange(self.m)) for class_id, dataset in self.class_id2dataset.items()
        }

    @property
    def m(self) -> int:
        return self.size // (len(self) or 1)

    def __len__(self) -> int:
        return len(self.class_id2dataset)
