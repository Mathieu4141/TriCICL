from avalanche.benchmarks.utils.data_loader import MultiTaskJoinedBatchDataLoader
from avalanche.training.plugins import StrategyPlugin
from avalanche.training.strategies import BaseStrategy

from tricicl.cil_memory.memory import CILMemory


class CILReplayPlugin(StrategyPlugin):
    def __init__(self, memory: CILMemory):
        super().__init__()
        self.memory = memory

    def before_training_exp(self, strategy: BaseStrategy, **kwargs):
        if not self.memory:
            return

        strategy.current_dataloader = MultiTaskJoinedBatchDataLoader(
            strategy.adapted_dataset,
            self.memory.dataset,
            batch_size=strategy.train_mb_size,
            shuffle=True,
            oversample_small_tasks=True,
        )
