from avalanche.training.strategies import BaseStrategy

from tricicl.cil_memory.plugin import make_per_class_subset
from tricicl.triplet_loss.trainer import TripletLossTrainerPlugin


class TripletLossPreTrainingPlugin(TripletLossTrainerPlugin):
    def before_training_exp(self, strategy: BaseStrategy, num_workers: int = 0, shuffle: bool = True, **kwargs):
        if not self.memory:
            return

        self.class_id2dataset = dict(make_per_class_subset(strategy.experience.dataset.train()))
        self.class_id2dataset.update(self.memory.class_id2dataset)

        self.train(strategy, num_workers=num_workers, shuffle=shuffle)
