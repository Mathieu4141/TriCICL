from avalanche.training.plugins import StrategyPlugin
from avalanche.training.strategies import BaseStrategy
from torch.nn import TripletMarginLoss

from tricicl.cil_memory.memory import CILMemory
from tricicl.cil_memory.plugin import make_per_class_subset
from tricicl.triplet_loss.penalty import TripletLossPenalty


class TripletLossDuringTrainingPlugin(StrategyPlugin):
    def __init__(self, *, memory: CILMemory, margin: float, coef: float):
        super().__init__()
        self.coef = coef
        self.memory = memory
        self.criterion = TripletMarginLoss(margin=margin)

        self._penalty: TripletLossPenalty = None

    def before_training_exp(self, strategy: BaseStrategy, **kwargs):
        strategy.model.start_features_capture()

        class_id2dataset = dict(make_per_class_subset(strategy.experience.dataset.train()))
        class_id2dataset.update(self.memory.class_id2dataset)

        self._penalty = TripletLossPenalty(
            criterion=self.criterion, class_id2dataset=class_id2dataset, model=strategy.model
        )

    def before_backward(self, strategy: BaseStrategy, **kwargs):
        strategy.loss += self.coef * self._penalty(strategy.model.last_features_, strategy.mb_y)

    def after_training_exp(self, strategy: BaseStrategy, **kwargs):
        strategy.model.stop_features_capture()
