from avalanche.training.strategies import BaseStrategy

from tricicl.cil_memory.memory import CILMemory
from tricicl.cil_memory.plugin import make_per_class_subset
from tricicl.triplet_loss.trainer import TripletLossTrainerPlugin


class TripletLossAlternateTrainingPlugin(TripletLossTrainerPlugin):
    def __init__(
        self,
        *,
        memory: CILMemory,
        margin: float,
        use_training_dataloader: bool,
        classification_loss_coef: float,
        after_last_epoch: bool,
        verbose: bool = False,
    ):
        super().__init__(
            memory=memory,
            margin=margin,
            use_training_dataloader=use_training_dataloader,
            classification_loss_coef=classification_loss_coef,
            verbose=verbose,
        )
        self.after_last_epoch = after_last_epoch

    def before_training_exp(self, strategy: BaseStrategy, **kwargs):
        self.class_id2dataset = dict(make_per_class_subset(strategy.experience.dataset.train()))
        self.class_id2dataset.update(self.memory.class_id2dataset)

    def after_training_epoch(self, strategy: BaseStrategy, *, num_workers: int = 0, shuffle: bool = True, **kwargs):
        if not self.after_last_epoch and strategy.epoch == strategy.train_epochs - 1:
            return

        self.train(strategy=strategy, num_workers=num_workers, shuffle=shuffle)
