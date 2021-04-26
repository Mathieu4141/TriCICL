from avalanche.training.plugins import LwFPlugin
from avalanche.training.strategies import BaseStrategy


class BeforeExpDistillationPlugin(LwFPlugin):
    """Save previous model before training, instead of after training"""

    def after_training_exp(self, strategy, **kwargs):
        pass

    def before_training_exp(self, strategy: BaseStrategy, **kwargs):
        super().after_training_exp(strategy)
