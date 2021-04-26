from copy import deepcopy
from typing import List, Optional, Sequence, Union

from avalanche.training.plugins import StrategyPlugin
from avalanche.training.strategies import BaseStrategy
from torch import log_softmax
from torch.nn import Module
from torch.nn.functional import kl_div, softmax


class LwFMCPlugin(StrategyPlugin):
    def __init__(
        self, alpha: Union[Sequence[float], float] = 1, temperature: float = 2, save_before_training: bool = False
    ):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha

        self.saved_model: Optional[Module] = None
        self.seen_classes: List[int] = []

        if save_before_training:
            self.before_training_exp = self.save_model
        else:
            self.after_training_exp = self.save_model

    def save_model(self, strategy: BaseStrategy, **kwargs):
        self.saved_model = deepcopy(strategy.model.train())
        self.seen_classes.extend(
            strategy.experience.scenario.classes_in_experience[strategy.experience.current_experience]
        )

    def before_backward(self, strategy: BaseStrategy, **kwargs):
        if not self.seen_classes:
            return

        y_prev = self.saved_model(strategy.mb_x).detach()

        strategy.loss += self.alpha * kl_div(
            log_softmax(strategy.logits / self.temperature, dim=1)[:, self.seen_classes],
            softmax(y_prev / self.temperature, dim=1)[:, self.seen_classes],
            reduction="batchmean",
        )
