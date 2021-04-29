from abc import ABC, abstractmethod
from typing import Dict, Union

import torch
from avalanche.evaluation.metric_definitions import Metric, PluginMetric
from avalanche.evaluation.metric_results import MetricValue, MetricResult
from avalanche.evaluation.metric_utils import get_metric_name
from avalanche.training.strategies import BaseStrategy
from torch import Tensor
from tricicl.models.feature_based_module import FeatureBasedModule


class Representation:
    def __init__(self):
        self.vectors = []

    def update(self, logits: Tensor) -> None:
        if logits.device.type == "cpu":
            self.vectors.extend(logits.detach().numpy().tolist())
        else:
            self.vectors.extend(logits.detach().cpu().numpy().tolist())

    def result(self) -> Tensor:
        return torch.as_tensor(self.vectors)

    def reset(self) -> None:
        self.vectors = []


class RepresentationShift(ABC, Metric[Union[None, Tensor, Dict[int, Tensor], float, Dict[int, float]]]):
    """
    The standalone Representation Shift metric.
    This metric returns the representation shift relative to a specific key.
    Alternatively, this metric returns a dict in which each key is associated
    to the representation shift.
    Representation shift is computed as the difference between the first value recorded
    for a specific key and the last value recorded for that key.
    The value associated to a key can be update with the `update` method.

    At initialization, this metric returns an empty dictionary.
    """

    def __init__(self) -> None:
        """
        Creates an instance of the standalone Representation Shift metric
        """

        super().__init__()

        self.initial: Dict[int, Tensor] = dict()
        """
        The initial value for each key.
        """

        self.last: Dict[int, Tensor] = dict()
        """
        The last value detected for each key
        """

    def update_initial(self, k: int, v: Tensor) -> None:
        self.initial[k] = v

    def update_last(self, k: int, v: Tensor) -> None:
        self.last[k] = v

    def update(self, k: int, v: Tensor, initial: bool = False) -> None:
        if initial:
            self.update_initial(k, v)
        else:
            self.update_last(k, v)

    @abstractmethod
    def _shift_measure(self, value1: Tensor, value2: Tensor) -> Union[float, Tensor]:
        pass

    def result(self, k=None) -> Union[None, Tensor, Dict[int, Tensor], float, Dict[int, float]]:
        """
        Representation Shift is returned only for keys encountered twice.

        :param k: the key for which returning Representation Shift. If k has not
            updated at least twice it returns None. If k is None,
            Representation Shift will be returned for all keys encountered at least
            twice.

        :return: the difference between the first and last value encountered
            for k, if k is not None. It returns None if k has not been updated
            at least twice. If k is None, returns a dictionary
            containing keys whose value has been updated at least twice. The
            associated value is the difference between the first and last
            value recorded for that key.
        """

        representation_shift = {}
        if k is not None:
            if k in self.initial and k in self.last:
                return self._shift_measure(value1=self.initial[k], value2=self.last[k])
            else:
                return None

        ik = set(self.initial.keys())
        both_keys = ik.intersection(set(self.last.keys()))

        for k in both_keys:
            representation_shift[k] = self._shift_measure(value1=self.initial[k], value2=self.last[k])

        return representation_shift

    def reset_last(self) -> None:
        self.last: Dict[int, Tensor] = dict()

    def reset(self) -> None:
        self.initial: Dict[int, Tensor] = dict()
        self.last: Dict[int, Tensor] = dict()


class MeanL2RepresentationShift(RepresentationShift):
    def _shift_measure(self, value1: Tensor, value2: Tensor) -> float:
        return torch.mean(torch.linalg.norm(value1 - value2, dim=1)).item()

    def __str__(self):
        return "MeanL2RepresentationShift"


class MeanCosineRepresentationShift(RepresentationShift):
    def _shift_measure(self, value1: Tensor, value2: Tensor) -> float:
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        return torch.mean(cos(value1, value2)).item()

    def __str__(self):
        return "MeanCosineRepresentationShift"


class ExperienceMeanRepresentationShift(PluginMetric[Dict[int, float]]):
    def __init__(self, shifting_criterion: RepresentationShift):

        super().__init__()
        self.x = 0

        self.shifting = shifting_criterion
        """
        The general metric to compute representation shift
        """

        self._current_representation = Representation()
        """
        The representation buffer
        """

        self.eval_exp_id = None
        """
        The current evaluation experience id
        """

        self.train_exp_id = None
        """
        The last encountered training experience id
        """

    def reset(self) -> None:
        """
        Resets the metric.

        Beware that this will also reset the initial representation of each
        experience!

        :return: None.
        """
        self.shifting.reset()

    def reset_last_accuracy(self) -> None:
        """
        Resets the last representation.

        This will preserve the initial representation value of each experience.
        To be used at the beginning of each eval experience.

        :return: None.
        """
        self.shifting.reset_last()

    def update(self, k: int, v: Tensor, initial=False):
        """
        Update shifting metric.
        See `RepresentationShift` for more detailed information.

        :param k: key to update
        :param v: value associated to k
        :param initial: update initial value. If False, update
            last value.
        """
        self.shifting.update(k, v, initial=initial)

    def result(self, k=None) -> Union[None, float, Dict[int, float]]:
        """
        See `RepresentationShift` documentation for more detailed information.

        k: optional key from which compute forgetting.
        """
        return self.shifting.result(k=k)

    def before_training_exp(self, strategy: "BaseStrategy") -> None:
        self.train_exp_id = strategy.experience.current_experience

    def before_eval(self, strategy) -> None:
        self.reset_last_accuracy()

    def before_eval_exp(self, strategy: "BaseStrategy") -> None:
        self._current_representation.reset()

    def after_eval_iteration(self, strategy: "BaseStrategy") -> None:
        super().after_eval_iteration(strategy)
        self.eval_exp_id = strategy.experience.current_experience
        if isinstance(strategy.model, FeatureBasedModule):
            self._current_representation.update(logits=strategy.model.featurize(strategy.mb_x))
        else:
            self._current_representation.update(logits=strategy.logits)

    def after_eval_exp(self, strategy: "BaseStrategy") -> MetricResult:
        # update experience on which training just ended
        if self.train_exp_id == self.eval_exp_id:
            self.update(self.eval_exp_id, self._current_representation.result(), initial=True)
        else:
            # update other experiences
            # if experience has not been encountered in training
            # its value will not be considered in shifting
            self.update(self.eval_exp_id, self._current_representation.result())

        # this checks if the evaluation experience has been
        # already encountered at training time
        # before the last training.
        # If not, shifting should not be returned.
        if self.result(k=self.eval_exp_id) is not None:
            return self._package_result(strategy)

    def get_global_counter(self):
        return self.x

    def after_training(self, strategy: "BaseStrategy") -> "MetricResult":
        self.x += 1

    def _package_result(self, strategy: "BaseStrategy") -> MetricResult:

        shifting = self.result(k=self.eval_exp_id)
        metric_name = get_metric_name(self, strategy, add_experience=True)
        plot_x_position = self.get_global_counter()

        metric_values = [MetricValue(self, metric_name, shifting, plot_x_position)]
        return metric_values

    def __str__(self):
        return f"Experience{self.shifting}"
