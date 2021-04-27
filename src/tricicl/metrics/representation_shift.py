import numpy
from avalanche.evaluation.metric_definitions import Metric
from typing import Optional, Dict, Union
import numpy as np

from abc import ABC, abstractmethod


class RepresentationShift(ABC, Metric[Optional[Union[np.ndarray, Dict[int, np.ndarray], float, Dict[int, float]]]]):
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

        self.initial: Dict[int, np.ndarray] = dict()
        """
        The initial value for each key.
        """

        self.last: Dict[int, np.ndarray] = dict()
        """
        The last value detected for each key
        """

    def update_initial(self, k: int, v: np.ndarray) -> None:
        self.initial[k] = v

    def update_last(self, k: int, v: np.ndarray) -> None:
        self.last[k] = v

    def update(self, k: int, v: np.ndarray, initial: bool = False) -> None:
        if initial:
            self.update_initial(k, v)
        else:
            self.update_last(k, v)

    @abstractmethod
    def __shift_measure(self, value1: np.ndarray, value2: numpy.ndarray) -> Union[float, np.ndarray]:
        pass

    def result(self, k=None) -> Optional[Union[np.ndarray, Dict[int, np.ndarray], float, Dict[int, float]]]:
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
                return self.__shift_measure(value1=self.initial[k], value2=self.last[k])
            else:
                return None

        ik = set(self.initial.keys())
        both_keys = list(ik.intersection(set(self.last.keys())))

        for k in both_keys:
            representation_shift[k] = self.__shift_measure(value1=self.initial[k], value2=self.last[k])

        return representation_shift

    def reset_last(self) -> None:
        self.last: Dict[int, np.ndarray] = dict()

    def reset(self) -> None:
        self.initial: Dict[int, np.ndarray] = dict()
        self.last: Dict[int, np.ndarray] = dict()
