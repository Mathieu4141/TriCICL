from sys import stdout
from typing import Iterable, Tuple

from avalanche.benchmarks.utils.data_loader import MultiTaskMultiBatchDataLoader
from avalanche.training.plugins import StrategyPlugin
from avalanche.training.strategies import BaseStrategy
from numpy import array, ndarray
from numpy.random import randint
from torch import Tensor, stack, tensor
from torch.nn import TripletMarginLoss
from torch.optim import Optimizer
from tqdm import tqdm

from tricicl.cil_memory.memory import CILMemory
from tricicl.constants import device
from tricicl.models.feature_based_module import FeatureBasedModule


class TripletLossPostTrainingPlugin(StrategyPlugin):
    def __init__(
        self,
        *,
        memory: CILMemory,
        margin: float,
        use_training_dataloader: bool,
        classification_loss_coef: float,
        verbose: bool = False,
    ):
        super().__init__()
        self.classification_loss_coef = classification_loss_coef
        self.use_training_dataloader = use_training_dataloader
        self.memory = memory
        self.verbose = verbose
        self.triplet_criterion = TripletMarginLoss(margin=margin)
        self.is_first = True

        self._optimizer: Optimizer = None
        self._model: FeatureBasedModule = None
        self._dataloader: Iterable = None
        self._classification_criterion = None

    def after_training_exp(self, strategy: BaseStrategy, *, num_workers: int = 0, shuffle: bool = True, **kwargs):
        if self._should_skip():
            return

        self._make_dataloader(strategy, num_workers=num_workers, shuffle=shuffle)
        self._model = strategy.model
        self._optimizer = strategy.optimizer
        self._classification_criterion = strategy.criterion

        self._train()

    def _should_skip(self) -> bool:
        if not self.is_first:
            return False

        self.is_first = False
        if self.verbose:
            print("Skipping TripletLoss")
        return True

    def _train(self):
        for mbatch in self._dataloader:
            self._optimizer.zero_grad()
            loss = sum(self._step(mb_x.to(device), mb_y.to(device)) for mb_x, mb_y, _ in mbatch.values())
            loss.backward()
            self._optimizer.step()

    def _step(self, mb_x: Tensor, mb_y: Tensor) -> Tensor:
        features, logits = self._model.forward_with_features(mb_x)

        classes = mb_y.detach().cpu().numpy()
        positives_features, positives_logits = self._generate_random_results_in_classes(classes)
        (negatives_features, negatives_logits), negatives_classes = self._generate_random_negative_results(classes)

        triplet_loss = self.triplet_criterion(features, positives_features, negatives_features)
        if not self.classification_loss_coef:
            return triplet_loss

        classification_loss = (
            self._classification_criterion(logits, mb_y)
            + self._classification_criterion(positives_logits, mb_y)
            + self._classification_criterion(negatives_logits, negatives_classes)
        )

        return triplet_loss + self.classification_loss_coef * classification_loss

    def _generate_random_negative_results(self, classes: ndarray) -> Tuple[Tuple[Tensor, Tensor], Tensor]:
        negative_classes = _generate_negative_classes(classes, array(list(self.memory.class_id2dataset.keys())))
        return self._generate_random_results_in_classes(negative_classes), tensor(negative_classes).to(device)

    def _generate_random_results_in_classes(self, classes: ndarray) -> Tuple[Tensor, Tensor]:
        indices = randint(self.memory.m, size=len(classes))
        images = stack([self.memory.class_id2dataset[class_id][index][0] for class_id, index in zip(classes, indices)])
        return self._model.forward_with_features(images.to(device))

    def _make_dataloader(self, strategy: BaseStrategy, *, num_workers: int, shuffle: bool):
        if self.use_training_dataloader:
            dataloader = strategy.dataloader
        else:
            dataloader = MultiTaskMultiBatchDataLoader(
                strategy.experience.dataset.train(),
                oversample_small_tasks=True,
                num_workers=num_workers,
                batch_size=strategy.train_mb_size,
                shuffle=shuffle,
            )
        self._dataloader = tqdm(dataloader, desc="TripletLoss", unit="batches", file=stdout, disable=not self.verbose)


def _generate_negative_classes(classes: ndarray, possible_classes: ndarray) -> ndarray:
    indices = randint(len(possible_classes) - 1, size=len(classes))
    negative_candidates = possible_classes[indices]
    indices += negative_candidates == classes
    return possible_classes[indices]
