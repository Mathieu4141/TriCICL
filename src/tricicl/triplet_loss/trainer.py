from sys import stdout
from typing import Dict, Iterable

from avalanche.benchmarks.utils import AvalancheSubset
from avalanche.benchmarks.utils.data_loader import MultiTaskMultiBatchDataLoader
from avalanche.training.plugins import StrategyPlugin
from avalanche.training.strategies import BaseStrategy
from torch import Tensor
from torch.nn import TripletMarginLoss
from tqdm import tqdm

from tricicl.cil_memory.memory import CILMemory
from tricicl.constants import device
from tricicl.models.feature_based_module import FeatureBasedModule
from tricicl.triplet_loss.penalty import TripletLossPenalty


class TripletLossTrainerPlugin(StrategyPlugin):
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

        self.class_id2dataset: Dict[int, AvalancheSubset] = None

    def train(
        self,
        strategy: BaseStrategy,
        *,
        num_workers: int = 0,
        shuffle: bool = True,
    ):
        triplet_loss_penalty = TripletLossPenalty(
            class_id2dataset=self.class_id2dataset or self.memory.class_id2dataset,
            criterion=self.triplet_criterion,
            model=strategy.model,
        )
        for mbatch in self._make_dataloader(strategy, num_workers=num_workers, shuffle=shuffle):
            strategy.optimizer.zero_grad()
            loss = sum(
                self._penalty(
                    mb_x.to(device),
                    mb_y.to(device),
                    triplet_loss_penalty=triplet_loss_penalty,
                    classification_criterion=strategy.criterion,
                    model=strategy.model,
                )
                for mb_x, mb_y, _ in mbatch.values()
            )
            loss.backward()
            strategy.optimizer.step()

    def _should_skip(self) -> bool:
        if not self.is_first:
            return False

        self.is_first = False
        if self.verbose:
            print("Skipping TripletLoss")
        return True

    def _penalty(
        self,
        mb_x: Tensor,
        mb_y: Tensor,
        *,
        triplet_loss_penalty: TripletLossPenalty,
        classification_criterion,
        model: FeatureBasedModule,
    ) -> Tensor:
        features, logits = model.forward_with_features(mb_x)

        triplet_loss = triplet_loss_penalty(features, mb_y)

        if not self.classification_loss_coef:
            return triplet_loss

        return triplet_loss + self.classification_loss_coef * classification_criterion(logits, mb_y)

    def _make_dataloader(self, strategy: BaseStrategy, *, num_workers: int, shuffle: bool) -> Iterable:
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
        return tqdm(dataloader, desc="TripletLoss", unit="batches", file=stdout, disable=not self.verbose)
