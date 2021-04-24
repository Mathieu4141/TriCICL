from typing import List

from avalanche.benchmarks import SplitCIFAR100
from avalanche.evaluation.metrics import StreamConfusionMatrix
from avalanche.logging import InteractiveLogger
from avalanche.training import EvaluationPlugin
from avalanche.training.plugins import ReplayPlugin, StrategyPlugin
from avalanche.training.strategies import BaseStrategy, Naive
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR, MultiStepLR

from tricicl.constants import TB_DIR, device
from tricicl.loggers.tb import TensorboardLogger
from tricicl.metrics.confusion_matrix import SortedCMImageCreator
from tricicl.metrics.normalized_accuracy import NormalizedExperienceAccuracy, NormalizedStreamAccuracy
from tricicl.models.resnet_32 import ResNet32
from tricicl.utils.time import create_time_id

N_CLASSES = 100


class LRSchedulerPlugin(StrategyPlugin):
    def __init__(self, start_lr: float = 2.0, gamma: float = 0.2, milestones: List[int] = None):
        super().__init__()

        self.start_lr = start_lr
        self.gamma = gamma
        self.milestones = milestones or [49, 63]
        self.scheduler: LambdaLR = None

    def before_training_exp(self, strategy: BaseStrategy, **kwargs):
        for g in strategy.optimizer.param_groups:
            g["lr"] = self.start_lr
        self.scheduler = MultiStepLR(strategy.optimizer, milestones=self.milestones, gamma=self.gamma)

    def after_training_epoch(self, strategy: BaseStrategy, **kwargs):
        self.scheduler.step()


def evaluate_on_cifar_100(
    *,
    method_name: str,
    plugins: List[StrategyPlugin],
    tb_dir: str = str(TB_DIR),
    seed: int = 42,
    verbose: bool = False,
    train_epochs: int = 70,
    n_classes_per_batch: int = 10,
):

    assert not N_CLASSES % n_classes_per_batch, "n_classes should be a multiple of n_classes_per_batch"

    scenario = SplitCIFAR100(n_experiences=N_CLASSES // n_classes_per_batch)
    model = ResNet32(n_classes=N_CLASSES)

    tb_logger = TensorboardLogger(tb_dir + f"/cifar100_{n_classes_per_batch}/{method_name}/{seed}_{create_time_id()}")

    loggers = [tb_logger]
    if verbose:
        loggers.append(InteractiveLogger())

    strategy = Naive(
        model=model,
        optimizer=SGD(model.parameters(), lr=2.0, weight_decay=0.00001, momentum=0.9),
        criterion=CrossEntropyLoss(),
        train_epochs=train_epochs,
        train_mb_size=128,
        device=device,
        plugins=plugins + [LRSchedulerPlugin()],
        evaluator=EvaluationPlugin(
            [NormalizedStreamAccuracy(), NormalizedExperienceAccuracy()],
            StreamConfusionMatrix(
                num_classes=N_CLASSES,
                image_creator=SortedCMImageCreator(scenario.classes_order),
            ),
            loggers=loggers,
        ),
    )

    for i, train_task in enumerate(scenario.train_stream, 1):
        strategy.train(train_task, num_workers=0)
        strategy.eval(scenario.test_stream[:i])

    tb_logger.writer.flush()


if __name__ == "__main__":
    evaluate_on_cifar_100(method_name="replay", plugins=[ReplayPlugin(2_000)], train_epochs=70, verbose=True)
