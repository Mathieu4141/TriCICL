from typing import List

import torch
from avalanche.benchmarks.classic import SplitMNIST
from avalanche.evaluation.metrics import StreamConfusionMatrix, accuracy_metrics
from avalanche.logging import TensorboardLogger
from avalanche.training import EvaluationPlugin
from avalanche.training.plugins import LwFPlugin, ReplayPlugin, StrategyPlugin
from avalanche.training.strategies import Naive
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from tqdm import tqdm

from tricicl.constants import SEEDS, TB_DIR
from tricicl.metrics.confusion_matrix import SortedCMImageCreator
from tricicl.models.simple_mlp import SimpleMLP
from tricicl.strategies.icarl import make_icarl_plugins
from tricicl.utils.time import create_time_id

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def evaluate_on_seed(name: str, plugins: List[StrategyPlugin], seed: int):
    perm_mnist = SplitMNIST(n_experiences=5, seed=seed)
    model = SimpleMLP(n_classes=perm_mnist.n_classes, input_size=28 * 28)

    tb_logger = TensorboardLogger(TB_DIR + f"/split_mnist/{name}/{seed}_{create_time_id()}")

    cl_strategy = Naive(
        model,
        SGD(model.parameters(), lr=0.001, momentum=0.9),
        CrossEntropyLoss(),
        train_mb_size=32,
        train_epochs=2,
        eval_mb_size=32,
        device=device,
        plugins=plugins,
        evaluator=EvaluationPlugin(
            accuracy_metrics(experience=True, stream=True),
            StreamConfusionMatrix(
                num_classes=perm_mnist.n_classes,
                image_creator=SortedCMImageCreator(perm_mnist.classes_order),
            ),
            loggers=[tb_logger],
        ),
    )

    for i, train_task in enumerate(perm_mnist.train_stream, 1):
        cl_strategy.train(train_task, num_workers=0)
        cl_strategy.eval(perm_mnist.test_stream[:i])

    tb_logger.writer.flush()


if __name__ == "__main__":
    for seed in tqdm(SEEDS, unit="seed", desc="Training and evaluating methods on seeds"):
        evaluate_on_seed("iCaRL", make_icarl_plugins(memory_size=200), seed)
        evaluate_on_seed("LwF", [LwFPlugin()], seed)
        evaluate_on_seed("naive", [], seed)
        evaluate_on_seed("replay", [ReplayPlugin()], seed)
        evaluate_on_seed("hybrid1", [ReplayPlugin(), LwFPlugin()], seed)
