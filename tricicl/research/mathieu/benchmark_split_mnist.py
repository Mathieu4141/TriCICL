import torch
from avalanche.benchmarks.classic import SplitMNIST
from avalanche.evaluation.metrics import (StreamConfusionMatrix,
                                          accuracy_metrics)
from avalanche.logging import InteractiveLogger, TensorboardLogger
from avalanche.models import SimpleMLP
from avalanche.training import EvaluationPlugin
from avalanche.training.plugins import LwFPlugin, ReplayPlugin, StrategyPlugin
from avalanche.training.strategies import Naive
from torch.nn import CrossEntropyLoss
from torch.optim import SGD

from tricicl.constants import TB_DIR

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

perm_mnist = SplitMNIST(n_experiences=5)


def evaluate(name: str, plugins: list[StrategyPlugin]):
    model = SimpleMLP(num_classes=10)

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
            StreamConfusionMatrix(num_classes=perm_mnist.n_classes, save_image=True),
            loggers=[InteractiveLogger(), TensorboardLogger(TB_DIR / "split_mnist" / name)],
        ),
    )

    for i, train_task in enumerate(perm_mnist.train_stream, 1):
        cl_strategy.train(train_task, num_workers=0)
        cl_strategy.eval(perm_mnist.test_stream[:i])


if __name__ == "__main__":
    evaluate("LwF", [LwFPlugin()])
    evaluate("naive", [])
    evaluate("replay", [ReplayPlugin()])
    evaluate("hybrid1", [ReplayPlugin(), LwFPlugin()])
