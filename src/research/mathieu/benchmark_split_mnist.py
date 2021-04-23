from pathlib import Path
from typing import List, Union

from avalanche.benchmarks.classic import SplitMNIST
from avalanche.evaluation.metrics import StreamConfusionMatrix
from avalanche.training import EvaluationPlugin
from avalanche.training.plugins import LwFPlugin, ReplayPlugin, StrategyPlugin
from avalanche.training.strategies import Naive
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from tqdm import tqdm

from tricicl.cil_memory.memory import CILMemory
from tricicl.cil_memory.plugin import CILMemoryPlugin
from tricicl.cil_memory.replay import CILReplayPlugin
from tricicl.cil_memory.strategy.herding import HerdingMemoryStrategy
from tricicl.cil_memory.strategy.strategy import CILMemoryStrategyABC
from tricicl.constants import SEEDS, TB_DIR, device
from tricicl.loggers.tb import TensorboardLogger
from tricicl.metrics.confusion_matrix import SortedCMImageCreator
from tricicl.metrics.normalized_accuracy import NormalizedExperienceAccuracy, NormalizedStreamAccuracy
from tricicl.models.nme.plugin import NMEPlugin
from tricicl.models.simple_mlp import SimpleMLP
from tricicl.strategies.icarl import make_icarl_plugins
from tricicl.triplet_loss.triplet_loss_post_training_plugin import TripletLossPostTrainingPlugin
from tricicl.utils.time import create_time_id


def evaluate_on_seed(
    name: str,
    plugins: List[StrategyPlugin],
    seed: int,
    tensorboard_logs_dir: Union[str, Path] = str(TB_DIR),
):

    split_mnist = SplitMNIST(n_experiences=5, seed=seed)

    model = SimpleMLP(n_classes=split_mnist.n_classes, input_size=28 * 28)
    # model = SimpleCNN(n_channels=1, n_classes=split_mnist.n_classes)

    tb_logger = TensorboardLogger(tensorboard_logs_dir + f"/split_mnist/{name}/{seed}_{create_time_id()}")

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
            [NormalizedStreamAccuracy(), NormalizedExperienceAccuracy()],
            StreamConfusionMatrix(
                num_classes=split_mnist.n_classes,
                image_creator=SortedCMImageCreator(split_mnist.classes_order),
            ),
            loggers=[
                tb_logger,
                # InteractiveLogger(),
            ],
        ),
    )

    for i, train_task in enumerate(split_mnist.train_stream, 1):
        cl_strategy.train(train_task, num_workers=0)
        cl_strategy.eval(split_mnist.test_stream[:i])

    tb_logger.writer.flush()


def make_tricicl_post_training_plugins(
    *,
    memory_size: int,
    use_replay: bool,
    use_training_dataloader: bool,
    classification_loss_coef: float,
    nme: bool,
    distillation: bool,
    memory_strategy: CILMemoryStrategyABC = HerdingMemoryStrategy(),
    margin: float = 1,
):
    memory = CILMemory(memory_size)
    plugins = [
        CILMemoryPlugin(memory, memory_strategy=memory_strategy),
        TripletLossPostTrainingPlugin(
            memory=memory,
            margin=margin,
            use_training_dataloader=use_training_dataloader,
            classification_loss_coef=classification_loss_coef,
        ),
    ]
    if use_replay:
        plugins.append(CILReplayPlugin(memory))
    if nme:
        plugins.append(NMEPlugin(memory))
    if distillation:
        plugins.append(LwFPlugin())
    return plugins


def make_replay_plugins(*, memory_size: int, memory_strategy: CILMemoryStrategyABC = HerdingMemoryStrategy()):
    memory = CILMemory(memory_size)
    return [CILMemoryPlugin(memory, memory_strategy=memory_strategy), CILReplayPlugin(memory)]


if __name__ == "__main__":
    for seed in tqdm(SEEDS, unit="seed", desc="Training and evaluating methods on seeds"):
        evaluate_on_seed("iCaRL", make_icarl_plugins(memory_size=200), seed)
        evaluate_on_seed("LwF", [LwFPlugin()], seed)
        evaluate_on_seed("naive", [], seed)
        evaluate_on_seed("replay", [ReplayPlugin()], seed)
        evaluate_on_seed("hybrid1", [ReplayPlugin(), LwFPlugin()], seed)
        evaluate_on_seed(
            "tl-post",
            make_tricicl_post_training_plugins(
                memory_size=200,
                use_replay=True,
                use_training_dataloader=True,
                classification_loss_coef=0,
                distillation=False,
                nme=False,
            ),
            seed,
        )
        evaluate_on_seed(
            "tl-post-D",
            make_tricicl_post_training_plugins(
                memory_size=200,
                use_replay=True,
                use_training_dataloader=True,
                classification_loss_coef=0,
                nme=False,
                distillation=True,
            ),
            seed,
        )
        evaluate_on_seed(
            "tl-post-nme",
            make_tricicl_post_training_plugins(
                memory_size=200,
                use_replay=True,
                use_training_dataloader=True,
                classification_loss_coef=0,
                nme=True,
                distillation=False,
            ),
            seed,
        )
        evaluate_on_seed(
            "tl-post-nme-D",
            make_tricicl_post_training_plugins(
                memory_size=200,
                use_replay=True,
                use_training_dataloader=True,
                classification_loss_coef=0,
                nme=True,
                distillation=True,
            ),
            seed,
        )
    # evaluate_on_seed("replay", [ReplayPlugin()], SEEDS[0])
    # evaluate_on_seed("replay-herding", make_replay_plugins(memory_size=200), SEEDS[0])
    # evaluate_on_seed(
    #     "tl",
    #     make_tricicl_post_training_plugins(
    #         memory_size=200, use_replay=False, use_training_dataloader=True, classification_loss_coef=0
    #     ),
    #     SEEDS[0],
    # )
    # evaluate_on_seed(
    #     "tl-cl.3",
    #     make_tricicl_post_training_plugins(
    #         memory_size=200, use_replay=False, use_training_dataloader=True, classification_loss_coef=0.3
    #     ),
    #     SEEDS[0],
    # )
    # evaluate_on_seed(
    #     "tl-replay",
    #     make_tricicl_post_training_plugins(
    #         memory_size=200, use_replay=True, use_training_dataloader=False, classification_loss_coef=0
    #     ),
    #     SEEDS[0],
    # )
    # evaluate_on_seed(
    #     "tl-replay-cl.3",
    #     make_tricicl_post_training_plugins(
    #         memory_size=200, use_replay=True, use_training_dataloader=False, classification_loss_coef=0.3
    #     ),
    #     SEEDS[0],
    # )
    # evaluate_on_seed(
    #     "tl-replay-dl-cl.3",
    #     make_tricicl_post_training_plugins(
    #         memory_size=200, use_replay=True, use_training_dataloader=True, classification_loss_coef=0.3
    #     ),
    #     SEEDS[0],
    # )
    # evaluate_on_seed(
    #     "tl-replay-dl",
    #     make_tricicl_post_training_plugins(
    #         memory_size=200, use_replay=True, use_training_dataloader=True, classification_loss_coef=0
    #     ),
    #     SEEDS[0],
    # )
