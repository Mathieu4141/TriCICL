from pathlib import Path
from typing import Any, List, Union

from avalanche.benchmarks.classic import SplitMNIST
from avalanche.evaluation.metrics import StreamConfusionMatrix
from avalanche.logging import InteractiveLogger
from avalanche.training import EvaluationPlugin
from avalanche.training.plugins import StrategyPlugin
from avalanche.training.strategies import Naive
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from tqdm import tqdm

from research.mathieu.tricicl_plugins import (
    make_tricicl_alternate_training_plugins,
    make_tricicl_during_training_plugin,
    make_tricicl_post_training_plugins,
    make_tricicl_pre_training_plugins,
)
from tricicl.constants import SEEDS, TB_DIR, device
from tricicl.loggers.tb import TensorboardLogger
from tricicl.metrics.confusion_matrix import SortedCMImageCreator
from tricicl.metrics.normalized_accuracy import NormalizedExperienceAccuracy, NormalizedStreamAccuracy
from tricicl.metrics.representation_shift import (
    ExperienceMeanRepresentationShift,
    MeanL2RepresentationShift,
    MeanCosineRepresentationShift,
)
from tricicl.models.simple_mlp import SimpleMLP
from tricicl.strategies.lwf_mc import LwFMCPlugin
from tricicl.utils.time import create_time_id


def evaluate_split_mnist(
    name: str,
    plugins: List[StrategyPlugin],
    seed: int,
    tensorboard_logs_dir: Union[str, Path] = str(TB_DIR),
    verbose: bool = False,
    criterion: Any = CrossEntropyLoss(),
):

    split_mnist = SplitMNIST(n_experiences=5, seed=seed)

    model = SimpleMLP(n_classes=split_mnist.n_classes, input_size=28 * 28)
    # model = SimpleCNN(n_channels=1, n_classes=split_mnist.n_classes)

    tb_logger = TensorboardLogger(tensorboard_logs_dir + f"/split_mnist/{name}/{seed}_{create_time_id()}")

    loggers = [tb_logger]
    if verbose:
        loggers.append(InteractiveLogger())

    cl_strategy = Naive(
        model=model,
        optimizer=SGD(model.parameters(), lr=0.001, momentum=0.9),
        criterion=criterion,
        train_mb_size=32,
        train_epochs=2,
        eval_mb_size=32,
        device=device,
        plugins=plugins,
        evaluator=EvaluationPlugin(
            [
                NormalizedStreamAccuracy(),
                NormalizedExperienceAccuracy(),
                ExperienceMeanRepresentationShift(MeanL2RepresentationShift()),
                ExperienceMeanRepresentationShift(MeanCosineRepresentationShift()),
            ],
            StreamConfusionMatrix(
                num_classes=split_mnist.n_classes,
                image_creator=SortedCMImageCreator(split_mnist.classes_order),
            ),
            loggers=loggers,
        ),
    )

    for i, train_task in enumerate(split_mnist.train_stream, 1):
        cl_strategy.train(train_task, num_workers=0)
        cl_strategy.eval(split_mnist.test_stream[:i])

    tb_logger.writer.flush()


if __name__ == "__main__":
    for seed in tqdm(SEEDS, unit="seed", desc="Training and evaluating methods on seeds"):
        # evaluate_split_mnist("LwF", [LwFPlugin()], seed)
        evaluate_split_mnist("LwF.MC", [LwFMCPlugin()], seed)
        evaluate_split_mnist(
            "tl-pre-all",
            make_tricicl_pre_training_plugins(
                memory_size=200,
                use_replay=True,
                use_training_dataloader=True,
                classification_loss_coef=0,
                distillation=False,
                pre_distillation=False,
                nme=False,
            ),
            seed,
        )
        evaluate_split_mnist(
            "tl-pre-D-all",
            make_tricicl_pre_training_plugins(
                memory_size=200,
                use_replay=True,
                use_training_dataloader=True,
                classification_loss_coef=0,
                nme=False,
                distillation=True,
                pre_distillation=False,
            ),
            seed,
        )
        evaluate_split_mnist(
            "tl-pre",
            make_tricicl_pre_training_plugins(
                memory_size=200,
                use_replay=True,
                use_training_dataloader=False,
                classification_loss_coef=0,
                distillation=False,
                pre_distillation=False,
                nme=False,
            ),
            seed,
        )
        evaluate_split_mnist(
            "tl-pre-D",
            make_tricicl_pre_training_plugins(
                memory_size=200,
                use_replay=True,
                use_training_dataloader=False,
                classification_loss_coef=0,
                nme=False,
                distillation=True,
                pre_distillation=False,
            ),
            seed,
        )
        evaluate_split_mnist(
            "tl-pre-PD-all",
            make_tricicl_pre_training_plugins(
                memory_size=200,
                use_replay=True,
                use_training_dataloader=True,
                classification_loss_coef=0,
                nme=False,
                distillation=False,
                pre_distillation=True,
            ),
            seed,
        )
        evaluate_split_mnist(
            "tl-pre-PD",
            make_tricicl_pre_training_plugins(
                memory_size=200,
                use_replay=True,
                use_training_dataloader=False,
                classification_loss_coef=0,
                nme=False,
                distillation=True,
                pre_distillation=True,
            ),
            seed,
        )
        evaluate_split_mnist(
            "tl-pre-nme",
            make_tricicl_pre_training_plugins(
                memory_size=200,
                use_replay=True,
                use_training_dataloader=True,
                classification_loss_coef=0,
                nme=True,
                distillation=False,
                pre_distillation=False,
            ),
            seed,
        )
        evaluate_split_mnist(
            "tl-pre-nme-D",
            make_tricicl_pre_training_plugins(
                memory_size=200,
                use_replay=True,
                use_training_dataloader=True,
                classification_loss_coef=0,
                nme=True,
                distillation=True,
                pre_distillation=False,
            ),
            seed,
        )
        evaluate_split_mnist(
            "tl-alternate",
            make_tricicl_alternate_training_plugins(
                memory_size=200,
                use_replay=True,
                use_training_dataloader=False,
                classification_loss_coef=0,
                nme=False,
                distillation=False,
                after_last_epoch=False,
            ),
            seed,
        )
        evaluate_split_mnist(
            "tl-alternate-last",
            make_tricicl_alternate_training_plugins(
                memory_size=200,
                use_replay=True,
                use_training_dataloader=False,
                classification_loss_coef=0,
                nme=False,
                distillation=False,
                after_last_epoch=True,
            ),
            seed,
        )
        evaluate_split_mnist(
            "tricicl",
            make_tricicl_during_training_plugin(
                memory_size=200,
                use_replay=True,
                nme=False,
                distillation=False,
            ),
            seed,
        )
        evaluate_split_mnist(
            "tricicl-only",
            make_tricicl_during_training_plugin(
                memory_size=200,
                use_replay=True,
                nme=True,
                distillation=False,
            ),
            seed,
            verbose=False,
            criterion=lambda *_, **__: 0,
        )
        # evaluate_on_seed("iCaRL", make_icarl_plugins(memory_size=200), seed)
        # evaluate_on_seed("LwF", [LwFPlugin()], seed)
        # evaluate_on_seed("naive", [], seed)
        # evaluate_on_seed("replay", [ReplayPlugin()], seed)
        # evaluate_on_seed("hybrid1", [ReplayPlugin(), LwFPlugin()], seed)
    #     evaluate_on_seed(
    #         "tl-post",
    #         make_tricicl_post_training_plugins(
    #             memory_size=200,
    #             use_replay=True,
    #             use_training_dataloader=True,
    #             classification_loss_coef=0,
    #             distillation=False,
    #             nme=False,
    #         ),
    #         seed,
    #     )
    #     evaluate_on_seed(
    #         "tl-post-D",
    #         make_tricicl_post_training_plugins(
    #             memory_size=200,
    #             use_replay=True,
    #             use_training_dataloader=True,
    #             classification_loss_coef=0,
    #             nme=False,
    #             distillation=True,
    #         ),
    #         seed,
    #     )
    #     evaluate_on_seed(
    #         "tl-post-nme",
    #         make_tricicl_post_training_plugins(
    #             memory_size=200,
    #             use_replay=True,
    #             use_training_dataloader=True,
    #             classification_loss_coef=0,
    #             nme=True,
    #             distillation=False,
    #         ),
    #         seed,
    #     )
    #     evaluate_on_seed(
    #         "tl-post-nme-D",
    #         make_tricicl_post_training_plugins(
    #             memory_size=200,
    #             use_replay=True,
    #             use_training_dataloader=True,
    #             classification_loss_coef=0,
    #             nme=True,
    #             distillation=True,
    #         ),
    #         seed,
    #     )
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
    #         memory_size=200,
    #         use_replay=True,
    #         use_training_dataloader=False,
    #         classification_loss_coef=0,
    #         nme=False,
    #         distillation=False,
    #     ),
    #     SEEDS[0],
    #     verbose=True,
    # )
    # evaluate_on_seed(
    #     "tl-alternate",
    #     make_tricicl_alternate_training_plugins(
    #         memory_size=200,
    #         use_replay=True,
    #         use_training_dataloader=False,
    #         classification_loss_coef=0,
    #         nme=False,
    #         distillation=False,
    #         after_last_epoch=False,
    #     ),
    #     SEEDS[0],
    #     verbose=True,
    # )
    # evaluate_on_seed(
    #     "tl-alternate-last",
    #     make_tricicl_alternate_training_plugins(
    #         memory_size=200,
    #         use_replay=True,
    #         use_training_dataloader=False,
    #         classification_loss_coef=0,
    #         nme=False,
    #         distillation=False,
    #         after_last_epoch=True,
    #     ),
    #     SEEDS[0],
    #     verbose=True,
    # )
    # evaluate_on_seed(
    #     "tricicl",
    #     make_tricicl_during_training_plugin(
    #         memory_size=200,
    #         use_replay=True,
    #         nme=False,
    #         distillation=False,
    #     ),
    #     SEEDS[0],
    #     verbose=True,
    # )
    # evaluate_on_seed(
    #     "tricicl-only",
    #     make_tricicl_during_training_plugin(
    #         memory_size=200,
    #         use_replay=True,
    #         nme=True,
    #         distillation=False,
    #     ),
    #     SEEDS[0],
    #     verbose=True,
    #     criterion=lambda *_, **__: 0,
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
