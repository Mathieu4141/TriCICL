from tricicl.cil_memory.memory import CILMemory
from tricicl.cil_memory.plugin import CILMemoryPlugin
from tricicl.cil_memory.replay import CILReplayPlugin
from tricicl.cil_memory.strategy.herding import HerdingMemoryStrategy
from tricicl.cil_memory.strategy.strategy import CILMemoryStrategyABC
from tricicl.models.nme.plugin import NMEPlugin
from tricicl.strategies.lwf_mc import LwFMCPlugin
from tricicl.triplet_loss.triplet_loss_alternate_training_plugin import TripletLossAlternateTrainingPlugin
from tricicl.triplet_loss.triplet_loss_during_training_plugin import TripletLossDuringTrainingPlugin
from tricicl.triplet_loss.triplet_loss_post_training_plugin import TripletLossPostTrainingPlugin
# TODO clean that, too many repetitions here
from tricicl.triplet_loss.triplet_loss_pre_training_plugin import TripletLossPreTrainingPlugin


def make_tricicl_post_training_plugins(
    memory_size: int,
    *,
    distillation: bool,
    use_replay: bool = True,
    use_training_dataloader: bool = True,
    classification_loss_coef: float = 0,
    nme: bool = False,
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
        plugins.append(LwFMCPlugin())
    return plugins


def make_tricicl_pre_training_plugins(
    memory_size: int,
    *,
    distillation: bool,
    pre_distillation: bool,
    use_replay: bool = True,
    use_training_dataloader: bool = True,
    classification_loss_coef: float = 0,
    nme: bool = False,
    memory_strategy: CILMemoryStrategyABC = HerdingMemoryStrategy(),
    margin: float = 1,
):
    memory = CILMemory(memory_size)
    plugins = [
        CILMemoryPlugin(memory, memory_strategy=memory_strategy),
        TripletLossPreTrainingPlugin(
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
        plugins.append(LwFMCPlugin())
    if pre_distillation:
        plugins.append(LwFMCPlugin(save_before_training=True))
    return plugins


def make_tricicl_alternate_training_plugins(
    *,
    memory_size: int,
    use_replay: bool,
    use_training_dataloader: bool,
    classification_loss_coef: float,
    nme: bool,
    distillation: bool,
    after_last_epoch: bool,
    memory_strategy: CILMemoryStrategyABC = HerdingMemoryStrategy(),
    margin: float = 1,
    verbose: bool = False,
):
    memory = CILMemory(memory_size)
    plugins = [
        CILMemoryPlugin(memory, memory_strategy=memory_strategy),
        TripletLossAlternateTrainingPlugin(
            memory=memory,
            margin=margin,
            use_training_dataloader=use_training_dataloader,
            classification_loss_coef=classification_loss_coef,
            after_last_epoch=after_last_epoch,
            verbose=verbose,
        ),
    ]
    if use_replay:
        plugins.append(CILReplayPlugin(memory))
    if nme:
        plugins.append(NMEPlugin(memory))
    if distillation:
        plugins.append(LwFMCPlugin())
    return plugins


def make_tricicl_during_training_plugin(
    *,
    memory_size: int,
    use_replay: bool,
    nme: bool,
    distillation: bool,
    memory_strategy: CILMemoryStrategyABC = HerdingMemoryStrategy(),
    margin: float = 1,
    coef_tl: float = 1,
):
    memory = CILMemory(memory_size)
    plugins = [
        CILMemoryPlugin(memory, memory_strategy=memory_strategy),
        TripletLossDuringTrainingPlugin(memory=memory, margin=margin, coef=coef_tl),
    ]
    if use_replay:
        plugins.append(CILReplayPlugin(memory))
    if nme:
        plugins.append(NMEPlugin(memory))
    if distillation:
        plugins.append(LwFMCPlugin())
    return plugins


def make_replay_plugins(*, memory_size: int, memory_strategy: CILMemoryStrategyABC = HerdingMemoryStrategy()):
    memory = CILMemory(memory_size)
    return [CILMemoryPlugin(memory, memory_strategy=memory_strategy), CILReplayPlugin(memory)]
