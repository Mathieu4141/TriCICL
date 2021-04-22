from typing import List

from avalanche.training.plugins import LwFPlugin, StrategyPlugin

from tricicl.cil_memory.memory import CILMemory
from tricicl.cil_memory.plugin import CILMemoryPlugin
from tricicl.cil_memory.replay import CILReplayPlugin
from tricicl.cil_memory.strategy.herding import HerdingMemoryStrategy
from tricicl.cil_memory.strategy.strategy import CILMemoryStrategyABC
from tricicl.models.nme.plugin import NMEPlugin


def make_icarl_plugins(
    memory_size: int, memory_strategy: CILMemoryStrategyABC = HerdingMemoryStrategy()
) -> List[StrategyPlugin]:
    memory = CILMemory(memory_size)
    return [
        CILMemoryPlugin(memory, memory_strategy=memory_strategy),
        NMEPlugin(memory),
        CILReplayPlugin(memory),
        LwFPlugin(),
    ]
