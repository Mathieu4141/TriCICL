from argparse import Namespace

from avalanche.training.plugins import ReplayPlugin

from research.mathieu.tricicl_plugins import (make_tricicl_alternate_training_plugins,
                                              make_tricicl_during_training_plugin, make_tricicl_post_training_plugins,
                                              make_tricicl_pre_training_plugins)
from tricicl.cil_memory.memory import CILMemory
from tricicl.cil_memory.plugin import CILMemoryPlugin
from tricicl.cil_memory.replay import CILReplayPlugin
from tricicl.cil_memory.strategy.herding import HerdingMemoryStrategy
from tricicl.strategies.icarl import make_icarl_plugins
from tricicl.strategies.lwf_mc import LwFMCPlugin


def get_method_plugins(args: Namespace):
    if args.method_name == "naive":
        return []
    elif args.method_name == "replay_avalanche":
        return [ReplayPlugin()]
    elif args.method_name == "replay":
        memory = CILMemory(args.memory_size)
        return [CILMemoryPlugin(memory, HerdingMemoryStrategy()), CILReplayPlugin(memory)]
    elif args.method_name == "hybrid1":
        memory = CILMemory(args.memory_size)
        return [CILMemoryPlugin(memory, HerdingMemoryStrategy()), CILReplayPlugin(memory), LwFMCPlugin()]
    elif args.method_name == "iCaRL":
        return make_icarl_plugins(args.memory_size)
    elif args.method_name == "tricicl-P-ND":
        return make_tricicl_post_training_plugins(args.memory_size, distillation=False)
    elif args.method_name == "tricicl-P-D":
        return make_tricicl_post_training_plugins(args.memory_size, distillation=True)
    elif args.method_name == "tricicl-B-D":
        return make_tricicl_pre_training_plugins(args.memory_size, pre_distillation=True)
    elif args.method_name == "tricicl-A-D":
        return make_tricicl_alternate_training_plugins(args.memory_size, distillation=True)
    elif args.method_name == "tricicl-P-D-NME":
        return make_tricicl_post_training_plugins(args.memory_size, distillation=True, nme=True)
    elif args.method_name == "tricicl-B-D-NME":
        return make_tricicl_pre_training_plugins(args.memory_size, pre_distillation=True, nme=True)
    elif args.method_name == "tricicl-A-D-NME":
        return make_tricicl_alternate_training_plugins(args.memory_size, distillation=True, nme=True)
    # elif args.method_name == "tricicl":
    #     return make_tricicl_during_training_plugin(args.memory_size, use_replay=True, nme=True, distillation=False)

    raise ValueError(f"Method {args.method_name} not supported")
