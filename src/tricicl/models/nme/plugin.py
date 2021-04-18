import torch
from avalanche.training.plugins import StrategyPlugin
from avalanche.training.strategies import BaseStrategy
from torch import stack, tensor
from torch.utils.data import DataLoader

from tricicl.cil_memory.memory import CILMemory
from tricicl.models.feature_based_module import FeatureBasedModule
from tricicl.models.nme.nme import NME


class NMEPlugin(StrategyPlugin):
    def __init__(self, memory: CILMemory):
        super().__init__()
        self.memory = memory
        self.nme: NME = None

    def before_training(self, strategy: BaseStrategy, **kwargs):
        if self.nme is not None:
            return

        assert isinstance(strategy.model, FeatureBasedModule)

        self.nme = NME(strategy.model)
        strategy.model = self.nme

    def after_training(self, strategy: BaseStrategy, **kwargs):
        self.nme.eval()
        centers, class_ids = [], []
        for class_id, dataset in self.memory.class_id2dataset.items():
            center = torch.zeros(self.nme.base_module.features_size, device=self.nme.device)
            n_images = len(dataset)
            for images, *_ in DataLoader(dataset, batch_size=strategy.eval_mb_size):
                center += self.nme.base_module.featurize(images.to(self.nme.device)).sum(dim=0) / n_images
            centers.append(center)
            class_ids.append(class_id)
        self.nme.centers = stack(centers)
        self.nme.class_ids = tensor(class_ids)
