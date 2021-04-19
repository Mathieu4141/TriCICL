from avalanche.evaluation.metrics import ExperienceAccuracy, StreamAccuracy

"""
Avalanche changed the behavior to the global training step 2w ago
https://github.com/ContinualAI/avalanche/commit/89afc1cc620d88f09245920ac217a8be508e9e73#diff-6db76ef303657a86b96bc69cff62c8695bc4f046f683c43424ca8a2e608244f0
which is not useful when plotting, getting the stream step is much more useful.
"""


class NormalizedStreamAccuracy(StreamAccuracy):
    def __init__(self):
        super().__init__()
        self.x = 0

    def get_global_counter(self):
        return self.x

    def after_training(self, strategy: "BaseStrategy") -> "MetricResult":
        self.x += 1


class NormalizedExperienceAccuracy(ExperienceAccuracy):
    def __init__(self):
        super().__init__()
        self.x = 0

    def get_global_counter(self):
        return self.x

    def after_training(self, strategy: "BaseStrategy") -> "MetricResult":
        self.x += 1
