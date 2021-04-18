from avalanche.evaluation.metric_results import AlternativeValues, MetricValue
from avalanche.logging import StrategyLogger
from matplotlib.figure import Figure
from PIL.Image import Image
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.functional import to_tensor

from tricicl.utils.path import safe_path


class TensorboardLogger(StrategyLogger):
    """
    This is the same logger as `avalanche.logging.tensorboard_logger.TensorboardLogger`,
     except it does not enforce the use of Path.
    It can be thus used with gcloud buckets / aws s3 path if tf is installed
    """

    def __init__(self, tb_log_dir: str):
        super().__init__()
        if not _is_gs_or_s3_path(tb_log_dir):
            tb_log_dir = safe_path(tb_log_dir)
        self.writer = SummaryWriter(tb_log_dir)

    def log_metric(self, metric_value: MetricValue, callback: str):
        super().log_metric(metric_value, callback)
        name = metric_value.name
        value = metric_value.value

        if isinstance(value, AlternativeValues):
            value = value.best_supported_value(Image, Tensor, Figure, float, int)

        if isinstance(value, Figure):
            self.writer.add_figure(name, value, global_step=metric_value.x_plot)
        elif isinstance(value, Image):
            self.writer.add_image(name, to_tensor(value), global_step=metric_value.x_plot)
        elif isinstance(value, Tensor):
            self.writer.add_histogram(name, value, global_step=metric_value.x_plot)
        elif isinstance(value, (float, int)):
            self.writer.add_scalar(name, value, global_step=metric_value.x_plot)


def _is_gs_or_s3_path(p: str) -> bool:
    return p.startswith("gs://") or p.startswith("s3://")
