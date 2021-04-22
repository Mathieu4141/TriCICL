import logging
from argparse import ArgumentParser

from avalanche.training.plugins import LwFPlugin, ReplayPlugin

from research.mathieu.benchmark_split_mnist import evaluate_on_seed
from tricicl.storage.gcloud import GCStorage


def get_method_plugins(method_name: str):
    if method_name == "naive":
        return []
    elif method_name == "hybrid1":
        return [ReplayPlugin(), LwFPlugin()]

    raise ValueError(f"Method {method_name} not supported")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--method-name", default="naive")
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--job-dir")
    args = parser.parse_args()

    logging.info(f"Downloading Mnist dataset")
    GCStorage().download_dataset("mnist")

    logging.info(f"Starting evaluation of {args.method_name} for seed {args.seed}")
    evaluate_on_seed(
        args.method_name,
        get_method_plugins(args.method_name),
        args.seed,
        tensorboard_logs_dir="gs://tricicl-public/logs/tb",
    )
