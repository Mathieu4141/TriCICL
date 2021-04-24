import logging
from argparse import ArgumentParser

from research.mathieu.benchmark_split_mnist import evaluate_split_mnist
from research.mathieu.gcloud.utils import get_method_plugins
from tricicl.storage.gcloud import GCStorage

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--method-name", default="naive")
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--job-dir")
    args = parser.parse_args()

    logging.info(f"Downloading Mnist dataset")
    GCStorage().download_dataset("mnist")

    logging.info(f"Starting evaluation of {args.method_name} for seed {args.seed}")
    evaluate_split_mnist(
        args.method_name,
        get_method_plugins(args.method_name),
        args.seed,
        tensorboard_logs_dir="gs://tricicl-public/logs/tb",
    )
