import logging
from argparse import ArgumentParser

from research.mathieu.demo_cifar_100 import evaluate_on_cifar_100
from research.mathieu.gcloud.utils import get_method_plugins
from tricicl.storage.gcloud import GCStorage

if __name__ == "__main__":
    logging.getLogger().setLevel("INFO")

    parser = ArgumentParser()
    parser.add_argument("--method-name", default="naive")
    parser.add_argument("--memory-size", default=2_000, type=int)
    parser.add_argument("--n-classes-per-batch", default=10, type=int)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--job-dir")
    args = parser.parse_args()

    logging.info(f"Downloading CIFAR100 dataset")
    GCStorage().download_dataset("cifar100")

    logging.info(f"Starting evaluation of {args.method_name} for seed {args.seed}")
    evaluate_on_cifar_100(
        method_name=args.method_name,
        plugins=get_method_plugins(args),
        seed=args.seed,
        tb_dir="gs://tricicl-public/logs/tb",
        verbose=False,
        n_classes_per_batch=args.n_classes_per_batch,
        train_epochs=70,
    )
