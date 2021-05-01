from tricicl.loggers.tb_aggregation import aggregate_task_results
from tricicl.storage.gcloud import GCStorage

if __name__ == "__main__":
    # aggregate_task_results("split_mnist", GCStorage())
    aggregate_task_results("cifar100_10", expected_size=10, remote_storage=GCStorage())
