from tricicl.loggers.tb_aggregation import aggregate_task_results
from tricicl.storage.gcloud import GCStorage

if __name__ == "__main__":
    aggregate_task_results("split_mnist", GCStorage())
