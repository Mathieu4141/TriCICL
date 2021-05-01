from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from numpy import array, mean, ndarray
from pandas import DataFrame
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter

from tricicl.constants import CSV_DIR, TB_DIR
from tricicl.storage.local_storage import LocalStorage
from tricicl.storage.remote_storage import RemoteStorageABC, SyncPath
from tricicl.utils.path import empty_dir


def aggregate_task_results(task_name: str, *, expected_size: int, remote_storage: RemoteStorageABC = LocalStorage()):
    remote_storage.download_directory(SyncPath.from_local(TB_DIR / task_name))
    results = _load_task_results(task_name, expected_size)
    _aggregate_task_results_to_csv(task_name, results)
    _aggregate_task_results_to_tb(task_name, results)


def get_aggregate_csv_file(task_name: str) -> Path:
    return (CSV_DIR / task_name).with_suffix(".csv")


@dataclass
class RunResult:
    seed: str
    steps: ndarray
    values: ndarray

    def __iter__(self) -> Iterable[Tuple[int, float]]:
        return zip(self.steps, self.values)


@dataclass
class AlgoResults:
    name: str
    metric_name2results: Dict[str, List[RunResult]] = field(default_factory=lambda: defaultdict(list))

    def __iter__(self) -> Iterable[Tuple[str, List[RunResult]]]:
        return self.metric_name2results.items().__iter__()


def _load_task_results(task_name: str, expected_size: int) -> List[AlgoResults]:
    return [_load_algo_results(algo_dir, expected_size) for algo_dir in _get_all_algo_dirs(task_name)]


def _load_algo_results(algo_dir: Path, expected_size: int) -> AlgoResults:
    res = AlgoResults(algo_dir.name)
    for run_dir in algo_dir.glob("*"):
        summary = EventAccumulator(str(run_dir)).Reload()

        if (
            "Top1_Acc_Stream/eval_phase/test_stream" not in summary.Tags()["scalars"]
            or len(summary.Scalars("Top1_Acc_Stream/eval_phase/test_stream")) < expected_size
        ):
            print(f"Skipping incomplete run {run_dir.parent.name}/{run_dir.name}")
            continue

        for metric_name in summary.Tags()["scalars"]:
            summary.Scalars(metric_name)
            res.metric_name2results[metric_name].append(
                RunResult(
                    seed=run_dir.name,
                    steps=array([event.step for event in summary.Scalars(metric_name)]),
                    values=array([event.value for event in summary.Scalars(metric_name)]),
                )
            )
    return res


def _aggregate_task_results_to_csv(task_name: str, task_results: List[AlgoResults]):
    df = DataFrame.from_records(
        [
            {
                "run_seed": result.seed,
                "run_algo": algo_results.name,
                "metric": metric_name,
                "step": step,
                "value": value,
            }
            for algo_results in task_results
            for metric_name, results in algo_results
            for result in results
            for step, value in result
        ]
    )
    df.to_csv(get_aggregate_csv_file(task_name), index=False)


def _aggregate_task_results_to_tb(task_name: str, task_results: List[AlgoResults]):
    for algo_results in task_results:
        _aggregate_algo_results_to_tb(task_name, algo_results)


def _aggregate_algo_results_to_tb(task_name: str, algo_results: AlgoResults):
    with SummaryWriter(str(empty_dir(TB_DIR / task_name / (algo_results.name + "-mean")))) as writer:
        for metric_name, results in algo_results:
            for step, value in _average_results(results):
                writer.add_scalar(metric_name, value, step)


def _average_results(results: List[RunResult]) -> RunResult:
    steps = results[0].steps
    for res in results[:1]:
        assert (steps == res.steps).all()
    return RunResult(seed="mean", steps=steps, values=mean([r.values for r in results], axis=0))


def _get_all_algo_dirs(task_name: str) -> Iterable[Path]:
    return (algo_dir for algo_dir in (TB_DIR / task_name).glob("*") if not algo_dir.name.endswith("-mean"))
