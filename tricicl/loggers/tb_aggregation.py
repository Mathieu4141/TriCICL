from pathlib import Path
from typing import Any, Dict, Iterable, List

from more_itertools import flatten
from pandas import DataFrame
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboard_reducer import read_tb_events, reduce_events, write_tb_events

from tricicl.constants import CSV_DIR, TB_DIR


def aggregate_task_results(task_name: str):
    for algo_dir in _get_all_algo_dirs(task_name):
        _aggregate_algo_tb_results(algo_dir)
    _aggregate_task_results_to_csv(task_name)


def get_aggregate_csv_file(task_name: str) -> Path:
    return (CSV_DIR / task_name).with_suffix(".csv")


def _aggregate_algo_tb_results(algo_dir: Path):
    events_dict = read_tb_events(str(algo_dir / "*"))
    reduced_events = reduce_events(events_dict, ["mean"])

    write_tb_events(reduced_events, str(algo_dir), overwrite=True)


def _aggregate_task_results_to_csv(task_name: str):
    df = DataFrame.from_records(
        flatten(
            _get_run_records(run_dir) for algo_dir in _get_all_algo_dirs(task_name) for run_dir in algo_dir.glob("*")
        )
    )
    df.to_csv(get_aggregate_csv_file(task_name), index=False)


def _get_run_records(run_dir: Path) -> List[Dict[str, Any]]:
    summary = EventAccumulator(str(run_dir)).Reload()
    return [
        {
            "run_seed": run_dir.name,
            "run_algo": run_dir.parent.name,
            "metric": tag,
            "step": event.step,
            "value": event.value,
        }
        for tag in summary.Tags()["scalars"]
        for event in summary.Scalars(tag)
    ]


def _get_all_algo_dirs(task_name: str) -> Iterable[Path]:
    return (
        algo_dir
        for algo_dir in (TB_DIR / task_name).glob("*")
        if not algo_dir.name.endswith("-mean") and not algo_dir.name == "aggregate"
    )
