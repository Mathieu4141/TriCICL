import os
from glob import glob
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy
from more_itertools import flatten
from numpy import array, ndarray
from pandas import DataFrame
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter

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


# Functions below are taken from https://pypi.org/project/tensorboard-reducer/
# There is a syntax error in python37 for tensorboard reducer
def read_tb_events(indirs_glob: str) -> Dict[str, ndarray]:
    indirs = glob(indirs_glob)
    assert len(indirs) > 0, f"No runs found for glob pattern '{indirs_glob}'"

    summary_iterators = [EventAccumulator(dirname).Reload() for dirname in indirs]

    tags = summary_iterators[0].Tags()["scalars"]

    for iterator in summary_iterators:
        # assert all runs have the same tags for scalar data
        assert iterator.Tags()["scalars"] == tags

    out_dict = {t: [] for t in tags}

    for tag in tags:
        for events in zip(*[acc.Scalars(tag) for acc in summary_iterators]):
            assert len({e.step for e in events}) == 1

            out_dict[tag].append([e.value for e in events])

    return {key: array(val) for key, val in out_dict.items()}


def reduce_events(events_dict: Dict[str, ndarray], reduce_ops: List[str]) -> Dict[str, Dict[str, ndarray]]:
    reductions = {}

    for op in reduce_ops:

        reductions[op] = {}

        for tag, arr in events_dict.items():

            reduce_op = getattr(numpy, op)

            reductions[op][tag] = reduce_op(arr, axis=-1)

    return reductions


def write_tb_events(
    data_to_write: Dict[str, Dict[str, ndarray]],
    outdir: str,
    overwrite: bool = False,
):
    # handle std reduction separately as we use writer.add_scalars to write mean +/- std
    if "std" in data_to_write.items():

        assert (
            "mean" in data_to_write.items()
        ), "cannot write data for std reduction as mean+/-std without mean reduction"

        std_dict = data_to_write.pop("std")
        mean_dict = data_to_write["mean"]

        std_dir = f"{outdir}-std"

        force_rm_or_raise(std_dir, overwrite)

        writer = SummaryWriter(std_dir)

        for (tag, means), stds in zip(mean_dict.items(), std_dict.values()):
            for idx, (mean, std) in enumerate(zip(means, stds)):
                writer.add_scalars(tag, {"mean+std": mean + std, "mean-std": mean - std}, idx)

        writer.close()

    # loop over each reduce operation (e.g. mean, min, max, median)
    for op, events_dict in data_to_write.items():

        op_outdir = f"{outdir}-{op}"

        force_rm_or_raise(op_outdir, overwrite)

        writer = SummaryWriter(op_outdir)

        for tag, data in events_dict.items():
            for idx, scalar in enumerate(data):
                writer.add_scalar(tag, scalar, idx)

        # Important for allowing write_events() to overwrite. Without it,
        # try_rmtree will raise OSError: [Errno 16] Device or resource busy
        # trying to delete the existing outdir.
        writer.close()


def force_rm_or_raise(path: str, overwrite: bool) -> None:
    if os.path.exists(path):  # True if dir is either file or directory

        # for safety, check dir is either TensorBoard run or CSV file
        # to make it harder to delete files not created by this program
        is_csv_file = path.endswith(".csv")
        is_tb_run_dir = os.path.isdir(path) and os.listdir(path)[0].startswith("events.out")

        if overwrite and (is_csv_file or is_tb_run_dir):
            os.system(f"rm -rf {path}")
        else:
            raise FileExistsError(f"'{path}' already exists, pass overwrite=True (-o in CLI) to proceed anyway")
