from re import Pattern
from typing import List, Tuple

from matplotlib.figure import Figure
from pandas import read_csv
from seaborn import FacetGrid, relplot

from tricicl.loggers.tb_aggregation import get_aggregate_csv_file


def display_aggregated_results(
    task_name: str,
    *,
    use_simplified_metric_name: bool = False,
    metrics_names: List[str] = None,
    exclude_regex: Pattern = None,
    include_regex: Pattern = None,
    renames: List[Tuple[str, str]] = None,
    n_steps: int,
):
    df = read_csv(get_aggregate_csv_file(task_name))

    if use_simplified_metric_name:
        df["metric"] = df["metric"].map(lambda s: s.replace("/eval_phase/test_stream", "").replace("/Task000", ""))

    if exclude_regex:
        df = df[df["run_algo"].map(lambda s: exclude_regex.match(s) is None)]
    if include_regex:
        df = df[df["run_algo"].map(lambda s: include_regex.match(s) is not None)]

    for algo_name, replacement in renames or []:
        df["run_algo"] = df["run_algo"].map(lambda s: replacement if s == algo_name else s)

    algo_name2score = dict(
        df[(df["step"] == n_steps) & (df["metric"] == "Top1_Acc_Stream")]
        .groupby("run_algo")
        .mean()["value"]
        .iteritems()
    )
    df["run_algo"] = df["run_algo"].map(lambda name: f"{name} ({algo_name2score[name]:.1%})")

    all_metrics_names = sorted(set(df["metric"]), key=_get_metric_name_priority)
    print(all_metrics_names)
    metrics_names = metrics_names or all_metrics_names

    g: FacetGrid = relplot(
        data=df,
        kind="line",
        x="step",
        y="value",
        hue="run_algo",
        col="metric",
        col_order=metrics_names,
        col_wrap=min(3, len(metrics_names)),
        facet_kws={"sharex": False, "sharey": False, "legend_out": False},
    )
    fig: Figure = g.fig
    fig.suptitle(task_name, fontsize=16)
    fig.tight_layout()
    fig.show()


def _get_metric_name_priority(name: str) -> Tuple[bool, str]:
    return name[-1].isnumeric(), name
