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
):
    df = read_csv(get_aggregate_csv_file(task_name))

    if use_simplified_metric_name:
        df["metric"] = df["metric"].map(lambda s: s.replace("/eval_phase/test_stream", "").replace("/Task000", ""))

    if exclude_regex:
        df = df[df["run_algo"].map(lambda s: exclude_regex.match(s) is None)]
    if include_regex:
        df = df[df["run_algo"].map(lambda s: include_regex.match(s) is not None)]

    metrics_names = metrics_names or sorted(set(df["metric"]), key=_get_metric_name_priority)

    g: FacetGrid = relplot(
        data=df,
        kind="line",
        x="step",
        y="value",
        hue="run_algo",
        col="metric",
        col_order=metrics_names,
        col_wrap=min(3, len(metrics_names)),
        facet_kws={"sharex": False, "legend_out": False},
    )
    fig: Figure = g.fig
    fig.suptitle(task_name, fontsize=16)
    fig.tight_layout()
    fig.show()


def _get_metric_name_priority(name: str) -> Tuple[bool, str]:
    return name[-1].isnumeric(), name
