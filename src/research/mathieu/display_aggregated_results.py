import re

import seaborn

from tricicl.loggers.csv_aggregation import display_aggregated_results

if __name__ == "__main__":
    seaborn.set()
    # display_aggregated_results(
    #     "split_mnist",
    #     use_simplified_metric_name=True,
    #     include_regex=re.compile("(tl-pre-PD$|tl-post-D|tl-alternate-last|iCaRL|hybrid1|replay)"),
    #     metrics_names=["Top1_Acc_Stream"],
    #     renames=[
    #         ("tl-pre-PD", "tricicl-P-D"),
    #         ("tl-post-D", "tricicl-B-D"),
    #         ("tl-alternate-last", "tricicl-A-D"),
    #     ],
    # )
    display_aggregated_results(
        "cifar100_10",
        use_simplified_metric_name=True,
        # exclude_regex=re.compile("(naive|LwF|tricicl|tl-post-|tl-alternate-last)", flags=re.IGNORECASE),
        # include_regex=re.compile("(tricicl-P-ND|tricicl-P-D$)"),
        include_regex=re.compile("(replay|hybrid1|iCaRL|tricicl-B-D|tricicl-P-D)"),
        metrics_names=["Top1_Acc_Stream"] + [f"Top1_Acc_Exp/Exp00{i}" for i in range(10)],
        n_steps=10,
    )
