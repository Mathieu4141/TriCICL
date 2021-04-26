import re

import seaborn

from tricicl.loggers.csv_aggregation import display_aggregated_results

if __name__ == "__main__":
    seaborn.set()
    display_aggregated_results(
        "split_mnist",
        use_simplified_metric_name=True,
        # exclude_regex=re.compile("(naive|LwF|tricicl|tl-post-|tl-alternate-last)", flags=re.IGNORECASE),
        include_regex=re.compile("(tl|tricicl)"),
    )
