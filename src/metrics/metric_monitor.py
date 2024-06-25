from collections import defaultdict


class MetricMonitor:
    def __init__(self) -> None:
        """
        Metric Monitor class. Accumulate and compute metrics.
        """
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

    def update(self, metric_name: str, val: float) -> None:
        """
        Update the metric with the given value.
        :param metric_name: name of the metric to update
        :param val: value to update the metric with
        :return:
        """
        metric = self.metrics[metric_name]

        metric["val"] += val
        metric["count"] += 1
        metric["avg"] = metric["val"] / metric["count"]

    def get_metrics(self) -> dict:
        """
        Get the metrics as a dictionary.
        :return: dictionary of metrics, with the metric name as key and the average value as value
        """
        return {metric_name: metric["avg"] for metric_name, metric in self.metrics.items()}
