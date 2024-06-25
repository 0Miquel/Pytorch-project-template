from .base_trainer import BaseTrainer
from src.metrics import MetricMonitor


class TemplateTrainer(BaseTrainer):
    def __init__(
            self,
            config,
            train_dl,
            val_dl,
            model,
            optimizer=None,
            criterion=None,
            scheduler=None,
            test_dl=None,
            *args,
            **kwargs
    ):
        """
        Trainer class.
        :param config:
        :param train_dl:
        :param val_dl:
        :param model:
        :param optimizer:
        :param criterion:
        :param scheduler:
        :param test_dl:
        :param args:
        :param kwargs:
        """
        super().__init__(
            config=config,
            train_dl=train_dl,
            val_dl=val_dl,
            test_dl=test_dl,
            criterion=criterion,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler
        )

    def compute_metrics(self, metric_monitor: MetricMonitor, output, batch) -> dict:
        """
        Update metric_monitor with the metrics computed from output and batch.
        :param metric_monitor:
        :param output:
        :param batch:
        :return:
        """
        return metric_monitor.get_metrics()
