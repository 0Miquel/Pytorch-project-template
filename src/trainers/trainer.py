from .base_trainer import BaseTrainer


class TemplateTrainer(BaseTrainer):
    def __init__(
            self,
            config,
            train_dl,
            val_dl,
            model,
            optimizer,
            criterion=None,
            scheduler=None,
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
        """
        super().__init__(
            config=config,
            train_dl=train_dl,
            val_dl=val_dl,
            criterion=criterion,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler
        )
