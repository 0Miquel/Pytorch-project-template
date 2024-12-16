from .base_trainer import BaseTrainer


class TemplateTrainer(BaseTrainer):
    def __init__(
            self,
            config,
            model,
            train_dl=None,
            val_dl=None,
            test_dl=None,
            optimizer=None,
            criterion=None,
            scheduler=None,
            seed=42,
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
            test_dl=test_dl,
            criterion=criterion,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            seed=seed,
        )
