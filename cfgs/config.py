from dataclasses import dataclass


@dataclass
class CFG:
    """
    Default configuration, overwritten by the .yaml file defined in folder cfgs/
    """
    # Dataset config
    data_path: str = None

    # Early stopping and model checkpoint config
    patience: int = 10000
    min_delta: float =  0.0
    max_mode: bool = False
    monitor: str = "loss"

    # Model config
    n_classes: int = 10

    # Trainer config
    batch_size: int = 64
    n_epochs: int = 1
    device: str = "cuda"
    wandb: str =  None

    # Optimizer config
    lr: float = 0.001
    max_lr: float = 0.01