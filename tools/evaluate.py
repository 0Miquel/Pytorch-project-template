import hydra

# Import some packages for off-the-shelf modules
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import torch.nn as nn

# Import from src for hand-crafted modules
from src.models import TemplateModel
from src.datasets import TemplateDataset
from src.trainers import TemplateTrainer
# from src.losses import ...
# from src.optimizers import ...


@hydra.main(version_base=None, config_path=".", config_name="config.yaml")
def main(cfg):
    """
    Load dataset and model from config and evaluate the model on the validation set.
    """
    # Define transformations
    transforms_val = A.Compose([
        A.Resize(width=64, height=64),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    # Create the dataset
    valid_dataset = TemplateDataset(train=False, data_path=cfg.data_path, transforms=transforms_val)

    # Create the dataloaders
    val_dl = torch.utils.data.DataLoader(valid_dataset, batch_size=cfg.batch_size, shuffle=True)

    # Create the model
    model = TemplateModel(n_classes=cfg.n_classes)

    # Initialize trainer
    trainer = TemplateTrainer(
        config=cfg,
        val_dl=val_dl,
        model=model,
    )

    # Start training
    trainer.evaluate()


if __name__ == "__main__":
    main()
