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
    Load dataset, model, optimizer, scheduler loss function and more from config and train the model here.
    """
    # Define transformations
    transforms_train = A.Compose([
        A.Resize(width=64, height=64),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    transforms_val = A.Compose([
        A.Resize(width=64, height=64),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    # Create the dataset
    train_dataset = TemplateDataset(train=True, data_path=cfg.data_path, transforms=transforms_train)
    valid_dataset = TemplateDataset(train=False, data_path=cfg.data_path, transforms=transforms_val)

    # Create the dataloaders
    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_dl = torch.utils.data.DataLoader(valid_dataset, batch_size=cfg.batch_size, shuffle=True)

    # Create the model
    model = TemplateModel(n_classes=cfg.n_classes)

    # Instantiate the loss function
    criterion = nn.CrossEntropyLoss()

    # Instantiate the optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg.max_lr,
                                                    total_steps=cfg.n_epochs * len(train_dl))

    # Initialize trainer
    trainer = TemplateTrainer(
        config=cfg,
        train_dl=train_dl,
        val_dl=val_dl,
        criterion=criterion,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    # Start training
    best_metric = trainer.fit()

    return best_metric


if __name__ == "__main__":
    main()
