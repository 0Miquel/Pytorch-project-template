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


@hydra.main(version_base=None, config_path="cfgs", config_name="config.yaml")
def main(cfg):
    """
    Load dataset, model, optimizer, scheduler loss function and more from config and train the model here.
    """
    # Define transformations
    transforms_test = A.Compose([
        A.Resize(width=64, height=64),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    # Create the dataset
    test_dataset = TemplateDataset(train=False, data_path=cfg.data_path, transforms=transforms_test)

    # Create the dataloaders
    test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=True)

    # Create the model
    model = TemplateModel(n_classes=cfg.n_classes)
    # Load the model weights
    model.load_state_dict(torch.load(cfg.model_ckpt))

    # Instantiate the loss function
    criterion = nn.CrossEntropyLoss()

    # Initialize trainer
    trainer = TemplateTrainer(
        config=cfg,
        test_dl=test_dl,
        criterion=criterion,
        model=model,
    )

    # Start evaluation
    trainer.evaluate()


if __name__ == "__main__":
    main()
