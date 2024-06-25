import torch
import torch.nn as nn
from torchvision import models


class TemplateModel(nn.Module):
    def __init__(self, *args, **kwargs):
        """
        Model class.
        """
        super().__init__(*args, **kwargs)

    def forward(self, x):
        """
        Forward pass of the model.
        :param x:
        :return:
        """
        return x
