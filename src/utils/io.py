from yaml.loader import SafeLoader
import yaml
import torch
import random
import numpy as np


def load_yaml_config(path):
    with open(path) as f:
        data = yaml.load(f, Loader=SafeLoader)
    return data


def load_batch_to_device(batch, device):
    """
    Load batch to device recursively in case it finds a list or dictionary in such batch of data,
    """
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, dict):
        new_batch = {}
        for key, value in batch.items():
            new_batch[key] = load_batch_to_device(value, device)
        return new_batch
    elif isinstance(batch, list):
        new_batch = []
        for item in batch:
            new_batch.append(load_batch_to_device(item, device))
        return new_batch
    else:
        return batch


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
