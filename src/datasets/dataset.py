from torch.utils.data import Dataset


class TemplateDataset(Dataset):
    def __init__(self, train, data_path, transforms, *args, **kwargs):
        """
        Dataset class.
        :param train: boolean flag to indicate if the dataset is for training or validation
        :param data_path: path to the dataset
        :param transforms: data transforms
        """
        pass

    def __len__(self) -> int:
        """
        Data length function for the dataset.
        :return:
        """
        return 0

    def __getitem__(self, idx) -> dict:
        """
        Data extractor function for the dataset.
        :param idx: index of the data sample to be extracted
        :return:
        """
        return {}

    @staticmethod
    def collate_fn(batch) -> dict:
        """
        Data collector function for the dataset.
        :param batch: batch of data samples to be collected
        :return:
        """
        return {}
