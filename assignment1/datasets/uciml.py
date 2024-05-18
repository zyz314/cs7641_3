import os
import pandas as pd

from torch import is_tensor, Tensor
from torch.utils.data import Dataset

class AdultDataset(Dataset):
    """
    Class representing the Adult dataset from UC Irvine ML archive
    """
    def __init__(self, data_dir='data', mode='test', transforms = None):
        """
        Load dataset from preprocessed pickle files
        """
        filename = 'uciml_adult_test.pkl' if mode == 'test' else 'uciml_adult_train.pkl'
        self.data = Tensor(pd.read_pickle(os.path.join(data_dir, filename)).values)
        self.transforms = transforms

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        if is_tensor(index):
            index = index.tolist()

        sample = self.data[index, :-1]
        # Label is base-1, convert to base-0
        label = self.data[index, -1] - 1

        if self.transforms:
            sample = self.transforms(sample)

        return (sample, label)


class DryBeanDataset(Dataset):
    """
    Class representing the Dry Bean dataset from UC Irvine ML archive
    """
    def __init__(self, data_dir='data', mode='test', transforms = None):
        """
        Load dataset from preprocessed pickle files
        """
        filename = 'uciml_drybean.pkl'
        self.data = Tensor(pd.read_pickle(os.path.join(data_dir, filename)).values)
        self.transforms = transforms

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        if is_tensor(index):
            index = index.tolist()

        sample = self.data[index, :-1]
        label = self.data[index, -1]

        if self.transforms:
            sample = self.transforms(sample)

        return (sample, label)
