import os
import pandas as pd
import numpy as np

from torch import is_tensor, Tensor, LongTensor
from torch.utils.data import Dataset


class DatasetWrapper(Dataset):
    """
    Class to wrap raw X and y into a Dataset object
    """
    def __init__(self, X, y, num_classes) -> None:
        super().__init__()
        self.X = np.asarray(X.copy(), dtype=np.float32)
        self.y = np.asarray(y.copy(), dtype=int)
        self.num_classes = num_classes

    def get_num_classes(self):
        return self.num_classes
    
    def get_num_features(self):
        return self.X.shape[1]
    
    def get_dataframe(self):
        return pd.DataFrame(self.X)

    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, index):
        if is_tensor(index):
            index = index.tolist()

        sample = self.X[index]
        label = self.y[index]

        return (sample, label)

class AdultDataset(Dataset):
    """
    Class representing the Adult dataset from UC Irvine ML archive
    """
    def __init__(self, data_dir='data', mode='test', transforms = None):
        """
        Load dataset from preprocessed pickle files
        """
        filename = 'uciml_adult_test.pkl' if mode == 'test' else 'uciml_adult_train.pkl'
        self.df = pd.read_pickle(os.path.join(data_dir, filename)).dropna()
        self.data = Tensor(self.df.values)
        if transforms:
            temp = transforms(self.data[:, :-1])
            self.data[:, :-1] = temp

    def get_num_classes(self):
        return 2
    
    def get_num_features(self):
        return 14

    def get_dataframe(self):
        return pd.DataFrame(self.data[:, :-1].numpy())

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        if is_tensor(index):
            index = index.tolist()

        sample = self.data[index, :-1]
        label = self.data[index, -1].long() - 1

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
        self.df = pd.read_pickle(os.path.join(data_dir, filename))
        self.data = Tensor(self.df.values)
        if transforms:
            temp = transforms(self.data[:, :-1])
            self.data[:, :-1] = temp

    def get_num_classes(self):
        return 7
    
    def get_num_features(self):
        return 16

    def get_dataframe(self):
        return pd.DataFrame(self.data[:, :-1].numpy())

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        if is_tensor(index):
            index = index.tolist()

        sample = self.data[index, :-1]
        label = self.data[index, -1].long()

        return (sample, label)
