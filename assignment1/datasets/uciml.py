import os
import pandas as pd

from torch import is_tensor
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
        self.data = pd.read_pickle(os.path.join(data_dir, filename))
        self.transforms = transforms

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        if is_tensor(index):
            index = index.tolist()
        
        sample = self.data.iloc[index]

        if self.transforms:
            sample = self.transforms(sample)

        return sample

# class PhishingDataset:
#     def __init__(self) -> None:
#         self.data_frame_ = None
    
#     def fetch(self) -> None:
#         self.data_frame_ = fetch_ucirepo(id=967)


