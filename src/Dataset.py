import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, train_data, train_target):

        if type(train_data) == type(pd.DataFrame()):
            self.train = train_data.values
        elif type(train_data) == np.ndarray:
            self.train = train_data
        else:
            raise ValueError("Unsupported data type %s" % type(train_data))

        if type(train_target) == type(pd.Series()):
            self.target = train_target.values
        elif type(train_target) == np.ndarray:
            self.target = train_target
        else:
            raise ValueError("Unsupported data type %s" % type(train_target))

        if len(train_data) != len(train_target):
            raise ValueError("Train and target sizes mismatch")

        self.len = len(train_data)

    def __len__(self):
        return self.len

    def __getitem__(self, index):

        return torch.Tensor(self.train[index].astype(np.float64)), \
               torch.Tensor(np.asarray(self.target[index])).float()