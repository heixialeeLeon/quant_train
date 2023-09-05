import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import pickle


class StockDailyData(Dataset):
    def __init__(self, config, train_flag = True):
        if train_flag:
            self.load_pickle(config.train_data)
        else:
            self.load_pickle(config.val_data)

    def __getitem__(self, index):
        input_data = torch.from_numpy(self.data[index]["input_data"])
        return_1 = self.data[index]["return_1"]
        label_P = torch.tensor([return_1])
        if return_1 >= 0:
            label_C = torch.LongTensor([1])
        else:
            label_C = torch.LongTensor([0])
        return input_data, label_P, label_C

    def __len__(self):
        return len(self.data)

    def load_pickle(self, file_path):
        self.data = list()
        with open(file_path, "rb") as fp:
            while True:
                try:
                    self.data.append(pickle.load(fp))
                except EOFError:
                    break