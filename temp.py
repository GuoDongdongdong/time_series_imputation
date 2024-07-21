import os
import argparse
import configparser

import torch
import numpy as np
import pandas as pd

from utils.tools import mask_original_dataset, add_missing
from torch.utils.data import Dataset, DataLoader


def generate_missing_datasets():
    missing_rate = 0.2
    # humidity
    dir_name = 'dataset/humidity'
    file_name = 'humidity.csv'
    cols = ['humidity']
    mask_original_dataset(dir_name, file_name, missing_rate, cols)
    # water
    dir_name = 'dataset/water_salt'
    file_name = 'water.csv'
    cols = ['water']
    mask_original_dataset(dir_name, file_name, missing_rate, cols)
    # wind and temperature
    dir_name = 'dataset/wind'
    file_name = 'wind.csv'
    cols = ['windSpeed3s','windSpeed2m','windSpeed10m','temperature']
    mask_original_dataset(dir_name, file_name, missing_rate, cols)

class MyDataset(Dataset):
    def __init__(self):
        self.x = np.arange(100)
        self.seq_len = 3

    def __getitem__(self, index):
        return self.x[index: index + self.seq_len]

    def __len__(self):
        return len(self.x) - self.seq_len + 1


if __name__ == '__main__':
    pass