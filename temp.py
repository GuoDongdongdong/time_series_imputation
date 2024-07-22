import os
import argparse
import configparser

import torch
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as st


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


def plot():

    matplotlib.rcParams.update({'font.size': 12})

    # generate dataset
    data_points = 50
    sample_points = 10000
    Mu = (np.linspace(-5, 5, num=data_points)) ** 2
    Sigma = np.ones(data_points) * 8
    data = np.random.normal(loc=Mu, scale=Sigma, size=(100, data_points))

    # predicted expect and calculate confidence interval
    predicted_expect = np.mean(data, 0)
    low_CI_bound, high_CI_bound = st.t.interval(0.95, data_points - 1,
                                                loc=np.mean(data, 0),
                                                scale=st.sem(data))

    # plot confidence interval
    x = np.linspace(0, data_points - 1, num=data_points)
    plt.plot(predicted_expect, linewidth=3., label='estimated value')
    plt.plot(Mu, color='r', label='grand truth')
    plt.fill_between(x, low_CI_bound, high_CI_bound, alpha=0.5,
                    label='confidence interval')
    plt.legend()
    plt.title('Confidence interval')
    plt.show()


if __name__ == '__main__':
    plot()