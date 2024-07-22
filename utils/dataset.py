import os
import pickle

import torch
import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

from utils.tools import add_missing


class CustomDataset(Dataset):
    def __init__(self, args, flag):
        assert flag in ['train', 'validation', 'test']
        self.args = args
        flag_dict = {
            'train'      : 0,
            'validation' : 1,
            'test'       : 2
        }
        self.flag = flag_dict[flag]
        self.scaler = StandardScaler()
        self._read_dataset(flag)

    def _read_dataset(self, flag):
        path     = os.path.join(self.args.dataset_dir, self.args.dataset_file)
        raw_data = pd.read_csv(path)
        raw_data = raw_data[['date'] + self.args.target]
        # split dataset to train vali test
        total_len     = len(raw_data)
        train_len     = int(total_len * self.args.train_ratio)
        vali_len      = int(total_len * self.args.vali_ratio)
        data_border_l = [0, train_len, train_len + vali_len]
        data_border_r = [train_len, train_len + vali_len, total_len]
        data          = raw_data[raw_data.columns[1:]]
        # normalize data
        train_data = data[data_border_l[0]:data_border_r[0]]
        self.scaler.fit(train_data)
        data       = self.scaler.transform(data)

        self.observed_data = data[data_border_l[self.flag]:data_border_r[self.flag]]
        self.observed_mask = 1 - np.isnan(self.observed_data)
        # artifical mask
        self.ground_truth_mask = add_missing(self.observed_data, self.args.missing_rate)

        self.observed_data = np.nan_to_num(self.observed_data)

        if flag == 'test':
            self.test_date  = raw_data['date'][train_len + vali_len:]

    def __getitem__(self, index):
        l, r = index, index + self.args.seq_len
        x = {
            'observed_data' : self.observed_data[l : r],
            'observed_mask' : self.observed_mask[l : r],
            'gt_mask'  : self.ground_truth_mask[l : r],
            'timepoints'    : np.arange(self.args.seq_len)
        }
        return x

    def __len__(self):
        return len(self.observed_data) - self.args.seq_len

    def inverse(self, data) :
        if torch.is_tensor(data):
            data = data.cpu()
        return self.scaler.inverse_transform(data)

    def save_result(self, observed_data, observed_mask, gt_mask, samples_data, impute_data):
        df = pd.DataFrame()
        df['date'] = self.test_date
        observed_data[observed_mask == 0 or gt_mask == 0] = np.nan
        df[self.args.target] = observed_data
        df[[target + '_imputation' for target in self.args.target]] = impute_data
        path = os.path.dirname(self.args.checkpoints_path)
        df.to_csv(os.path.join(path, 'result.csv'), index=False, float_format='%.2f')
        np.save(os.path.join(path, 'samples_data.npy'), samples_data)
        return 
        # [n_samples, B*L, D]
        n_samples, L, D = samples_data.shape
        lower_CI_list, higher_CI_list = [0] * n_samples, [0] * n_samples
        for i in range(n_samples):
            lower_CI_list[i], higher_CI_list[i] = st.t.interval(0.95,
                                                    L - 1,
                                                    loc=np.mean(impute_data[i, :, :], 0),
                                                    scale=st.sem(impute_data[i, :, :]))

def data_provider(args, flag : str):
    assert flag in ['train', 'validation', 'test']
    dataset = CustomDataset(args, flag)
    sampler = None if flag == 'train' else iter(range(0, len(dataset), args.seq_len))
    dataloader = DataLoader(dataset=dataset,
                            batch_size=args.batch_size,
                            shuffle=True if flag == 'train' else False,
                            num_workers=args.num_workers,
                            sampler=sampler
                            )
    return dataset, dataloader