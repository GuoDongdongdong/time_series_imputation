import os
from typing import Union

import torch
import numpy as np
import pandas as pd
import scipy.stats as st
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

from utils.tools import add_missing, get_deltas, locf


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
        # csv file needs to have time column named 'date' and we ignore columns that are not in self.args.target
        raw_data = raw_data[['date'] + self.args.target]
        # split dataset to train vali test
        total_len     = len(raw_data)
        train_len     = int(total_len * self.args.train_ratio)
        vali_len      = int(total_len * self.args.vali_ratio)
        data_border_l = [0, train_len, train_len + vali_len]
        data_border_r = [train_len, train_len + vali_len, total_len]
        data          = raw_data[raw_data.columns[1 : ]]
        # normalize data
        train_data = data[data_border_l[0]:data_border_r[0]]
        self.scaler.fit(train_data)
        data       = self.scaler.transform(data)
        self.observed_data = data[data_border_l[self.flag]:data_border_r[self.flag]]
        self.observed_mask = 1 - np.isnan(self.observed_data)
        # artifical mask
        self.ground_truth_mask = add_missing(self.observed_data, self.args.missing_rate)
        # time_gap
        self.time_gap = get_deltas(self.ground_truth_mask)
        self.observed_data = np.nan_to_num(self.observed_data)

        if self.args.model == 'GRUD':
            unnormalized_data = self.scaler.inverse_transform(self.observed_data)
            L, D = unnormalized_data.shape
            self.X_LOCF = locf(unnormalized_data.reshape(-1, L, D))
            self.X_LOCF = self.X_LOCF.reshape(L, D)
            self.empirical_mean = np.sum(unnormalized_data * self.observed_mask, axis=0) / np.sum(self.observed_mask, axis=0)
            self.empirical_mean = np.nan_to_num(self.empirical_mean, 0)

        if flag == 'test':
            self.test_date  = raw_data['date'][train_len + vali_len:]

    def __getitem__(self, index):
        l, r = index, index + self.args.seq_len
        x = dict()
        model = self.args.model
        if model == 'BRITS' or model == 'MRNN':
            x = {
                'forward':{
                    'X'            : self.observed_data[l : r],
                    'missing_mask' : self.ground_truth_mask[l : r],
                    'deltas'       : self.time_gap[l : r]
                },
                'backward' : {
                    'X'            : self.observed_data[l : r],
                    'missing_mask' : self.ground_truth_mask[l : r],
                    'deltas'       : self.time_gap[l : r]
                }
            }
        elif model == 'SAITS' or model == 'TimesNet':
            x = {
                'X' : self.observed_data[l : r],
                'missing_mask' : self.ground_truth_mask[l : r],
                'X_ori' : self.observed_data[l : r],
                'indicating_mask' : self.observed_mask[l : r] - self.ground_truth_mask[l : r] 
            }
        elif model == 'CSDI':
            x = {
                'timepoints'    : np.arange(self.args.seq_len),
                'index'         : index
            }
        elif model == 'GRUD':
            x = {
                'X' : self.observed_data[l : r],
                'missing_mask' : self.ground_truth_mask[l : r],
                'deltas' : self.time_gap[l : r],
                'empirical_mean' : self.empirical_mean,
                'X_filledLOCF' : self.X_LOCF[l : r]
            }
        x['observed_data'] = self.observed_data[l : r]
        x['observed_mask'] = self.observed_mask[l : r]
        x['gt_mask']       = self.ground_truth_mask[l : r]
        return x

    def __len__(self):
        return len(self.observed_data) - self.args.seq_len

    def _inverse_ndarry(self, data : np.ndarray):
        assert data.ndim == 2, f"innverse need data's shape like [length, features], but got {data.shape}"
        return self.scaler.inverse_transform(data)

    def _inverse_tensor(self, data : torch.Tensor) :
        device = data.device
        res = self._inverse_ndarry(data.cpu().numpy())
        return torch.from_numpy(res).to(device)

    def inverse(self, data : Union[np.ndarray | torch.Tensor]) :
        if isinstance(data, np.ndarray):
            return self._inverse_ndarry(data)
        if isinstance(data, torch.Tensor):
            return self._inverse_tensor(data)
        raise TypeError(f'only support ndarry and Tensor, but got {type(data)}.')

    def save_result(self, observed_data, observed_mask, gt_mask, samples_data, impute_data):
        df = pd.DataFrame()
        BL, D = observed_data.shape
        df['date'] = self.test_date[0 : BL]
        observed_data[observed_mask == 0 or gt_mask == 0] = np.nan
        df[self.args.target] = observed_data
        df[[target + '_imputation' for target in self.args.target]] = impute_data
        df.to_csv(os.path.join(self.args.checkpoints_path, 'result.csv'), index=False, float_format='%.2f')
        np.save(os.path.join(self.args.checkpoints_path, 'samples_data.npy'), samples_data)
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