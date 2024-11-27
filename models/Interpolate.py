import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from utils.tools import logger

class Model:
    def __init__(self, args):
        self.targets = args.target
        self.checkpoints_path = args.checkpoints_path
        self.kind = args.Interpolate_kind

    def impute(self, test_dataset):
        test_data = test_dataset.observed_data
        observed_data = test_dataset.observed_data
        observed_mask = test_dataset.observed_mask
        gt_mask = test_dataset.ground_truth_mask
        test_data = test_dataset.inverse(test_data)
        observed_data = test_dataset.inverse(observed_data)
        test_data[gt_mask == 0] = np.nan
        L, D = test_data.shape
        for _ in range(D):
            data = test_data[:, _]
            nan_idx = np.argwhere(~np.isnan(data)).squeeze()
            x = np.arange(1, L + 1)
            x = x[nan_idx]
            data_without_nan = data[nan_idx]
            interpolate_func = interp1d(x, data_without_nan, kind=self.kind, axis=0)
            axis_y = np.linspace(1, L, L)
            test_data[:, _] = interpolate_func(axis_y)
        
        target_mask = observed_mask - gt_mask
        MAE = np.sum(np.abs((observed_data - test_data) * target_mask))
        RMSE = np.sum(((observed_data - test_data) * target_mask) ** 2)
        logger.info(f"RMSE: {MAE / np.sum(target_mask)}")
        logger.info(f"MAE: {np.sqrt(RMSE / np.sum(target_mask))}")

        df = pd.DataFrame()
        df['date'] = test_dataset.test_date
        observed_data[gt_mask == 0] = np.nan
        df[self.targets] = observed_data
        df[[target +  '_imputation' for target in self.targets]] = test_data
        df.to_csv(os.path.join(self.checkpoints_path, 'result.csv'), index=False, float_format='%.2f')
