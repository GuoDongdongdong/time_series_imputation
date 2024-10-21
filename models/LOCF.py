import os
import numpy as np
import pandas as pd


from pypots.imputation import LOCF


from utils.tools import logger


class Model:
    def __init__(self, args):
        self.n_steps               = args.seq_len
        self.n_features            = args.features
        self.targets               = args.target
        self.first_step_imputation = args.LOCF_first_step_imputation
        self.checkpoints_path      = args.checkpoints_path
        self.model = LOCF(
            first_step_imputation=self.first_step_imputation
        )

    def impute(self, test_dataset):
        test_data               = test_dataset.observed_data
        observed_data           = test_dataset.observed_data
        observed_mask           = test_dataset.observed_mask
        gt_mask                 = test_dataset.ground_truth_mask
        test_data               = test_dataset.inverse(test_data)
        observed_data           = test_dataset.inverse(observed_data)
        test_data[gt_mask == 0] = np.nan
        # pypots requires input's shape should be like [B, L, D]
        test_data               = test_data.reshape(1, -1, self.n_features)
        input                   = {"X" : test_data}
        output                  = self.model.impute(input)
        output                  = output.reshape(-1, self.n_features)
        target_mask             = observed_mask - gt_mask
        MAE                     = np.sum(np.abs((observed_data - output) * target_mask))
        RMSE                    = np.sum(((observed_data - output) * target_mask) ** 2)
        logger.info(f"RMSE: {MAE / np.sum(target_mask)}")
        logger.info(f"MAE: {np.sqrt(RMSE / np.sum(target_mask))}")

        df                          = pd.DataFrame()
        df['date']                  = test_dataset.test_date
        observed_data[gt_mask == 0] = np.nan
        df[self.targets]            = observed_data
        df[[target +  '_imputation' for target in self.targets]] = output
        df.to_csv(os.path.join(self.checkpoints_path, 'result.csv'), index=False, float_format='%.2f')