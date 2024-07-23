import os
import random

import torch
import numpy as np
import pandas as pd
from pygrinder import mcar

# pypots logger
from pypots.utils.logging import logger_creator, logger


class EarlyStopping:
    def __init__(self, patience, delta=0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

        self.lr_decrse = False

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.stop = True
            self.lr_decrse = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0
            self.lr_decrse = False

    def save_checkpoint(self, val_loss, model, path):
        logger.info(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join(path, 'checkpoint.pth'))
        self.val_loss_min = val_loss


def fix_random_seed(random_seed : int) :
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def set_logger(saving_dir: str, file_name: str = 'logfile') :
    format = "[%(asctime)s][%(levelname)s][%(filename)s:%(lineno)d] %(message)s"
    logger_creator.set_logging_format(format)
    logger_creator.set_saving_path(saving_dir, file_name)
    logger_creator.set_level('debug')

"""
    If our original dataset has no missing values, we need to simulate the situation where the dataset has missing values. Note that we will never know these masked data and cannot use them for model testing.
    There are various types of missing data in the original data set. Here we use completely random missing(MCAR).
    TODO: What impact will other situations have on filling if the original missing type of the dataset is not completely random missing?
"""
def mask_original_dataset(dir_name:str, file_name:str, missing_rate: float, targets: list[str]):
    file_path = os.path.join(dir_name, file_name)
    data = pd.read_csv(file_path)

    for col in targets:
        missing_col = col + '_missing'
        data[missing_col] = mcar(np.array(data[col]), p=missing_rate)

    file_name, file_suffix = file_name.split('.')
    data.to_csv(f'{file_name}_{int(missing_rate * 100)}per_missing.{file_suffix}', index=False)

"""
    The observed data in data are completely randomly masked with probability p
"""
def add_missing(data : np.ndarray, p : float):
    observed_mask = (1 - np.isnan(data)).reshape(-1)
    observed_index = np.where(observed_mask)[0]
    artifical_missing_index = np.random.choice(observed_index,
                                               int(len(observed_index) * p),
                                               replace=False)
    artifical_mask = observed_mask.copy()
    artifical_mask[artifical_missing_index] = False
    artifical_mask = artifical_mask.reshape(data.shape)
    return artifical_mask