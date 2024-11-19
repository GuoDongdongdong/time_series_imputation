import os
import random

import torch
import numpy as np
import pandas as pd
from typing import Union, Optional
from pygrinder import mcar

# pypots logger
from pypots.utils.logging import logger_creator, logger


STATISTICAL_MODEL_LIST    = ['LOCF']
GENERATIVE_MODEL_LIST     = ['CSDI']
DISCRIMINATIVE_MODEL_LIST = ['SAITS', 'BRITS', 'Transformer']


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


"""
    BRITS model need time gap matrix.
    The following function return deltas matrix (i.e. time gap matrix) from mask matrix.
"""
def get_deltas(mask : np.ndarray):
    assert mask.ndim == 2, f"mask shape should like [L, D], but got shape: {mask.shape}"
    def func(col):
        res = np.zeros_like(col, dtype=np.float32)
        for i in range(1, len(col)):
            if col[i] != 0:
                res[i] = 1.0
            else :
                res[i] = 1.0 + res[i - 1]
        return res
    deltas = np.apply_along_axis(func, 0, mask)
    return deltas


"""
Evaluation metrics related to error calculation (like in tasks regression, imputation etc).
"""

def _check_inputs(
    predictions: Union[np.ndarray, torch.Tensor, list],
    targets: Union[np.ndarray, torch.Tensor, list],
    masks: Optional[Union[np.ndarray, torch.Tensor, list]] = None,
    check_shape: bool = True,
):
    # check type
    assert isinstance(predictions, type(targets)), (
        f"types of `predictions` and `targets` must match, but got"
        f"`predictions`: {type(predictions)}, `target`: {type(targets)}"
    )
    lib = np if isinstance(predictions, np.ndarray) else torch
    # check shape
    prediction_shape = predictions.shape
    target_shape = targets.shape
    if check_shape:
        assert (
            prediction_shape == target_shape
        ), f"shape of `predictions` and `targets` must match, but got {prediction_shape} and {target_shape}"
    # check NaN
    assert not lib.isnan(
        predictions
    ).any(), "`predictions` mustn't contain NaN values, but detected NaN in it"
    assert not lib.isnan(
        targets
    ).any(), "`targets` mustn't contain NaN values, but detected NaN in it"

    if masks is not None:
        # check type
        assert isinstance(masks, type(targets)), (
            f"types of `masks`, `predictions`, and `targets` must match, but got"
            f"`masks`: {type(masks)}, `targets`: {type(targets)}"
        )
        # check shape, masks shape must match targets
        mask_shape = masks.shape
        assert mask_shape == target_shape, (
            f"shape of `masks` must match `targets` shape, "
            f"but got `mask`: {mask_shape} that is different from `targets`: {target_shape}"
        )
        # check NaN
        assert not lib.isnan(
            masks
        ).any(), "`masks` mustn't contain NaN values, but detected NaN in it"

    return lib


def calc_mae(
    predictions: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
    masks: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> Union[float, torch.Tensor]:
    """Calculate the Mean Absolute Error between ``predictions`` and ``targets``.
    ``masks`` can be used for filtering. For values==0 in ``masks``,
    values at their corresponding positions in ``predictions`` will be ignored.

    Parameters
    ----------
    predictions :
        The prediction data to be evaluated.

    targets :
        The target data for helping evaluate the predictions.

    masks :
        The masks for filtering the specific values in inputs and target from evaluation.
        When given, only values at corresponding positions where values ==1 in ``masks`` will be used for evaluation.

    Examples
    --------

    >>> import numpy as np
    >>> from pypots.utils.metrics import calc_mae
    >>> targets = np.array([1, 2, 3, 4, 5])
    >>> predictions = np.array([1, 2, 1, 4, 6])
    >>> mae = calc_mae(predictions, targets)

    mae = 0.6 here, the error is from the 3rd and 5th elements and is :math:`|3-1|+|5-6|=3`, so the result is 3/5=0.6.

    If we want to prevent some values from MAE calculation, e.g. the first three elements here,
    we can use ``masks`` to filter out them:

    >>> masks = np.array([0, 0, 0, 1, 1])
    >>> mae = calc_mae(predictions, targets, masks)

    mae = 0.5 here, the first three elements are ignored, the error is from the 5th element and is :math:`|5-6|=1`,
    so the result is 1/2=0.5.

    """
    # check shapes and values of inputs
    lib = _check_inputs(predictions, targets, masks)

    if masks is not None:
        return lib.sum(lib.abs(predictions - targets) * masks) / (
            lib.sum(masks) + 1e-12
        )
    else:
        return lib.mean(lib.abs(predictions - targets))


def calc_mse(
    predictions: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
    masks: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> Union[float, torch.Tensor]:
    """Calculate the Mean Square Error between ``predictions`` and ``targets``.
    ``masks`` can be used for filtering. For values==0 in ``masks``,
    values at their corresponding positions in ``predictions`` will be ignored.

    Parameters
    ----------
    predictions :
        The prediction data to be evaluated.

    targets :
        The target data for helping evaluate the predictions.

    masks :
        The masks for filtering the specific values in inputs and target from evaluation.
        When given, only values at corresponding positions where values ==1 in ``masks`` will be used for evaluation.

    Examples
    --------

    >>> import numpy as np
    >>> from pypots.utils.metrics import calc_mse
    >>> targets = np.array([1, 2, 3, 4, 5])
    >>> predictions = np.array([1, 2, 1, 4, 6])
    >>> mse = calc_mse(predictions, targets)

    mse = 1 here, the error is from the 3rd and 5th elements and is :math:`|3-1|^2+|5-6|^2=5`, so the result is 5/5=1.

    If we want to prevent some values from MSE calculation, e.g. the first three elements here,
    we can use ``masks`` to filter out them:

    >>> masks = np.array([0, 0, 0, 1, 1])
    >>> mse = calc_mse(predictions, targets, masks)

    mse = 0.5 here, the first three elements are ignored, the error is from the 5th element and is :math:`|5-6|^2=1`,
    so the result is 1/2=0.5.

    """
    # check shapes and values of inputs
    lib = _check_inputs(predictions, targets, masks)

    if masks is not None:
        return lib.sum(lib.square(predictions - targets) * masks) / (
            lib.sum(masks) + 1e-12
        )
    else:
        return lib.mean(lib.square(predictions - targets))


def calc_rmse(
    predictions: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
    masks: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> Union[float, torch.Tensor]:
    """Calculate the Root Mean Square Error between ``predictions`` and ``targets``.
    ``masks`` can be used for filtering. For values==0 in ``masks``,
    values at their corresponding positions in ``predictions`` will be ignored.

    Parameters
    ----------
    predictions :
        The prediction data to be evaluated.

    targets :
        The target data for helping evaluate the predictions.

    masks :
        The masks for filtering the specific values in inputs and target from evaluation.
        When given, only values at corresponding positions where values ==1 in ``masks`` will be used for evaluation.

    Examples
    --------

    >>> import numpy as np
    >>> from pypots.utils.metrics import calc_rmse
    >>> targets = np.array([1, 2, 3, 4, 5])
    >>> predictions = np.array([1, 2, 1, 4, 6])
    >>> rmse = calc_rmse(predictions, targets)

    rmse = 1 here, the error is from the 3rd and 5th elements and is :math:`|3-1|^2+|5-6|^2=5`,
    so the result is :math:`\\sqrt{5/5}=1`.

    If we want to prevent some values from RMSE calculation, e.g. the first three elements here,
    we can use ``masks`` to filter out them:

    >>> masks = np.array([0, 0, 0, 1, 1])
    >>> rmse = calc_rmse(predictions, targets, masks)

    rmse = 0.707 here, the first three elements are ignored, the error is from the 5th element and is :math:`|5-6|^2=1`,
    so the result is :math:`\\sqrt{1/2}=0.5`.

    """
    # don't have to check types and NaN here, since calc_mse() will do it
    lib = np if isinstance(predictions, np.ndarray) else torch
    return lib.sqrt(calc_mse(predictions, targets, masks))


def calc_mre(
    predictions: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
    masks: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> Union[float, torch.Tensor]:
    """Calculate the Mean Relative Error between ``predictions`` and ``targets``.
    ``masks`` can be used for filtering. For values==0 in ``masks``,
    values at their corresponding positions in ``predictions`` will be ignored.

    Parameters
    ----------
    predictions :
        The prediction data to be evaluated.

    targets :
        The target data for helping evaluate the predictions.

    masks :
        The masks for filtering the specific values in inputs and target from evaluation.
        When given, only values at corresponding positions where values ==1 in ``masks`` will be used for evaluation.

    Examples
    --------

    >>> import numpy as np
    >>> from pypots.utils.metrics import calc_mre
    >>> targets = np.array([1, 2, 3, 4, 5])
    >>> predictions = np.array([1, 2, 1, 4, 6])
    >>> mre = calc_mre(predictions, targets)

    mre = 0.2 here, the error is from the 3rd and 5th elements and is :math:`|3-1|+|5-6|=3`,
    so the result is :math:`\\sqrt{3/(1+2+3+4+5)}=1`.

    If we want to prevent some values from MRE calculation, e.g. the first three elements here,
    we can use ``masks`` to filter out them:

    >>> masks = np.array([0, 0, 0, 1, 1])
    >>> mre = calc_mre(predictions, targets, masks)

    mre = 0.111 here, the first three elements are ignored, the error is from the 5th element and is :math:`|5-6|^2=1`,
    so the result is :math:`\\sqrt{1/2}=0.5`.

    """
    # check shapes and values of inputs
    lib = _check_inputs(predictions, targets, masks)

    if masks is not None:
        return lib.sum(lib.abs(predictions - targets) * masks) / (
            lib.sum(lib.abs(targets * masks)) + 1e-12
        )
    else:
        return lib.sum(lib.abs(predictions - targets)) / (
            lib.sum(lib.abs(targets)) + 1e-12
        )


def calc_quantile_loss(predictions, targets, q: float, eval_points) -> float:
    quantile_loss = 2 * torch.sum(
        torch.abs(
            (predictions - targets) * eval_points * ((targets <= predictions) * 1.0 - q)
        )
    )
    return quantile_loss


def calc_quantile_crps(
    predictions: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
    masks: Union[np.ndarray, torch.Tensor],
    scaler_mean=0,
    scaler_stddev=1,
) -> float:
    """Continuous rank probability score for distributional predictions.

    Parameters
    ----------
    predictions :
        The prediction data to be evaluated.

    targets :
        The target data for helping evaluate the predictions.

    masks :
        The masks for filtering the specific values in inputs and target from evaluation.
        Only values at corresponding positions where values ==1 in ``masks`` will be used for evaluation.

    scaler_mean:
        Mean value of the scaler used to scale the data.

    scaler_stddev:
        Standard deviation value of the scaler used to scale the data.

    Returns
    -------
    CRPS :
        Value of continuous rank probability score.

    """
    # check shapes and values of inputs
    _ = _check_inputs(predictions, targets, masks, check_shape=False)

    if isinstance(predictions, np.ndarray):
        predictions = torch.from_numpy(predictions)
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets)
    if isinstance(masks, np.ndarray):
        masks = torch.from_numpy(masks)

    targets = targets * scaler_stddev + scaler_mean
    predictions = predictions * scaler_stddev + scaler_mean

    quantiles = np.arange(0.05, 1.0, 0.05)
    denominator = torch.sum(torch.abs(targets * masks))
    CRPS = torch.tensor(0.0)
    for i in range(len(quantiles)):
        q_pred = []
        for j in range(len(predictions)):
            q_pred.append(torch.quantile(predictions[j : j + 1], quantiles[i], dim=1))
        q_pred = torch.cat(q_pred, 0)
        q_loss = calc_quantile_loss(targets, q_pred, quantiles[i], masks)
        CRPS += q_loss / denominator
    return CRPS.item() / len(quantiles)


def calc_quantile_crps_sum(
    predictions: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
    masks: Union[np.ndarray, torch.Tensor],
    scaler_mean=0,
    scaler_stddev=1,
) -> float:
    """Sum continuous rank probability score for distributional predictions.

    Parameters
    ----------
    predictions :
        The prediction data to be evaluated.

    targets :
        The target data for helping evaluate the predictions.

    masks :
        The masks for filtering the specific values in inputs and target from evaluation.
        Only values at corresponding positions where values ==1 in ``masks`` will be used for evaluation.

    scaler_mean:
        Mean value of the scaler used to scale the data.

    scaler_stddev:
        Standard deviation value of the scaler used to scale the data.

    Returns
    -------
    CRPS :
        Sum value of continuous rank probability score.

    """
    # check shapes and values of inputs
    _ = _check_inputs(predictions, targets, masks, check_shape=False)

    if isinstance(predictions, np.ndarray):
        predictions = torch.from_numpy(predictions)
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets)
    if isinstance(masks, np.ndarray):
        masks = torch.from_numpy(masks)

    masks = masks.mean(-1)
    targets = targets * scaler_stddev + scaler_mean
    targets = targets.sum(-1)
    predictions = predictions * scaler_stddev + scaler_mean

    quantiles = np.arange(0.05, 1.0, 0.05)
    denominator = torch.sum(torch.abs(targets * masks))
    CRPS = torch.tensor(0.0)
    for i in range(len(quantiles)):
        q_pred = torch.quantile(predictions.sum(-1), quantiles[i], dim=1)
        q_loss = calc_quantile_loss(targets, q_pred, quantiles[i], masks)
        CRPS += q_loss / denominator
    return CRPS.item() / len(quantiles)
