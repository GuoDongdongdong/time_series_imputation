"""
The core wrapper assembles the submodules of GRU-D imputation model
and takes over the forward progress of the algorithm.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter

from utils.tools import calc_mse


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.n_steps = args.seq_len
        self.n_features = args.features
        self.rnn_hidden_size = args.GRUD_rnn_hidden_size

        # create models
        self.backbone = BackboneGRUD(
            self.n_steps,
            self.n_features,
            self.rnn_hidden_size,
        )
        self.output_projection = nn.Linear(self.rnn_hidden_size, self.n_features)

    def evaluate(self, batch:dict, training:bool=True) -> torch.Tensor:
        res = self.forward(batch, training)
        return res['loss']

    def impute(self, batch:dict, n_samples:int=None) -> torch.Tensor:
        res = self.forward(batch, False)
        return res['imputed_data']

    def forward(self, inputs: dict, training: bool = True) -> dict:
        """Forward processing of GRU-D.

        Parameters
        ----------
        inputs :
            The input data.

        training :
            Whether in training mode.

        Returns
        -------
        dict,
            A dictionary includes all results.
        """
        X = inputs["X"]
        missing_mask = inputs["missing_mask"]
        deltas = inputs["deltas"]
        empirical_mean = inputs["empirical_mean"]
        X_filledLOCF = inputs["X_filledLOCF"]

        hidden_states, _ = self.backbone(
            X, missing_mask, deltas, empirical_mean, X_filledLOCF
        )

        # project back the original data space
        reconstruction = self.output_projection(hidden_states)

        imputed_data = missing_mask * X + (1 - missing_mask) * reconstruction
        results = {
            "imputed_data": imputed_data,
        }

        results["loss"] = calc_mse(reconstruction, X, missing_mask)

        return results


class BackboneGRUD(nn.Module):
    def __init__(
        self,
        n_steps: int,
        n_features: int,
        rnn_hidden_size: int,
    ):
        super().__init__()
        self.n_steps = n_steps
        self.n_features = n_features
        self.rnn_hidden_size = rnn_hidden_size

        # create models
        self.rnn_cell = nn.GRUCell(
            self.n_features * 2 + self.rnn_hidden_size, self.rnn_hidden_size
        )
        self.temp_decay_h = TemporalDecay(
            input_size=self.n_features, output_size=self.rnn_hidden_size, diag=False
        )
        self.temp_decay_x = TemporalDecay(
            input_size=self.n_features, output_size=self.n_features, diag=True
        )

    def forward(
        self, X, missing_mask, deltas, empirical_mean, X_filledLOCF
    ) -> Tuple[torch.Tensor, ...]:
        """Forward processing of GRU-D.

        Parameters
        ----------
        X:

        missing_mask:

        deltas:

        empirical_mean:

        X_filledLOCF:

        Returns
        -------
        classification_pred:

        logits:


        """

        hidden_state = torch.zeros((X.size()[0], self.rnn_hidden_size), device=X.device)

        representation_collector = []
        for t in range(self.n_steps):
            # for data, [batch, time, features]
            x = X[:, t, :]  # values
            m = missing_mask[:, t, :]  # mask
            d = deltas[:, t, :]  # delta, time gap
            x_filledLOCF = X_filledLOCF[:, t, :]

            gamma_h = self.temp_decay_h(d)
            gamma_x = self.temp_decay_x(d)
            hidden_state = hidden_state * gamma_h
            representation_collector.append(hidden_state)

            x_h = gamma_x * x_filledLOCF + (1 - gamma_x) * empirical_mean
            x_replaced = m * x + (1 - m) * x_h
            data_input = torch.cat([x_replaced, hidden_state, m], dim=1)
            hidden_state = self.rnn_cell(data_input, hidden_state)

        representation_collector = torch.stack(representation_collector, dim=1)

        return representation_collector, hidden_state


class TemporalDecay(nn.Module):
    """The module used to generate the temporal decay factor gamma in the GRU-D model.
    Please refer to the original paper :cite:`che2018GRUD` for more details.

    Attributes
    ----------
    W: tensor,
        The weights (parameters) of the module.
    b: tensor,
        The bias of the module.

    Parameters
    ----------
    input_size : int,
        the feature dimension of the input

    output_size : int,
        the feature dimension of the output

    diag : bool,
        whether to product the weight with an identity matrix before forward processing

    References
    ----------
    .. [1] `Che, Zhengping, Sanjay Purushotham, Kyunghyun Cho, David Sontag, and Yan Liu.
        "Recurrent neural networks for multivariate time series with missing values."
        Scientific reports 8, no. 1 (2018): 6085.
        <https://www.nature.com/articles/s41598-018-24271-9.pdf>`_

    """

    def __init__(self, input_size: int, output_size: int, diag: bool = False):
        super().__init__()
        self.diag = diag
        self.W = Parameter(torch.Tensor(output_size, input_size))
        self.b = Parameter(torch.Tensor(output_size))

        if self.diag:
            assert input_size == output_size
            m = torch.eye(input_size, input_size)
            self.register_buffer("m", m)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        std_dev = 1.0 / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-std_dev, std_dev)
        if self.b is not None:
            self.b.data.uniform_(-std_dev, std_dev)

    def forward(self, delta: torch.Tensor) -> torch.Tensor:
        """Forward processing of this NN module.

        Parameters
        ----------
        delta : tensor, shape [n_samples, n_steps, n_features]
            The time gaps.

        Returns
        -------
        gamma : tensor, of the same shape with parameter `delta`, values in (0,1]
            The temporal decay factor.
        """
        if self.diag:
            gamma = F.relu(F.linear(delta, self.W * Variable(self.m), self.b))
        else:
            gamma = F.relu(F.linear(delta, self.W, self.b))
        gamma = torch.exp(-gamma)
        return gamma
