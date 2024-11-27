"""
The core wrapper assembles the submodules of M-RNN imputation model
and takes over the forward progress of the algorithm.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from utils.tools import calc_rmse


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.n_steps = args.seq_len
        self.n_features = args.features
        self.rnn_hidden_size = args.MRNN_rnn_hidden_size
        self.backbone = BackboneMRNN(self.n_steps, self.n_features, self.rnn_hidden_size)

    def evaluate(self, batch:dict, training:bool=True) -> torch.Tensor:
        res = self.forward(batch, training)
        return res['loss']

    def impute(self, batch:dict, n_samples:int=None) -> torch.Tensor:
        res = self.forward(batch, False)
        return res['imputed_data']
    
    def forward(self, inputs: dict, training: bool = True) -> dict:
        X = inputs["forward"]["X"]
        M = inputs["forward"]["missing_mask"]

        RNN_estimation, RNN_imputed_data, FCN_estimation = self.backbone(inputs)

        imputed_data = M * X + (1 - M) * FCN_estimation
        results = {
            "imputed_data": imputed_data,
        }

        # if in training mode, return results with losses
        RNN_loss = calc_rmse(RNN_estimation, X, M)
        FCN_loss = calc_rmse(FCN_estimation, RNN_imputed_data)
        reconstruction_loss = RNN_loss + FCN_loss
        results["loss"] = reconstruction_loss

        return results


class BackboneMRNN(nn.Module):
    def __init__(self, n_steps, n_features, rnn_hidden_size):
        super().__init__()

        self.n_steps = n_steps
        self.n_features = n_features
        self.rnn_hidden_size = rnn_hidden_size

        self.f_rnn = nn.GRU(3, self.rnn_hidden_size, batch_first=True)
        self.b_rnn = nn.GRU(3, self.rnn_hidden_size, batch_first=True)
        self.concated_hidden_project = nn.Linear(self.rnn_hidden_size * 2, 1)
        self.fcn_regression = MrnnFcnRegression(n_features)

    def gene_hidden_states(self, inputs, feature_idx):
        X_f = inputs["forward"]["X"][:, :, feature_idx].unsqueeze(dim=2)
        M_f = inputs["forward"]["missing_mask"][:, :, feature_idx].unsqueeze(dim=2)
        D_f = inputs["forward"]["deltas"][:, :, feature_idx].unsqueeze(dim=2)
        X_b = inputs["backward"]["X"][:, :, feature_idx].unsqueeze(dim=2)
        M_b = inputs["backward"]["missing_mask"][:, :, feature_idx].unsqueeze(dim=2)
        D_b = inputs["backward"]["deltas"][:, :, feature_idx].unsqueeze(dim=2)
        device = X_f.device
        batch_size = X_f.size()[0]

        f_hidden_state_0 = torch.zeros(
            (1, batch_size, self.rnn_hidden_size), device=device
        )
        b_hidden_state_0 = torch.zeros(
            (1, batch_size, self.rnn_hidden_size), device=device
        )
        f_input = torch.cat([X_f, M_f, D_f], dim=2)
        b_input = torch.cat([X_b, M_b, D_b], dim=2)
        hidden_states_f, _ = self.f_rnn(f_input, f_hidden_state_0)
        hidden_states_b, _ = self.b_rnn(b_input, b_hidden_state_0)
        hidden_states_b = torch.flip(hidden_states_b, dims=[1])

        feature_estimation = self.concated_hidden_project(
            torch.cat([hidden_states_f, hidden_states_b], dim=2)
        )

        return feature_estimation, hidden_states_f, hidden_states_b

    def forward(self, inputs: dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        X = inputs["forward"]["X"]
        M = inputs["forward"]["missing_mask"]

        feature_collector = []
        for f in range(self.n_features):
            feat_estimation, hid_states_f, hid_states_b = self.gene_hidden_states(
                inputs, f
            )
            feature_collector.append(feat_estimation)

        RNN_estimation = torch.concat(feature_collector, dim=2)
        RNN_imputed_data = M * X + (1 - M) * RNN_estimation
        FCN_estimation = self.fcn_regression(X, M, RNN_imputed_data)
        return RNN_estimation, RNN_imputed_data, FCN_estimation


class MrnnFcnRegression(nn.Module):
    """M-RNN fully connection regression Layer"""

    def __init__(self, feature_num):
        super().__init__()
        self.U = Parameter(torch.Tensor(feature_num, feature_num))
        self.V1 = Parameter(torch.Tensor(feature_num, feature_num))
        self.V2 = Parameter(torch.Tensor(feature_num, feature_num))
        self.beta = Parameter(torch.Tensor(feature_num))  # bias beta
        self.final_linear = nn.Linear(feature_num, feature_num)

        m = torch.ones(feature_num, feature_num) - torch.eye(feature_num, feature_num)
        self.register_buffer("m", m)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.U.size(0))
        self.U.data.uniform_(-stdv, stdv)
        self.V1.data.uniform_(-stdv, stdv)
        self.V2.data.uniform_(-stdv, stdv)
        self.beta.data.uniform_(-stdv, stdv)

    def forward(self, x, missing_mask, target):
        h_t = torch.sigmoid(
            F.linear(x, self.U * self.m)
            + F.linear(target, self.V1 * self.m)
            + F.linear(missing_mask, self.V2)
            + self.beta
        )
        x_hat_t = torch.sigmoid(self.final_linear(h_t))
        return x_hat_t
