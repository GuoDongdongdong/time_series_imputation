import math
from typing import Tuple, Optional

import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F

from utils.tools import calc_mse

class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.seq_len = args.seq_len
        self.n_features = args.features
        
        self.n_layers = args.TimesNet_n_layers
        self.top_k = args.TimesNet_top_k
        self.d_model = args.TimesNet_d_model
        self.d_ffn = args.TimesNet_d_ffn
        self.n_kernels = args.TimesNet_n_kernels
        self.dropout = args.TimesNet_dropout
        self.apply_nonstationary_norm = args.TimesNet_apply_nonstationary_norm

        self.enc_embedding = DataEmbedding(
            self.n_features,
            self.d_model,
            dropout=self.dropout,
            n_max_steps=self.seq_len,
        )
        self.model = BackboneTimesNet(
            self.n_layers,
            self.seq_len,
            0,  # n_pred_steps should be 0 for the imputation task
            self.top_k,
            self.d_model,
            self.d_ffn,
            self.n_kernels,
        )
        self.layer_norm = nn.LayerNorm(self.d_model)

        # for the imputation task, the output dim is the same as input dim
        self.projection = nn.Linear(self.d_model, self.n_features)

    def evaluate(self, batch:dict, training:bool) -> torch.Tensor:
        res = self.forward(batch, training)
        return res['loss']
    
    def impute(self, batch:dict, n_sample:int=None)-> torch.Tensor:
        res = self.forward(batch, False)
        return res['imputed_data']

    def forward(self, inputs: dict, training: bool = True) -> dict:
        X, missing_mask = inputs["X"], inputs["missing_mask"]

        if self.apply_nonstationary_norm:
            # Normalization from Non-stationary Transformer
            X, means, stdev = nonstationary_norm(X, missing_mask)

        # embedding
        input_X = self.enc_embedding(X)  # [B,T,C]
        # TimesNet processing
        enc_out = self.model(input_X)

        # project back the original data space
        dec_out = self.projection(enc_out)

        if self.apply_nonstationary_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = nonstationary_denorm(dec_out, means, stdev)

        imputed_data = missing_mask * X + (1 - missing_mask) * dec_out
        results = {
            "imputed_data": imputed_data,
        }

        # `loss` is always the item for backward propagating to update the model
        loss = calc_mse(dec_out, inputs["X_ori"], inputs["indicating_mask"])
        results["loss"] = loss

        return results

"""
Embedding methods for Transformer models are put here.


This implementation is inspired by the official one https://github.com/zhouhaoyi/Informer2020/blob/main/models/embed.py
"""
class PositionalEncoding(nn.Module):
    """The original positional-encoding module for Transformer.

    Parameters
    ----------
    d_hid:
        The dimension of the hidden layer.

    n_positions:
        The max number of positions.

    """

    def __init__(self, d_hid: int, n_positions: int = 1000):
        super().__init__()
        pe = torch.zeros(n_positions, d_hid, requires_grad=False).float()
        position = torch.arange(0, n_positions).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_hid, 2).float()
            * -(torch.log(torch.tensor(10000)) / d_hid)
        ).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pos_table", pe)

    def forward(self, x: torch.Tensor, return_only_pos: bool = False) -> torch.Tensor:
        """Forward processing of the positional encoding module.

        Parameters
        ----------
        x:
            Input tensor.

        return_only_pos:
            Whether to return only the positional encoding.

        Returns
        -------
        If return_only_pos is True:
            pos_enc:
                The positional encoding.
        else:
            x_with_pos:
                Output tensor, the input tensor with the positional encoding added.
        """
        pos_enc = self.pos_table[:, : x.size(1)].clone().detach()

        if return_only_pos:
            return pos_enc

        x_with_pos = x + pos_enc
        return x_with_pos


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super().__init__()
        padding = 1 if torch.__version__ >= "1.5.0" else 2
        self.tokenConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=padding,
            padding_mode="circular",
            bias=False,
        )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super().__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type="fixed", freq="h"):
        super().__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == "fixed" else nn.Embedding
        if freq == "t":
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = (
            self.minute_embed(x[:, :, 4]) if hasattr(self, "minute_embed") else 0.0
        )
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, freq="h"):
        super().__init__()

        freq_map = {"h": 4, "t": 5, "s": 6, "m": 1, "a": 1, "w": 2, "d": 3, "b": 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(
        self,
        c_in,
        d_model,
        embed_type="fixed",
        freq="h",
        dropout=0.1,
        with_pos=True,
        n_max_steps=1000,
    ):
        super().__init__()

        self.with_pos = with_pos

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        if with_pos:
            self.position_embedding = PositionalEncoding(
                d_hid=d_model, n_positions=n_max_steps
            )
        self.temporal_embedding = (
            TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
            if embed_type != "timeF"
            else TimeFeatureEmbedding(d_model=d_model, freq=freq)
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_timestamp=None):
        if x_timestamp is None:
            x = self.value_embedding(x)
            if self.with_pos:
                x += self.position_embedding(x, return_only_pos=True)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_timestamp)
            if self.with_pos:
                x += self.position_embedding(x, return_only_pos=True)
        return self.dropout(x)


"""
Store normalization functions for neural networks.
"""
def nonstationary_norm(
    X: torch.Tensor,
    missing_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalization from Non-stationary Transformer. Please refer to :cite:`liu2022nonstationary` for more details.

    Parameters
    ----------
    X : torch.Tensor
        Input data to be normalized. Shape: (n_samples, n_steps (seq_len), n_features).

    missing_mask : torch.Tensor, optional
        Missing mask has the same shape as X. 1 indicates observed and 0 indicates missing.

    Returns
    -------
    X_enc : torch.Tensor
        Normalized data. Shape: (n_samples, n_steps (seq_len), n_features).

    means : torch.Tensor
        Means values for de-normalization. Shape: (n_samples, n_features) or (n_samples, 1, n_features).

    stdev : torch.Tensor
        Standard deviation values for de-normalization. Shape: (n_samples, n_features) or (n_samples, 1, n_features).

    """
    if torch.isnan(X).any():
        if missing_mask is None:
            missing_mask = torch.isnan(X)
        else:
            raise ValueError("missing_mask is given but X still contains nan values.")

    if missing_mask is None:
        means = X.mean(1, keepdim=True).detach()
        X_enc = X - means
        variance = torch.var(X_enc, dim=1, keepdim=True, unbiased=False) + 1e-9
        stdev = torch.sqrt(variance).detach()
    else:
        # for data contain missing values, add a small number to avoid dividing by 0
        missing_sum = torch.sum(missing_mask == 1, dim=1, keepdim=True) + 1e-9
        means = torch.sum(X, dim=1, keepdim=True) / missing_sum
        X_enc = X - means
        X_enc = X_enc.masked_fill(missing_mask == 0, 0)
        variance = torch.sum(X_enc * X_enc, dim=1, keepdim=True) + 1e-9
        stdev = torch.sqrt(variance / missing_sum)

    X_enc /= stdev
    return X_enc, means, stdev


def nonstationary_denorm(
    X: torch.Tensor,
    means: torch.Tensor,
    stdev: torch.Tensor,
) -> torch.Tensor:
    """De-Normalization from Non-stationary Transformer. Please refer to :cite:`liu2022nonstationary` for more details.

    Parameters
    ----------
    X : torch.Tensor
        Input data to be de-normalized. Shape: (n_samples, n_steps (seq_len), n_features).

    means : torch.Tensor
        Means values for de-normalization . Shape: (n_samples, n_features) or (n_samples, 1, n_features).

    stdev : torch.Tensor
        Standard deviation values for de-normalization. Shape: (n_samples, n_features) or (n_samples, 1, n_features).

    Returns
    -------
    X_denorm : torch.Tensor
        De-normalized data. Shape: (n_samples, n_steps (seq_len), n_features).

    """
    assert (
        len(X) == len(means) == len(stdev)
    ), "Input data and normalization parameters should have the same number of samples."
    if len(means.shape) == 2:
        means = means.unsqueeze(1)
    if len(stdev.shape) == 2:
        stdev = stdev.unsqueeze(1)

    X = X * stdev  # (stdev.repeat(1, n_steps, 1))
    X = X + means  # (means.repeat(1, n_steps, 1))
    return X


class BackboneTimesNet(nn.Module):
    def __init__(
        self,
        n_layers,
        n_steps,
        n_pred_steps,
        top_k,
        d_model,
        d_ffn,
        n_kernels,
    ):
        super().__init__()

        self.seq_len = n_steps
        self.n_layers = n_layers

        self.n_pred_steps = n_pred_steps
        self.model = nn.ModuleList(
            [
                TimesBlock(n_steps, n_pred_steps, top_k, d_model, d_ffn, n_kernels)
                for _ in range(n_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, X) -> torch.Tensor:

        for i in range(self.n_layers):
            enc_out = self.layer_norm(self.model[i](X))

        return enc_out


def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class InceptionBlockV1(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i)
            )
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res


class TimesBlock(nn.Module):
    def __init__(self, seq_len, pred_len, top_k, d_model, d_ffn, num_kernels):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.top_k = top_k

        # parameter-efficient design
        self.conv = nn.Sequential(
            InceptionBlockV1(d_model, d_ffn, num_kernels=num_kernels),
            nn.GELU(),
            InceptionBlockV1(d_ffn, d_model, num_kernels=num_kernels),
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.top_k)

        res = []
        for i in range(self.top_k):
            period = period_list[i]
            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros(
                    [x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]
                ).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = self.seq_len + self.pred_len
                out = x
            # reshape
            out = (
                out.reshape(B, length // period, period, N)
                .permute(0, 3, 1, 2)
                .contiguous()
            )
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, : (self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res
