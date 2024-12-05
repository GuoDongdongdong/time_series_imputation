"""
The core wrapper assembles the submodules of GP-VAE imputation model
and takes over the forward progress of the algorithm.

"""

# Created by Jun Wang <jwangfx@connect.ust.hk> and Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """model GPVAE with Gaussian Process prior

    Parameters
    ----------
    input_dim : int,
        the feature dimension of the input

    time_length : int,
        the length of each time series

    latent_dim : int,
        the feature dimension of the latent embedding

    encoder_sizes : tuple,
        the tuple of the network size in encoder

    decoder_sizes : tuple,
        the tuple of the network size in decoder

    beta : float,
        the weight of the KL divergence

    M : int,
        the number of Monte Carlo samples for ELBO estimation

    K : int,
        the number of importance weights for IWAE model

    kernel : str,
        the Gaussian Process kernel ["cauchy", "diffusion", "rbf", "matern"]

    sigma : float,
        the scale parameter for a kernel function

    length_scale : float,
        the length scale parameter for a kernel function

    kernel_scales : int,
        the number of different length scales over latent space dimensions
    """

    def __init__(self, args):
        super().__init__()
        self.input_dim = args.features
        self.time_length = args.seq_len
        self.latent_dim = args.GPVAE_latent_dim
        self.encoder_sizes = args.GPVAE_encoder_sizes
        self.decoder_sizes = args.GPVAE_decoder_sizes
        self.beta = args.GPVAE_beta
        self.M = args.GPVAE_M
        self.K = args.GPVAE_K
        self.kernel = args.GPVAE_kernel
        self.sigma = args.GPVAE_sigma
        self.length_scale = args.GPVAE_length_scale
        self.kernel_scales = args.GPVAE_kernel_scales
        self.window_size = args.GPVAE_window_size
        self.backbone = BackboneGPVAE(
            self.input_dim,
            self.time_length,
            self.latent_dim,
            self.encoder_sizes,
            self.decoder_sizes,
            self.beta,
            self.M,
            self.K,
            self.kernel,
            self.sigma,
            self.length_scale,
            self.kernel_scales,
            self.window_size,
        )

    def evaluate(self, batch:dict, training=True) -> torch.Tensor:
        res = self.forward(batch, training)
        return res['loss']
    
    def impute(self, batch:dict, n_samples:int=None) -> torch.Tensor:
        res = self.forward(batch, True, n_samples)
        return res['imputed_data']
    
    def forward(self, inputs, training=True, n_sampling_times=1):
        X, missing_mask = inputs["X"], inputs["missing_mask"]
        results = {}

        elbo_loss = self.backbone(X, missing_mask)
        results["loss"] = elbo_loss
        imputed_data = self.backbone.impute(X, missing_mask, n_sampling_times)
        results["imputed_data"] = imputed_data

        return results


class BackboneGPVAE(nn.Module):
    """model GPVAE with Gaussian Process prior

    Parameters
    ----------
    input_dim : int,
        the feature dimension of the input

    time_length : int,
        the length of each time series

    latent_dim : int,
        the feature dimension of the latent embedding

    encoder_sizes : tuple,
        the tuple of the network size in encoder

    decoder_sizes : tuple,
        the tuple of the network size in decoder

    beta : float,
        the weight of the KL divergence

    M : int,
        the number of Monte Carlo samples for ELBO estimation

    K : int,
        the number of importance weights for IWAE model

    kernel : str,
        the Gaussian Process kernel ["cauchy", "diffusion", "rbf", "matern"]

    sigma : float,
        the scale parameter for a kernel function

    length_scale : float,
        the length scale parameter for a kernel function

    kernel_scales : int,
        the number of different length scales over latent space dimensions
    """

    def __init__(
        self,
        input_dim,
        time_length,
        latent_dim,
        encoder_sizes=(64, 64),
        decoder_sizes=(64, 64),
        beta=1,
        M=1,
        K=1,
        kernel="cauchy",
        sigma=1.0,
        length_scale=7.0,
        kernel_scales=1,
        window_size=24,
    ):
        super().__init__()
        self.kernel = kernel
        self.sigma = sigma
        self.length_scale = length_scale
        self.kernel_scales = kernel_scales

        self.input_dim = input_dim
        self.time_length = time_length
        self.latent_dim = latent_dim
        self.beta = beta
        self.encoder = GpvaeEncoder(input_dim, latent_dim, encoder_sizes, window_size)
        self.decoder = GpvaeDecoder(latent_dim, input_dim, decoder_sizes)
        self.M = M
        self.K = K

        self.prior = None

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        if not torch.is_tensor(z):
            z = torch.tensor(z).float()
        num_dim = len(z.shape)
        assert num_dim > 2
        return self.decoder(torch.transpose(z, num_dim - 1, num_dim - 2))

    @staticmethod
    def kl_divergence(a, b):
        return torch.distributions.kl.kl_divergence(a, b)

    def _init_prior(self, device="cpu"):
        # Compute kernel matrices for each latent dimension
        kernel_matrices = []
        for i in range(self.kernel_scales):
            if self.kernel == "rbf":
                kernel_matrices.append(
                    rbf_kernel(self.time_length, self.length_scale / 2**i)
                )
            elif self.kernel == "diffusion":
                kernel_matrices.append(
                    diffusion_kernel(self.time_length, self.length_scale / 2**i)
                )
            elif self.kernel == "matern":
                kernel_matrices.append(
                    matern_kernel(self.time_length, self.length_scale / 2**i)
                )
            elif self.kernel == "cauchy":
                kernel_matrices.append(
                    cauchy_kernel(
                        self.time_length, self.sigma, self.length_scale / 2**i
                    )
                )

        # Combine kernel matrices for each latent dimension
        tiled_matrices = []
        total = 0
        for i in range(self.kernel_scales):
            if i == self.kernel_scales - 1:
                multiplier = self.latent_dim - total
            else:
                multiplier = int(np.ceil(self.latent_dim / self.kernel_scales))
                total += multiplier
            tiled_matrices.append(
                torch.unsqueeze(kernel_matrices[i], 0).repeat(multiplier, 1, 1)
            )
        kernel_matrix_tiled = torch.cat(tiled_matrices)
        assert len(kernel_matrix_tiled) == self.latent_dim
        prior = torch.distributions.MultivariateNormal(
            loc=torch.zeros(self.latent_dim, self.time_length, device=device),
            covariance_matrix=kernel_matrix_tiled.to(device),
        )
        return prior

    def impute(self, X, missing_mask, n_sampling_times=1):
        n_samples, n_steps, n_features = X.shape
        X = X.repeat(n_sampling_times, 1, 1)
        missing_mask = missing_mask.repeat(n_sampling_times, 1, 1).type(torch.bool)
        decode_x_mean = self.decode(self.encode(X).mean).mean
        imputed_data = decode_x_mean * ~missing_mask + X * missing_mask
        imputed_data = imputed_data.reshape(
            n_sampling_times, n_samples, n_steps, n_features
        ).permute(1, 0, 2, 3)
        return imputed_data

    def forward(self, X, missing_mask):
        X = X.repeat(self.K * self.M, 1, 1)
        missing_mask = missing_mask.repeat(self.K * self.M, 1, 1).type(torch.bool)

        if self.prior is None:
            self.prior = self._init_prior(device=X.device)

        qz_x = self.encode(X)
        z = qz_x.rsample()
        px_z = self.decode(z)
        nll = -px_z.log_prob(X)
        nll = torch.where(torch.isfinite(nll), nll, torch.zeros_like(nll))
        if missing_mask is not None:
            nll = torch.where(missing_mask, nll, torch.zeros_like(nll))
        nll = nll.sum(dim=(1, 2))

        if self.K > 1:
            kl = qz_x.log_prob(z) - self.prior.log_prob(z)
            kl = torch.where(torch.isfinite(kl), kl, torch.zeros_like(kl))
            kl = kl.sum(1)

            weights = -nll - kl
            weights = torch.reshape(weights, [self.M, self.K, -1])

            elbo = torch.logsumexp(weights, dim=1)
            elbo = elbo.mean()
        else:
            kl = self.kl_divergence(qz_x, self.prior)
            kl = torch.where(torch.isfinite(kl), kl, torch.zeros_like(kl))
            kl = kl.sum(1)

            elbo = -nll - self.beta * kl
            elbo = elbo.mean()

        return -elbo


class GpvaeEncoder(nn.Module):
    def __init__(self, input_size, z_size, hidden_sizes=(128, 128), window_size=24):
        """This module is an encoder with 1d-convolutional network and multivariate Normal posterior used by GP-VAE with
        proposed banded covariance matrix

        Parameters
        ----------
        input_size : int,
            the feature dimension of the input

        z_size : int,
            the feature dimension of the output latent embedding

        hidden_sizes : tuple,
            the tuple of the hidden layer sizes, and the tuple length sets the number of hidden layers

        window_size : int
            the kernel size for the Conv1D layer
        """
        super().__init__()
        self.z_size = int(z_size)
        self.input_size = input_size
        self.net, self.mu_layer, self.logvar_layer = make_cnn(
            input_size, (z_size, z_size * 2), hidden_sizes, window_size
        )

    def forward(self, x):
        mapped = self.net(x)
        batch_size = mapped.size(0)
        time_length = mapped.size(1)

        num_dim = len(mapped.shape)
        mu = self.mu_layer(mapped)
        logvar = self.logvar_layer(mapped)
        mapped_mean = torch.transpose(mu, num_dim - 1, num_dim - 2)
        mapped_covar = torch.transpose(logvar, num_dim - 1, num_dim - 2)
        mapped_covar = torch.sigmoid(mapped_covar)
        mapped_reshaped = mapped_covar.reshape(batch_size, self.z_size, 2 * time_length)

        dense_shape = [batch_size, self.z_size, time_length, time_length]
        idxs_1 = np.repeat(np.arange(batch_size), self.z_size * (2 * time_length - 1))
        idxs_2 = np.tile(
            np.repeat(np.arange(self.z_size), (2 * time_length - 1)), batch_size
        )
        idxs_3 = np.tile(
            np.concatenate([np.arange(time_length), np.arange(time_length - 1)]),
            batch_size * self.z_size,
        )
        idxs_4 = np.tile(
            np.concatenate([np.arange(time_length), np.arange(1, time_length)]),
            batch_size * self.z_size,
        )
        idxs_all = np.stack([idxs_1, idxs_2, idxs_3, idxs_4], axis=1)

        mapped_values = mapped_reshaped[:, :, :-1].reshape(-1)
        prec_sparse = torch.sparse_coo_tensor(
            torch.LongTensor(idxs_all).t().to(mapped.device),
            (mapped_values).to(mapped.device),
            (dense_shape),
        )
        prec_sparse = prec_sparse.coalesce()
        prec_tril = prec_sparse.to_dense()
        eye = (
            torch.eye(prec_tril.shape[-1])
            .unsqueeze(0)
            .repeat(prec_tril.shape[0], prec_tril.shape[1], 1, 1)
            .to(mapped.device)
        )
        prec_tril = prec_tril + eye
        cov_tril = torch.linalg.solve_triangular(prec_tril, eye, upper=True)
        cov_tril = torch.where(
            torch.isfinite(cov_tril), cov_tril, torch.zeros_like(cov_tril)
        ).to(mapped.device)

        num_dim = len(cov_tril.shape)
        cov_tril_lower = torch.transpose(cov_tril, num_dim - 1, num_dim - 2)

        z_dist = torch.distributions.MultivariateNormal(
            loc=mapped_mean, scale_tril=cov_tril_lower
        )
        return z_dist


class GpvaeDecoder(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=(256, 256)):
        """This module is a decoder with Gaussian output distribution.

        Parameters
        ----------
        output_size : int,
            the feature dimension of the output

        hidden_sizes: tuple
            the tuple of hidden layer sizes, and the tuple length sets the number of hidden layers.
        """
        super().__init__()
        self.net = make_nn(input_size, output_size, hidden_sizes)

    def forward(self, x):
        mu = self.net(x)
        var = torch.ones_like(mu)
        return torch.distributions.Normal(mu, var)


def rbf_kernel(T, length_scale):
    xs = torch.arange(T).float()
    xs_in = torch.unsqueeze(xs, 0)
    xs_out = torch.unsqueeze(xs, 1)
    distance_matrix = (xs_in - xs_out) ** 2
    distance_matrix_scaled = distance_matrix / length_scale**2
    kernel_matrix = torch.exp(-distance_matrix_scaled)
    return kernel_matrix


def diffusion_kernel(T, length_scale):
    assert length_scale < 0.5, (
        "length_scale has to be smaller than 0.5 for the "
        "kernel matrix to be diagonally dominant"
    )
    sigmas = torch.ones(T, T) * length_scale
    sigmas_tridiag = torch.diagonal(sigmas, offset=0, dim1=-2, dim2=-1)
    sigmas_tridiag += torch.diagonal(sigmas, offset=1, dim1=-2, dim2=-1)
    sigmas_tridiag += torch.diagonal(sigmas, offset=-1, dim1=-2, dim2=-1)
    kernel_matrix = sigmas_tridiag + torch.eye(T) * (1.0 - length_scale)
    return kernel_matrix


def matern_kernel(T, length_scale):
    xs = torch.arange(T).float()
    xs_in = torch.unsqueeze(xs, 0)
    xs_out = torch.unsqueeze(xs, 1)
    distance_matrix = torch.abs(xs_in - xs_out)
    distance_matrix_scaled = distance_matrix / torch.sqrt(length_scale).type(
        torch.float32
    )
    kernel_matrix = torch.exp(-distance_matrix_scaled)
    return kernel_matrix


def cauchy_kernel(T, sigma, length_scale):
    xs = torch.arange(T).float()
    xs_in = torch.unsqueeze(xs, 0)
    xs_out = torch.unsqueeze(xs, 1)
    distance_matrix = (xs_in - xs_out) ** 2
    distance_matrix_scaled = distance_matrix / length_scale**2
    kernel_matrix = sigma / (distance_matrix_scaled + 1.0)

    alpha = 0.001
    eye = torch.eye(kernel_matrix.shape[-1])
    return kernel_matrix + alpha * eye


def make_nn(input_size, output_size, hidden_sizes):
    """This function used to creates fully connected neural network.

    Parameters
    ----------
    input_size : int,
        the dimension of input embeddings

    output_size : int,
        the dimension of out embeddings

    hidden_sizes : tuple,
        the tuple of hidden layer sizes, and the tuple length sets the number of hidden layers

    Returns
    -------
    output: tensor
        the processing embeddings
    """
    layers = []
    for i in range(len(hidden_sizes)):
        if i == 0:
            layers.append(
                nn.Linear(in_features=input_size, out_features=hidden_sizes[i])
            )
        else:
            layers.append(
                nn.Linear(in_features=hidden_sizes[i - 1], out_features=hidden_sizes[i])
            )
        layers.append(nn.ReLU())
    layers.append(nn.Linear(in_features=hidden_sizes[-1], out_features=output_size))
    return nn.Sequential(*layers)


def make_cnn(input_size, output_size, hidden_sizes, kernel_size=3):
    """This function used to construct neural network consisting of
       one 1d-convolutional layer that utilizes temporal dependencies,
       fully connected network

    Parameters
    ----------
    input_size : int,
        the dimension of input embeddings

    output_size : int,
        the dimension of out embeddings

    hidden_sizes : tuple,
        the tuple of hidden layer sizes, and the tuple length sets the number of hidden layers,

    kernel_size : int
        kernel size for convolutional layer

    Returns
    -------
    output: tensor
        the processing embeddings
    """
    padding = kernel_size // 2

    cnn_layer = CustomConv1d(
        input_size, hidden_sizes[0], kernel_size=kernel_size, padding=padding
    )
    layers = [cnn_layer]

    for i, h in zip(hidden_sizes, hidden_sizes[1:]):
        layers.extend([nn.Linear(i, h), nn.ReLU()])
    if isinstance(output_size, tuple):
        net = nn.Sequential(*layers)
        return [net] + [nn.Linear(hidden_sizes[-1], o) for o in output_size]

    layers.append(nn.Linear(hidden_sizes[-1], output_size))
    return nn.Sequential(*layers)


class CustomConv1d(torch.nn.Conv1d):
    def __init(self, in_channels, out_channels, kernel_size, padding):
        super().__init__(in_channels, out_channels, kernel_size, padding)

    def forward(self, x):
        if len(x.shape) > 2:
            shape = list(np.arange(len(x.shape)))
            new_shape = [0, shape[-1]] + shape[1:-1]
            out = super().forward(x.permute(*new_shape))
            shape = list(np.arange(len(out.shape)))
            new_shape = [0, shape[-1]] + shape[1:-1]
            if self.kernel_size[0] % 2 == 0:
                out = F.pad(out, (0, -1), "constant", 0)
            return out.permute(new_shape)

        return super().forward(x)

