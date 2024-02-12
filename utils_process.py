#%% Imports
import torch
import numpy as np
from torch import matmul
from typing import Union, List
import matplotlib.pyplot as plt

import kernels
from kernels import RBFKernel
from utils import diagonalize
from recognition_parametrised_model import RPM
from flexible_multivariate_normal import FlexibleMultivariateNormal


def plot_loss(
        rpm: RPM,
        offset: int =0,
        **kwargs
):

    plt.figure()
    plt.plot(rpm.loss_tot[offset:], c='k', lw=2, **kwargs)
    plt.ylabel('loss')
    plt.xlabel('Iterations')
    plt.title('- Free Energy')
    plt.tight_layout()


def plot_rpgpfa_summary(
    rpm: RPM,
    plot_id_factors: Union[str, List] = 'all',
    plot_id_observations: Union[str, List] = 'all',
    plot_variational: bool = True,
    plot_regressed: bool = False,
    plot_variance: bool = True,
    plot_type: str = 'linear',
    plot_true: bool = False,
    latent_true: torch.Tensor = None,
    regress_param: dict = None,
):

    # Factors to be plotted
    plot_id_factors = range(rpm.num_factors) if plot_id_factors == 'all' else plot_id_factors

    # Observations to be plotted
    plot_id_observations = range(rpm.num_observation) if plot_id_observations == 'all' else plot_id_observations

    # Name Variational
    name_q = 'E[q](Z)'

    # Name Factors
    name_f = ['E[f' + str(fc) + '](Z)' for fc in plot_id_factors]

    # Sufficient Statistics Variational
    qmean, qcova = [
        xx.detach().cpu()
        for xx in rpm.dist_marginals.mean_covariance()
    ]

    # Sufficient Statistics Recognition Factors
    fmean, fcova = [
        xx.detach().cpu()
        for xx in rpm.dist_factors.mean_covariance()
    ]

    # Keep Only some factors
    fmean, fcova = [
        [
            xx[fc] for fc in plot_id_factors
        ] for xx in [fmean, fcova]
    ]

    # Gather names
    names = [name_q] if plot_variational else []
    names = [*names, *name_f]

    # Gather means
    means = [qmean] if plot_variational else []
    means = [*means, *fmean]

    # Gather covariances
    covas = [qcova] if plot_variational else []
    covas = [*covas, *fcova]

    # Regress to True Mean if Provided
    for fc in range(len(means)):
        if not latent_true is None and plot_regressed:

            # Current mean and covariance
            latent_fit = means[fc]
            latent_var = covas[fc]

            # Regress
            means[fc], covas[fc], latent_true, _, _ = regress(
                latent_fit,
                latent_var,
                latent_true,
                regression='linear',
                regression_param=regress_param,
            )

    # Filter out specified observations
    means, covas = [
        [yy[plot_id_observations] for yy in xx] for xx in [means, covas]
    ]

    dims = means[0].shape[-1]
    if plot_type == 'linear':

        # Plot each dimensions across time
        plt.figure()
        for mm in range(len(means)):
            for dim in range(dims):
                plt.subplot(dims, len(means), mm + dim * len(means) + 1)

                # E[Z]
                xx = means[mm][..., dim].reshape(-1)

                # V[Z]
                yy = covas[mm][..., dim, dim].reshape(-1)

                # Upper and lower bounds
                up = xx + 2 * np.sqrt(yy)
                lo = xx - 2 * np.sqrt(yy)

                tt = np.linspace(0, 1, xx.shape[0])

                # Plot MAP mean
                plt.plot(tt, xx, c='k', label='MAP')

                # Plot CI
                if plot_variance:
                    plt.fill_between(tt, lo, up, color='k', alpha=.25)

                if latent_true is not None and plot_true:
                    zz = latent_true[plot_id_observations][..., dim].reshape(-1)
                    plt.plot(tt, zz, color='b', label='True')

                if dim == 0:
                    plt.title(names[mm])
                    if mm ==0:
                        plt.legend()

    elif plot_type == '3D':
        assert dims == 3, 'Error: 3D plot only supported when latent is 3D'

        # 3D Plot
        for mm in range(len(means)):
            ax = plt.figure().add_subplot(projection='3d')
            for nn in range(means[mm].shape[0]):

                # E[Z]
                xx = means[mm][nn].numpy()

                ax.plot(
                    xx[:, 0],
                    xx[:, 1],
                    xx[:, 2],
                    lw=0.5, color='k'
                )

                ax.set_xlabel("Z[1]")
                ax.set_ylabel("Z[2]")
                ax.set_zlabel("Z[3]")
                ax.set_title(names[mm])

        if latent_true is not None and plot_true:
            ax = plt.figure().add_subplot(projection='3d')
            for nn in range(means[mm].shape[0]):
                # E[Z]
                xx = latent_true[nn].numpy()

                ax.plot(
                    xx[:, 0],
                    xx[:, 1],
                    xx[:, 2],
                    lw=0.5, color='k'
                )

                ax.set_xlabel("Z[1]")
                ax.set_ylabel("Z[2]")
                ax.set_zlabel("Z[3]")
                ax.set_title("True Latent")

    else:
        raise NotImplementedError()




def regress(
    latent_fit: torch.Tensor,
    latent_var: torch.Tensor,
    latent_true: torch.Tensor,
    regression: str = 'krr',
    regression_param: dict = None,
):
    """ Use linear or kernel regression to regress the fit latent to the true latent when provided """

    shape_fit = latent_fit.shape
    shape_true = latent_true.shape

    unfolded_fit = [np.prod(shape_fit[:-1]).astype(int), shape_fit[-1]]
    unfolded_true = [np.prod(shape_true[:-1]).astype(int), shape_fit[-1]]

    # Unfold and 0 center Latents
    latent_true = latent_true.reshape(unfolded_true)
    latent_true = latent_true - latent_true.mean(dim=0)
    latent_fit = latent_fit.reshape(unfolded_fit)
    latent_fit = latent_fit - latent_fit.mean(dim=0)

    # Regress Latent - True Latent
    if regression == 'linear':
        latent_fit, latent_jac, regressor, jacobian = \
            regress_linear(latent_fit, latent_true, regress_param=regression_param)
    elif regression == 'krr':
        latent_fit, latent_jac, regressor, jacobian = \
            regress_krr(latent_fit, latent_true, regress_param=regression_param)
    else:
        raise NotImplementedError()

    # True new variance or linear approximation with Jacobian
    latent_var = latent_var.reshape(*unfolded_fit, latent_var.shape[-1])
    latent_var = matmul(matmul(latent_jac.transpose(-1, -2), latent_var), latent_jac)
    latent_var = latent_var.reshape(*shape_true, latent_fit.shape[-1])

    # Reshape True and Regressed latent
    latent_fit = latent_fit.reshape(shape_true)
    latent_true = latent_true.reshape(shape_true)

    return latent_fit, latent_var, latent_true, regressor, jacobian


def sample_XYtrain(X, Y, train_pct):
    len_input = X.shape[0]
    len_train = int(len_input * train_pct)
    idx_train = np.random.choice(len_input, len_train, replace=False)
    Xtrain = X[idx_train, :]
    Ytrain = Y[idx_train, :]

    return Xtrain, Ytrain


def regress_linear(X, Y, regress_param=None):

    if regress_param is None:
        regress_param = {}
        
    if not ('train_pct' in regress_param.keys()):
        train_pct = 0.8
    else:
        train_pct = regress_param['train_pct']

    if not ('alpha' in regress_param.keys()):
        alpha = 1e-6
    else:
        alpha = regress_param['alpha']

    Xtrain, Ytrain = sample_XYtrain(X, Y, train_pct)
    XXinv = torch.linalg.inv(alpha * torch.eye(Xtrain.shape[-1], device=Xtrain.device, dtype=Xtrain.dtype) + torch.matmul(Xtrain.transpose(-1, -2), Xtrain))
    beta_hat = matmul(XXinv, matmul(Xtrain.transpose(-1, -2), Ytrain))

    def regressor(X0):
        return matmul(X0, beta_hat)

    def jacobian(X0):
        return beta_hat.unsqueeze(0)

    Yhat = regressor(X)
    Jhat = jacobian(X)

    return Yhat, Jhat, regressor, jacobian


def regress_krr(X, Y, regress_param=None):

    # Default params
    if regress_param is None:
        regress_param = {}

    if 'train_pct' not in regress_param.keys():
        train_pct = 0.8
    else:
        train_pct = regress_param['train_pct']

    if 'alpha' not in regress_param.keys():
        alpha = 1e-3
    else:
        alpha = regress_param['alpha']

    if 'kernel_param' not in regress_param.keys():
        o1 = torch.ones(1, device=X.device, dtype=X.dtype)
        kernel_param = {'type': 'RBF', 'param': {'scale': o1, 'lengthscale': 2 * o1}}
    else:
        kernel_param = regress_param['kernel_param']

    # Init kernel
    if kernel_param['type'] == 'RBF':
        kernel = kernels.RBFKernel(**kernel_param['param'])
    if kernel_param['type'] == 'RQ':
        kernel = kernels.RQKernel(**kernel_param['param'])
    if kernel_param['type'] == 'POLY':
        kernel = kernels.POLYKernel(**kernel_param['param'])



    Xtrain, Ytrain = sample_XYtrain(X, Y, train_pct)
    KXtrainXtrain = kernel.forward(Xtrain, Xtrain).squeeze(0)
    INN = torch.eye(KXtrainXtrain.shape[0],device=KXtrainXtrain.device, dtype=KXtrainXtrain.dtype)
    beta_hat = matmul(torch.linalg.inv(KXtrainXtrain + alpha * INN), Ytrain)

    # Linear approximation to the new mean
    def regressor(X0):
        KxXtrain = kernel.forward(X0, Xtrain).squeeze(0)
        return matmul(KxXtrain, beta_hat)

    Yhat = regressor(X)

    if kernel_param['type'] == 'RBF':
        # Linear approximation to the variance
        def jacobian(X0):
            KxXtrain = kernel.forward(X0, Xtrain).squeeze(0).unsqueeze(-1)
            betaK = beta_hat.unsqueeze(0) * KxXtrain
            dX = Xtrain.unsqueeze(0) - X0.unsqueeze(1)
            return (2 * matmul(betaK.transpose(-1, -2), dX) / (kernel.lengthscale**2)).transpose(-1, -2)


        Jhat = jacobian(X)

    else:
        Jhat = None
        jacobian = None

    return Yhat, Jhat, regressor, jacobian




def plot_gradient_line(xx, cc, **kwargs):
    for tt in range(xx.shape[0]-1):
        plt.plot(xx[tt:tt+2, 0], xx[tt:tt+2,  1], c=cc[tt], **kwargs)


def max_diagonal_factors(factors):
    """
        Custom Varimax Criterion: Maximize the diagonal elements of the factors
    """

    # Factors dimension
    D = factors.shape[-1]

    # Fit parameters parameters
    ite_max = 1000
    reltol = 1e-6

    # Initialize rotation
    rotation = torch.eye(D, dtype=factors.dtype)

    # Iterate
    for ite_cur in range(ite_max):
        max_theta = torch.tensor(0)

        # Loop over all pairs of dimensions
        for ii in np.arange(0, D-1):
            for jj in range(ii+1, D):

                Vuu = factors[:, ii, ii]
                Vvv = factors[:, jj, jj]
                Vuv = factors[:, ii, jj]

                numer = torch.sum(4 * Vuv * (Vuu - Vvv), dim=0)
                denum = torch.sum(4 * Vuv ** 2 - (Vuu - Vvv) ** 2, dim=0)
                theta = torch.atan2(numer, denum) / 4 + 1 * torch.pi / 4

                max_theta = torch.max(max_theta, torch.abs(theta))

                rotation_cur = torch.eye(D, dtype=factors.dtype)
                rotation_cur[ii, ii] = torch.cos(theta)
                rotation_cur[ii, jj] = -torch.sin(theta)
                rotation_cur[jj, ii] = torch.sin(theta)
                rotation_cur[jj, jj] = torch.cos(theta)

                rotation = matmul(rotation_cur, rotation)
                factors = matmul(matmul(rotation_cur, factors), rotation_cur.transpose(-1, -2))

        if max_theta < reltol:
            print('theta converged ite=' + str(ite_cur))
            break

    return factors, rotation


def get_precision_gain(model):
    """Precision Gain from prior to Recognition Factors"""

    # Get Factor Posterior Means
    factors_mean = model.factors.suff_stat_mean[0]

    # Mean Center Factors across Time
    factors_mean = factors_mean - factors_mean.mean(dim=-2, keepdim=True)

    # Normalize factors to have max absolute value of 1
    factors_normalizer = diagonalize(factors_mean.abs().max(dim=-2, keepdim=True)[0].max(dim=-3, keepdim=True)[0])
    factors_mean = matmul(torch.linalg.inv(factors_normalizer), factors_mean.unsqueeze(-1)).squeeze(-1)

    # Prior Precision
    natural2_prior = diagonalize(model._get_prior_marginals()[1].permute((1, 0)))[0].unsqueeze(0)

    # Factors Precision (if not time dependent, then only one value)
    factors_precision = model.factors.natural2[:, 0, 0].unsqueeze(1).unsqueeze(1)

    # Precision gain (normalise)
    precision_gain = - factors_precision + natural2_prior
    precision_gain = matmul(matmul(factors_normalizer, precision_gain), factors_normalizer)
    precision_gain = precision_gain.squeeze(1).squeeze(1)

    return precision_gain, factors_mean


def trim_dimensions(precision, threshold=0.95, verbose=True):
    """ Remove Latent dimensions with no or few precision gain across all factors """

    # Look at the contribution of each latent dimension to the precision gain
    contrib = precision.diagonal(dim1=-1, dim2=-2)
    contrib = (contrib / contrib.sum(dim=-1, keepdim=True)).mean(dim=0)
    contrib, contrib_indices = contrib.sort(descending=True)
    contrib = contrib.cumsum(dim=-1)

    # Remove dimensions above threshold
    above_thr = contrib >= threshold
    thr_index = torch.argmax((above_thr != 0).to(dtype=torch.int), dim=-1)
    kept_dimensions = contrib_indices[:thr_index + 1].sort()[0]

    if above_thr.sum() > 1 and verbose:
        print('Removed ' + str(above_thr.sum().item() - 1) + ' dimensions')

    # Trim precision gain
    precision_gain_trimmed = precision[..., kept_dimensions][..., kept_dimensions, :]

    return precision_gain_trimmed, kept_dimensions, contrib


def rotate_and_trim_precision(precision, rotate=True, normalize=False, threshold=0.95, verbose=True):
    """ Trim and rotate precision gain to maximize diagonal criterion """

    if rotate:
        # Rotate precision gain using diagonal criterion
        precision_rotated, rotation = max_diagonal_factors(precision)
        rotation = rotation.transpose(-1, -2)
    else:
        precision_rotated = precision
        rotation = torch.eye(precision.shape[-1], dtype=precision.dtype)

    # Trim dimensions
    precision_rotated_trimmed, kept_latent, contrib = \
        trim_dimensions(precision_rotated, threshold=threshold, verbose=verbose)

    # Incorporate Trimming in rotation
    Id = torch.eye(precision_rotated.shape[-1], dtype=precision_rotated.dtype)
    Id = Id[:, kept_latent]
    rotation = matmul(rotation, Id)

    # Normalize over factors
    if normalize:
        precision_rotated_trimmed = precision_rotated_trimmed / precision_rotated_trimmed.sum(dim=0, keepdim=True)

    return precision_rotated_trimmed, rotation, kept_latent, contrib
