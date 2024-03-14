import torch
from torch import matmul, inference_mode

import numpy as np

import flexible_multivariate_normal
from flexible_multivariate_normal import (
    FlexibleMultivariateNormal,
    vector_to_tril,
    kl,
    flexible_kl,
    get_log_normalizer)

from typing import Union, List, Dict
from utils import print_loss, get_minibatches

from prior import GPPrior
import recognition

import fast_initializations
import _updates

from utils import diagonalize

# TODO: Freeze auxiliary
# TODO: rotate to diagonalize covariances
# TODO: Check Amortized RPGPFA
# TODO: remove T !!
# MAKE SURE PARAMETERS ARE BEING PTIMIZED !
# TODO: Modif description


class RPM(fast_initializations.Mixin, _updates.Mixin):
    """
    Recognition Parametrised Model (RPM)

    Summary:
        Flexible Class for Recognition Parametrised Factor Analysis (RPFA).
        and Recognition Parametrised Gaussian Process Factor Analysis (RPGPFA).

    Args:
        - observations (torch.Tensor or List[torch.Tensor]): Multimodal (possibly time series) observations.
            Sizes:
                len(observations) = num_factors
                observations[j] ~ num_observations x len_observations x *dim_j
            Where:
                num_factors: number of conditionally independent factors
                num_observations: number of observation samples
                len_observations: length of the time series
                dim_j: dimension of j=th observations
        - observation_locations (torch.Tensor): Locations (e.g. time) at which observations are recorded
            Sizes:
                len_observations x dim_observation_locations
        - inducing_locations (torch.Tensor): Locations (e.g. time) at of inducing points
        - loss_tot (List[float]): Stored Loss defined as - Free Energy
        - fit_params (Dict): Fit Parameters. See _initializations.py for Details
        - prior (GPPrior): Latent Prior Distribution
        - recognition_factors ()
        - recognition_factors (recognition.Encoder)
        - recognition_auxiliary (recognition.Encoder)
        - recognition_variational (recognition.Encoder)

    notation: for compactness, we sometimes denote:
        N: num_observations
        T: len_observations
        K: dim_latent
        M: num_inducing_points (<= T)


    """

    def __init__(
            self,
            observations: Union[torch.Tensor, List[torch.tensor]],
            loss_tot: List = None,
            fit_params: Dict = None,
            precision_chol_vec_factors: torch.Tensor = None,
            precision_chol_vec_auxiliary: torch.Tensor = None,
            recognition_factors: recognition.Encoder = None,
            recognition_auxiliary: recognition.Encoder = None,

    ):

        # Transform Observation in list if necessary
        observations = [observations]\
            if not (type(observations) is list) and not (type(observations) is tuple) else observations

        # Problem dimensions
        self.num_factors = len(observations)
        self.num_observation = observations[0].shape[0]
        self.len_observation = observations[0].shape[1]

        # Device and data type
        self.dtype = observations[0].dtype
        self.device = observations[0].device
        str_device = 'GPU' if torch.cuda.is_available() else 'CPU'
        print('RPM on ' + str_device + ' Observations on ' + str(self.device))

        # Fit / Config parameters
        self.fit_params = fit_params

        # Init Loss
        self.loss_tot = [] if loss_tot is None else loss_tot

        # Initialize Distributions Parametrization
        self.precision_chol_vec_factors = precision_chol_vec_factors
        self.precision_chol_vec_auxiliary = precision_chol_vec_auxiliary
        self.recognition_factors = recognition_factors
        self.recognition_auxiliary = recognition_auxiliary
        self._init_all(observations)
        
        # Sanity Checks
        assert all([i.shape[0] == self.num_observation for i in observations]), "Inconsistent number of observations"

        # Init Forwarded
        self.forwarded_factors = None
        self.forwarded_auxiliary = None

        # Init Distributions and parameters
        self.dist_factors = None
        self.dist_variational = None

        with torch.no_grad():
            batch = self.batches[0][0]
            batched_observations = [obsi[batch] for obsi in observations] \
                if len(batch) < self.num_observation else observations
            self._forward_all(batched_observations)
            self.loss_tot.append(self._get_loss().item())

    def _forward_all(self, observations):
        # TODO: When forwarding factors and auxiliary:
        # Get mean
        # Get precision chol. Build Precision. Be carefull with delta and not delta !!
        self._forward_factors(observations)
        self._forward_auxiliary(observations)


        # # #TODO: what follows is temporary and should be removed !
        # print('TMP')
        #
        # natural1_prior, natural2_prior = self.forwarded_prior
        # natural1_factor, natural2_factor = self.forwarded_factors
        # natural1_auxiliary, natural2_auxiliary = self.forwarded_auxiliary
        #
        # natural2_factor = natural2_factor.unsqueeze(1).repeat(1, natural1_factor.shape[1], 1, 1)
        # natural2_auxiliary = natural2_auxiliary.unsqueeze(1).repeat(1, natural1_auxiliary.shape[1], 1, 1)
        #
        # natural1_prior = natural1_prior.unsqueeze(0).unsqueeze(0)
        # natural2_prior = natural2_prior.unsqueeze(0).unsqueeze(0)
        #
        # natural1_variational = (natural1_prior + (natural1_factor - natural1_auxiliary).sum(dim=0)) / (self.num_factors + 1)
        # natural2_variational = (natural2_prior + (natural2_factor - natural2_auxiliary).sum(dim=0)) / (self.num_factors + 1)
        #
        # self.dist_variational = FlexibleMultivariateNormal(
        #     natural1_variational,
        #     natural2_variational,
        #     init_natural=True,
        #     init_cholesky=False,
        #     store_suff_stat_mean=True,
        #     jitter=0.0,
        # )
        #
        # self.dist_factors = FlexibleMultivariateNormal(
        #     natural1_factor,
        #     natural2_factor,
        #     init_natural=True,
        #     init_cholesky=False,
        #     store_suff_stat_mean=True,
        #     jitter=0.0,
        # )
        #
        # self.dist_auxiliary = FlexibleMultivariateNormal(
        #     natural1_auxiliary,
        #     natural2_auxiliary,
        #     init_natural=True,
        #     init_cholesky=False,
        #     store_suff_stat_mean=True,
        #     jitter=0.0,
        # )
        #
        # self.dist_prior = FlexibleMultivariateNormal(
        #     natural1_prior,
        #     natural2_prior,
        #     init_natural=True,
        #     init_cholesky=False,
        #     store_suff_stat_mean=True,
        #     jitter=0.0,
        # )
        #
        # self._update_delta_TMP()
        # self.KLMARGINAL = self._kl_marginals()
        # self.KLPRIOR =self._kl_prior()
        #
        # dist_prior_kl = FlexibleMultivariateNormal(
        #     natural1_prior.repeat(1, natural1_variational.shape[1], 1),
        #     natural2_prior.repeat(1, natural1_variational.shape[1], 1, 1),
        #     init_natural=True,
        #     init_cholesky=False,
        #     store_suff_stat_mean=True,
        #     jitter=0.0,
        # )
        #
        # dist_ratios_kl = FlexibleMultivariateNormal(
        #     natural1_factor - natural1_auxiliary,
        #     natural2_factor - natural2_auxiliary,
        #     init_natural=True,
        #     init_cholesky=False,
        #     store_suff_stat_mean=True,
        #     jitter=0.0,
        # )
        #
        # dist_variational_kl = FlexibleMultivariateNormal(
        #     natural1_variational.repeat(self.num_factors, 1, 1),
        #     natural2_variational.repeat(self.num_factors, 1, 1, 1),
        #     init_natural=True,
        #     init_cholesky=False,
        #     store_suff_stat_mean=True,
        #     jitter=0.0,
        # )
        #
        # prior_kl_alt = flexible_kl(self.dist_variational, dist_prior_kl).sum()
        # margi_kl_alt = flexible_kl(dist_variational_kl, dist_ratios_kl).sum()
        #
        # phi1 = self.dist_variational.log_normalizer.sum() * (1 + self.num_factors)
        # phi_prior = self.dist_prior.log_normalizer[0, 0]
        # phi_delta = self.dist_delta.log_normalizer.diagonal(dim1=-1, dim2=-2).sum(dim=0)
        # phi2 = (phi_prior + phi_delta).sum()
        # phi21 = phi_prior * self.num_observation
        # phi22 = phi_delta.sum()
        #
        # old = -(prior_kl_alt + margi_kl_alt)
        # new = (phi1 - phi2)
        #
        # self.phi1 = phi1
        # self.phi2 = phi2
        # self.phi21 = phi21
        # self.phi22 = phi22
        #
        # print(0)



    def _update_delta_TMP(self):
        """
        Build all the ration distributions (factors - factors_tilde)
        """

        # Natural Parameters of the factors ~ J x 1 x N x T x K (x K)
        factors_natural1 = self.dist_factors.natural1.unsqueeze(1)
        factors_natural2 = self.dist_factors.natural2.unsqueeze(1)
        factors_log_normaliser = self.dist_factors.log_normalizer.unsqueeze(1)

        # Pseudo Natural Parameters of the auxiliary factors ~ J x N x 1 x T x K (x K)
        factors_tilde_natural1 = self.dist_auxiliary.natural1.unsqueeze(2)
        factors_tilde_natural2 = self.dist_auxiliary.natural2.unsqueeze(2)

        # eta_m - eta_tilde_n ~ J x N x N x T x K (x K)
        delta_natural1 = factors_natural1 - factors_tilde_natural1
        delta_natural2 = factors_natural2 - factors_tilde_natural2

        # fhat in the paper
        self.dist_delta = FlexibleMultivariateNormal(
            delta_natural1,
            delta_natural2,
            init_natural=True,
            init_cholesky=False,
            store_suff_stat_mean=True
        )

        # Ratio of log-nomaliser differences ~ J x N x N x T
        delta_log_normalizer = self.dist_delta.log_normalizer - factors_log_normaliser

        # In the ergodic cae, the sum is over T and N
        if self.fit_params['ergodic']:
            log_weights = delta_log_normalizer - torch.logsumexp(delta_log_normalizer, dim=(2, 3), keepdim=True)
        else:
            log_weights = delta_log_normalizer - torch.logsumexp(delta_log_normalizer, dim=2, keepdim=True)

        # log Gamma ~ J x N x T
        self.log_gamma = log_weights.diagonal(dim1=1, dim2=2)






    def _forward_factors(self, observations):

        _, natural2_prior = self.forwarded_prior

        natural1_factors = torch.cat(
            [
                facti(obsi).unsqueeze(0)
                for facti, obsi in zip(self.recognition_factors, observations)
            ],
            axis=0
        )

        natural2_factors_tril = vector_to_tril(self.precision_chol_vec_factors)
        natural2_factors = natural2_prior - torch.matmul(natural2_factors_tril, natural2_factors_tril.transpose(-1, -2))

        self.forwarded_factors = [natural1_factors, natural2_factors]


    def _forward_auxiliary(self, observations):

        _, natural2_factors = self.forwarded_factors

        natural1_auxiliary = torch.cat(
            [facti(obsi).unsqueeze(0) for facti, obsi in zip(self.recognition_auxiliary, observations)],
            axis=0
        )

        natural2_auxiliary_tril = vector_to_tril(self.precision_chol_vec_auxiliary)
        naturalaj2 = natural2_factors + torch.matmul(natural2_auxiliary_tril, natural2_auxiliary_tril.transpose(-1, -2))

        self.forwarded_auxiliary = [natural1_auxiliary, naturalaj2]

    def get_posteriors(self, observations):

        with torch.no_grad():
            self._forward_all(observations)
            natural01, natural02 = self.forwarded_prior
            naturalj1, naturalj2 = self.forwarded_auxiliary
            naturalaj1, naturalaj2 = self.forwarded_auxiliary

            naturalq1 = (natural01 + (naturalj1 - naturalaj1).sum(0)) / (1 + self.num_factors)
            naturalq2 = (natural02 + (naturalj2 - naturalaj2).sum(0)) / (1 + self.num_factors)

            naturalq2 = naturalq2.unsqueeze(0).repeat(naturalq1.shape[0], 1, 1)
            naturalj2 = naturalj2.unsqueeze(1).repeat(1, naturalaj1.shape[1], 1, 1)

            distq = FlexibleMultivariateNormal(
                naturalq1,
                naturalq2,
                init_natural=True,
                init_cholesky=False,
                store_suff_stat_mean=True,
            )

            distf = FlexibleMultivariateNormal(
                naturalj1,
                naturalj2,
                init_natural=True,
                init_cholesky=False,
                store_suff_stat_mean=True,
            )

        return distq, distf

    def _get_loss(self):

        # Problem Dimensions
        dim_latent = self.dim_latent
        num_factors = self.num_factors
        num_observation = self.num_observation_batch
        normalizer = self.num_observation_batch * self.len_observation

        # Natural parameters
        _, natural2_prior = self.forwarded_prior
        natural1_factors, natural2_factors = self.forwarded_factors
        natural1_auxiliary, natural2_auxiliary = self.forwarded_auxiliary

        # Delta natural1 for each factor ~ J x N x K
        delta_natural1_jn = natural1_factors - natural1_auxiliary

        # ~ N x K
        delta_natural1_n = delta_natural1_jn.sum(0)

        # Delta natural2 for each factor ~ J x K x K
        delta_natural2_j = natural2_factors - natural2_auxiliary

        # Delta natural2 ~ K x K
        delta_natural2 = delta_natural2_j.sum(dim=0)

        # ~ K x K
        var_natural1 = matmul(delta_natural1_n.unsqueeze(-1), delta_natural1_n.unsqueeze(-2)).sum(dim=0)

        # ~ J x K x K
        var_natural1_j = matmul(delta_natural1_jn.unsqueeze(-1), delta_natural1_jn.unsqueeze(-2)).sum(dim=1)

        # Invert and Determinent ~ K x K and 1 x 1
        delta2_inv, delta2_logdet = _chol_inv_det(delta_natural2 + natural2_prior)

        # Invert and Determinent ~ J x K x K and J x 1 x 1
        delta2_inv_j, delta2_logdet_j = _chol_inv_det(delta_natural2_j)

        # ~ J x K x K
        nautral2_inv_j, _ = _chol_inv_det(natural2_factors)

        # Log Normalizer of the average
        constant1 = torch.tensor(
            0.5 * dim_latent * num_observation * (1 + num_factors) * np.log(np.pi * (1 + num_factors))
        )
        logdet1 = - 0.5 * num_observation * (1 + num_factors) * delta2_logdet
        trace1 = - (delta2_inv * var_natural1).sum() / 4
        phi1 = constant1 + logdet1 + trace1

        # Average Of the log normalizer
        constant2 = torch.tensor(
            0.5 * dim_latent * num_observation * (1 + num_factors) * np.log(np.pi)
        )
        logdet2 = - 0.5 * (1 + num_factors) * (
            num_observation * torch.log(torch.linalg.det(-natural2_prior)) + delta2_logdet_j.sum()
        )
        trace2 = - (delta2_inv_j * var_natural1_j).sum() / 4
        phi2 = constant2 + logdet2 + trace2



        # KL normalizer
        kl_normalizer = phi1 - phi2

        # print('Phi 1')
        # print('Classic      ' + str(self.phi1))
        # print('New approach ' + str(phi1))
        # print('Diff         ' + str( (phi1 - self.phi1) / self.phi1))
        # print('')
        #
        # print('Phi 2')
        # print('Classic      ' + str(self.phi2))
        # print('New approach ' + str(phi2))
        # print('Diff         ' + str( (phi2 - self.phi2) / self.phi2))
        # print('')
        #
        # print('Overall')
        # print('Classic ' + str(self.phi1 - self.phi2))
        # print('New approach ' + str(phi1 - phi2))
        # print('Diff         ' + str((self.phi1 - self.phi2) - (phi1 - phi2)))
        #



        # phi_prior = self.dist_prior.log_normalizer[0, 0]
        # constant = torch.tensor(
        #     0.5 * dim_latent * np.log(np.pi)
        # )
        # det = -0.5 * torch.log(torch.linalg.det(-self.dist_prior.natural2.squeeze()))
        # trace = 0
        # phi_prior_alt = constant + det + trace
        #




        # phi_delta = self.dist_delta.log_normalizer.diagonal(dim1=-1, dim2=-2)
        # constant = torch.tensor(
        #     0.5 * dim_latent * np.log(np.pi)
        # )
        # det = -0.5 * torch.log(torch.linalg.det(-self.dist_delta.natural2.diagonal(dim1=1, dim2=-2)))
        # var = matmul(self.dist_delta.natural1.diagonal(dim1=-1, dim2=-2).unsqueeze(-1), self.dist_delta.natural1.diagonal(dim1=-1, dim2=-2).unsqueeze(-2) )
        # trace = - (torch.linalg.inv(self.dist_delta.natural2.diagonal(dim1=1, dim2=-2)) * var).sum(dim=(-1, -2)) / 4
        # phi_delta_alt = constant + det + trace
        #
        # (phi_delta.su - phi_delta_alt).abs().max()
        #
        # print('problems here')
        # n1 = self.dist_delta.natural1[0, 0, 0]
        # n2 = self.dist_delta.natural2[0, 0, 0]
        # phiphi = self.dist_delta.log_normalizer[0, 0, 0]
        # constant = torch.tensor(
        #     0.5 * dim_latent * np.log(np.pi)
        # )
        # det = - 0.5 * torch.log(torch.linalg.det(- n2))
        # trace = -(torch.linalg.inv(n2) * matmul(n1.unsqueeze(-1), n1.unsqueeze(-2))).sum(dim=(-1, -2)) / 4
        # phiphi_alt = det + constant + trace

        #
        #
        #
        # phi2_nnn = phi2
        # phi2_old = num_observation * phi_prior + phi_delta.sum()
        # phi2_alt = num_observation * phi_prior_alt + phi_delta_alt.sum()
        #
        #





        # Responsabilities tmp1 ~ J x M x 1 (M = N)
        prod1 = torch.matmul((delta2_inv_j - nautral2_inv_j).unsqueeze(1), natural1_factors.unsqueeze(-1))
        prod1 = torch.matmul(natural1_factors.unsqueeze(-2), prod1).squeeze(-1)

        # Responsabilities tmp1 ~ J x M x N (M = N)
        prod2 = torch.matmul(delta2_inv_j.unsqueeze(1), natural1_auxiliary.unsqueeze(-1))
        prod2 = torch.matmul(natural1_factors.unsqueeze(2).unsqueeze(-2), prod2.unsqueeze(1)).squeeze(-1).squeeze(-1)
        sj_mn = - prod1 / 4 + prod2 / 2

        # Log Gamma ~ J x N
        log_numerator = sj_mn.diagonal(dim1=-1, dim2=-2)
        log_denominator = torch.logsumexp(sj_mn, dim=1)
        log_gamma = log_numerator - log_denominator
        log_gamma = log_gamma.sum()

        free_energy = (kl_normalizer + log_gamma) / normalizer

        # old_klprior = self.KLPRIOR.sum()
        # old_klmarg  = self.KLMARGINAL.sum()
        # alt_klall = -(old_klprior + old_klmarg)
        #
        # #new_log_norm = kl_normalizer
        # alt_log_norm = self.dist_variational.log_normalizer.sum() * (self.num_factors + 1)
        #
        # new_log_gamma = log_gamma
        # alt_log_gamma = self.log_gamma.sum()

        return - free_energy

    def fit(self, observations):
        """Fit the model to the observations"""

        # Transform Observation in list if necessary
        observations = [observations]\
            if not (type(observations) is list) and not (type(observations) is tuple) else observations


        # Fit params
        fit_params = self.fit_params
        num_epoch = fit_params['num_epoch']

        # Recognition Factors Parameters
        factors_param = [self.precision_chol_vec_factors]
        for cur_factor in self.recognition_factors:
            factors_param += cur_factor.parameters()

        # Recognition Auxiliary Factors Parameters
        auxiliary_param = [self.precision_chol_vec_auxiliary]
        for cur_factor in self.recognition_auxiliary:
            auxiliary_param += cur_factor.parameters()
            
        # # Prior Parameters TODO: add the gamma priors ?
        # prior_param = self.prior.parameters()

        all_params = [
            #[prior_param, fit_params['prior_params']],
            [factors_param, fit_params['factors_params']],
            [auxiliary_param, fit_params['auxiliary_params']],
        ]

        all_optimizers = [
            opt['optimizer'](param) for param, opt in all_params
        ]

        all_scheduler = [
            params[1]['scheduler'](opt) for params, opt in zip(all_params, all_optimizers)
        ]

        # Fit
        for epoch in range(num_epoch):

            self.epoch = epoch
            batches = self.batches[epoch]

            # Current epoch losses
            loss_batch = []

            for batch_id, batch in enumerate(batches):

                self.batch = batch_id
                self.num_observation_batch = len(batch)
                batched_observations = [obsi[batch] for obsi in observations] \
                    if len(batch) < self.num_observation else observations

                # Forward pass
                self._forward_all(batched_observations)

                # Loss
                loss = self._get_loss()
                loss_batch.append(loss.item())

                # Reset Optimizers
                for opt in all_optimizers:
                    opt.zero_grad()

                # Gradients
                loss.backward()

                # Gradient Steps
                for opt in all_optimizers:
                    opt.step()

                self.get_posteriors(batched_observations)

            # Scheduler Steps
            for sched in all_scheduler:
                sched.step()

            # Gather loss
            self.loss_tot.append(np.mean(loss_batch))



            # Logger
            print_loss(
                self.loss_tot[-1],
                epoch + 1,
                num_epoch,
                pct=self.fit_params['pct']
            )

    def _get_loss_old(self):

        # Log-Gamma term ~ J x N x T -> sum to scalar
        log_gamma = self.log_gamma.sum()

        # KL[Variational || Prior] ~ N x K or N x 1 -> sum to scalar
        KLprior = self._kl_prior().sum()

        # KL[Variational || fhat] ~ J x N x T -> sum to scalar
        KLmarginals = self._kl_marginals().sum()

        # - Loss
        normalizer = self.num_observation_batch * self.len_observation
        free_energy = (log_gamma - KLmarginals - KLprior) / normalizer



    def _kl_marginals(self):

        # Dimension of the problem
        num_observation = self.num_observation_batch  # num_observation = self.num_observation

        # Natural Parameter from the marginal ~ 1 x N x T x K (x K)
        variational = self.dist_variational
        variational_natural1 = variational.natural1
        variational_natural2 = variational.natural2
        variational_log_normalizer = variational.log_normalizer
        variational_suff_stat = [
            variational.suff_stat_mean[0],
            variational.suff_stat_mean[1]
        ]

        # Grasp only the m = n distribution for KL estimation
        # From J x N x N x K (x K) to J x N x K (x K)
        diag_id = range(num_observation)
        factors_delta = self.dist_delta
        diag_delta_natural1 = factors_delta.natural1[:, diag_id, diag_id]
        diag_delta_natural2 = factors_delta.natural2[:, diag_id, diag_id]
        diag_delta_factors_log_normalizer = factors_delta.log_normalizer[:, diag_id, diag_id]

        # KL(q || fhat) ~ num_factors x num_observations x len_observations
        KLmarginals = kl(
            (variational_natural1, variational_natural2),
            (diag_delta_natural1, diag_delta_natural2),
            variational_log_normalizer,
            diag_delta_factors_log_normalizer,
            variational_suff_stat
        )

        return KLmarginals

    def _kl_prior(self):
        """
        KL[variational || prior]

            if inference_mode == parametrised:
                prior       size:     K x M (x M)
                variational size: N x K x M (x M)
                KL div      size: N x K

            if inference_mode == amortized:
                prior       size:     K x M (x M)
                variational size: N x M x K (x K)

                if M == 1: diagonalize prior to
                    prior       size:     1 x K (x K)
                    variational size: N x 1 x K (x K)
                    KL div      size: N x 1

                if M > 1 : diagonalize variational to
                    prior       size:     K x M (x M)
                    variational size: N x K x M (x M)
                    KL div      size: N x K
        """


        natural1 = self.dist_variational.natural1.squeeze(0)
        natural2 = self.dist_variational.natural2.squeeze(0)
        log_normalizer = self.dist_variational.log_normalizer.squeeze(0)
        suff_stat = [
            self.dist_variational.suff_stat_mean[0].squeeze(0),
            self.dist_variational.suff_stat_mean[1].squeeze(0)
        ]

        natural01 = self.dist_prior.natural1.squeeze(0).repeat(natural1.shape[0], 1)
        natural02 = self.dist_prior.natural2.squeeze(0).repeat(natural1.shape[0], 1, 1)
        log_normalizer0 = self.dist_prior.log_normalizer.squeeze(0).repeat(natural1.shape[0])

        # KL(q(U) || p(U)) ~ N x 1 or N x K
        KL = kl(
            [natural1, natural2],
            [natural01, natural02],
            log_normalizer,
            log_normalizer0,
            suff_stat
        )

        return KL

    def _init_all(self, observations):
        """ Init all parameters (see _initializations.Mixin) """

        # Fit parameters
        self._init_fit_params()
        self.dim_latent = self.fit_params['dim_latent']
        self.batches = get_minibatches(
            self.fit_params['num_epoch'],
            self.num_observation,
            self.fit_params['batch_size'],
        )
        self.epoch = 0
        self.batch = 0
        self.num_observation_batch = len(self.batches[self.epoch][self.batch])

        self._init_prior()
        self._init_factors(observations)
        self._init_auxiliary(observations)

        self._init_precision_factors()
        self._init_precision_auxiliary()

def _chol_inv_det(nsd):
    chol = torch.linalg.cholesky(-nsd)
    inv = - torch.cholesky_inverse(chol)
    det = 2 * torch.log(chol.diagonal(dim1=-1, dim2=-2)).sum(dim=-1)

    return inv, det
