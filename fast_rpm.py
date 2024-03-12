import torch
from torch import matmul

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
            [
                facti(obsi).unsqueeze(0)
                for facti, obsi in zip(self.recognition_auxiliary, observations)
            ],
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

        # Constant tern
        constant1 = 0.5 * np.log(2 * np.pi) * num_observation * dim_latent
        constant2 = - 0.5 * num_observation * dim_latent * (1 + num_factors) * np.log(2 / (num_factors + 1))
        constants = torch.tensor(constant1 + constant2, dtype=self.dtype, device=self.device)

        # All natural parameters
        _, natural2_prior = self.forwarded_prior
        natural1_factors, natural2_factors = self.forwarded_factors
        natural1_auxiliary, natural2_auxiliary = self.forwarded_auxiliary

        # Delta natural for each factor ~ J x N x K
        deltaj1 = natural1_factors - natural1_auxiliary
        deltaj2 = natural2_factors - natural2_auxiliary

        # Delta natural ~ N x K
        delta1 = deltaj1.sum(dim=0)
        delta2 = deltaj2.sum(dim=0)

        # Variance of the first parameter
        delta1_var = matmul(delta1.unsqueeze(-1), delta1.unsqueeze(-2)).sum(dim=0)

        # Inverse naturals 2 ~ J x K x K
        Id = torch.eye(self.dim_latent, dtype=self.dtype, device=self.device).unsqueeze(0)
        deltaj2_inv = torch.linalg.inv(deltaj2 - 1e-8 * Id)
        tildej2_inv = torch.linalg.inv(natural2_factors)

        # Cholesky Decompose, Invert and Determinant
        choldec = torch.linalg.cholesky(-(natural2_prior + delta2))
        cholinv = - torch.cholesky_inverse(choldec)
        choldet = 2 * torch.log(choldec.diagonal(dim1=-1, dim2=-2)).sum()

        # Overal Log Normaliser
        mahalanobis = - (cholinv * delta1_var).sum() / 4
        volume = - num_observation * (num_factors + 1) * choldet / 2
        log_normalizer = mahalanobis + volume

        # Responsabilities tmp1 ~ J x M x 1 (M = N)
        prod1 = torch.matmul((deltaj2_inv - tildej2_inv).unsqueeze(1), natural1_factors.unsqueeze(-1))
        prod1 = torch.matmul(natural1_factors.unsqueeze(-2), prod1).squeeze(-1)

        # Responsabilities tmp1 ~ J x M x N (M = N)
        prod2 = torch.matmul(deltaj2_inv.unsqueeze(1), natural1_auxiliary.unsqueeze(-1))
        prod2 = torch.matmul(natural1_factors.unsqueeze(2).unsqueeze(-2), prod2.unsqueeze(1)).squeeze(-1).squeeze(-1)
        sj_mn = - (prod1 - 2 * prod2) / 4

        # Log Gamma ~ J x N
        log_numerator = sj_mn.diagonal(dim1=-1, dim2=-2)
        log_denominator = torch.logsumexp(sj_mn, dim=1)
        log_gamma = log_numerator - log_denominator
        log_gamma = log_gamma.sum()

        free_energy = (constants + log_normalizer + log_gamma) / normalizer

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
        variational = self.dist_marginals
        variational_natural1 = variational.natural1.unsqueeze(0)
        variational_natural2 = variational.natural2.unsqueeze(0)
        variational_log_normalizer = variational.log_normalizer.unsqueeze(0)
        variational_suff_stat = [
            variational.suff_stat_mean[0].unsqueeze(0),
            variational.suff_stat_mean[1].unsqueeze(0)
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

        # Amortized or Parametrized variational inference
        inference_mode = self.fit_params['variational_params']['inference_mode']

        # Prior's Mean and Variance: size ~ K x M x (x M)
        prior_mean = self.prior.mean(self.inducing_locations, self.inducing_locations)
        prior_vari = self.prior.covariance(self.inducing_locations, self.inducing_locations)

        if inference_mode == 'amortized':

            # To size ~ 1 x K x M x (x M)
            prior_mean = prior_mean.unsqueeze(0)
            prior_vari = prior_vari.unsqueeze(0)
            # prior_nat1 = prior_nat1.unsqueeze(0)

            # Amortized Variational Parameters: size N x M x K ( x K)
            variational_natural = (self.dist_variational.natural1, self.dist_variational.natural2)
            variational_suffstat = self.dist_variational.suff_stat_mean
            variational_log_normalizer = self.dist_variational.log_normalizer

            if self.num_inducing_points == 1:
                # Not a time series (M = 1):
                # Reshape Prior distribution to 1 x 1 x K ( x K) by leveraging M = 1

                # Prior's Natural Parameters (reshape only works since M = 1)
                prior_mean = prior_mean.reshape(1, 1, self.dim_latent)
                prior_vard = prior_vari.reshape(1, 1, self.dim_latent)
                prior_nat1 = prior_mean / prior_vard
                prior_nat2 = - 0.5 * diagonalize(1 / (prior_vard + 1e-6))
                prior_natural = (prior_nat1, prior_nat2)

                # Prior Log-Normalizer
                prior_log_normalizer = get_log_normalizer(prior_mean, prior_vard, prior_natural[0])

            else:
                # Time series (M > 1)
                # Reshape variational distribution to N x K x M ( x M) by leveraging diagonal over K
                variational_nat1 = variational_natural[0].permute(0, 2, 1)
                variational_nat2 = variational_natural[1].diagonal(dim1=-1, dim2=-2).permute(0, 2, 1)
                variational_vard = - 0.5 * 1 / (variational_nat2 - 1e-6)
                variational_vari = diagonalize(variational_vard)
                variational_mean = variational_vard * variational_nat1
                variational_nat2 = diagonalize(variational_nat2)
                variational_log_normalizer = get_log_normalizer(variational_mean, variational_vard, variational_nat1)
                variational_natural = (variational_nat1, variational_nat2)
                variational_suffstat = (
                    variational_mean,
                    variational_vari + matmul(variational_mean.unsqueeze(-1), variational_mean.unsqueeze(-2))
                )

                dist_prior = FlexibleMultivariateNormal(
                    prior_mean,
                    prior_vari,
                    init_natural=False,
                    init_cholesky=False,
                )

                prior_natural = (dist_prior.natural1, dist_prior.natural2)
                prior_log_normalizer = dist_prior.log_normalizer

            # KL(q(U) || p(U)) ~ N x 1 or N x K
            KL = kl(
                variational_natural,
                prior_natural,
                variational_log_normalizer,
                prior_log_normalizer,
                variational_suffstat
            )

        elif inference_mode == 'parametrized':

            # Build Prior Distribution
            dist_prior = FlexibleMultivariateNormal(
                prior_mean,
                prior_vari,
                init_natural=False,
                init_cholesky=False,
            )

            # KL(q(U) || p(U)) ~ N x K
            KL = flexible_kl(
                self.dist_variational,
                dist_prior,
                repeat1=[0, 1],
                repeat2=[1]
            )

        else:
            raise NotImplementedError()

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
