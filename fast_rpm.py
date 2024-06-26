import torch
from torch import matmul, inference_mode

import numpy as np

import flexible_multivariate_normal
import prior
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

from utils import diagonalize, chol_inv_det


# TODO: use a prior to select dimensions
# TODO: only inititalize auxiliary network if necessary !


class RPM(fast_initializations.Mixin, _updates.Mixin):
    """
    Recognition Parametrised Model (RPM)

    Summary:
        Simple and Fast Recognition Parametrised Factor Analysis (RPFA).

    Args:
        - observations (torch.Tensor or List[torch.Tensor]): Multimodal (possibly time series) observations.
            Sizes:
                len(observations) = num_factors
                observations[j] ~ num_observations x *dim_j
            Where:
                num_factors: number of conditionally independent factors
                num_observations: number of observation samples
                dim_j: dimension of j=th observations
        - loss_tot (List[float]): Stored Loss defined as - Free Energy
        - fit_params (Dict): Fit Parameters. See _initializations.py for Details
        - recognition_factors (recognition.Encoder)

    notation: for compactness, we sometimes denote:
        N: num_observations
        K: dim_latent
        M: num_inducing_points (<= T)


    """

    def __init__(
            self,
            observations: Union[torch.Tensor, List[torch.tensor]],
            loss_tot: List = None,
            fit_params: Dict = None,
            prior: prior.MixturePrior = None,
            recognition_factors: recognition.Encoder = None,
            precision_chol_vec_factors: torch.Tensor = None,
            recognition_variational: recognition.Encoder = None,
            precision_chol_vec_variational: torch.Tensor = None,
            recognition_auxiliary: recognition.Encoder = None,
            precision_chol_vec_auxiliary: torch.Tensor = None,
    ):

        # Transform Observation in list if necessary
        observations = [observations] \
            if not (type(observations) is list) and not (type(observations) is tuple) else observations

        # Problem dimensions
        self.num_factors = len(observations)
        self.num_observation = observations[0].shape[0]

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
        self.prior = prior
        self.recognition_factors = recognition_factors
        self.recognition_variational = recognition_variational
        self.precision_chol_vec_factors = precision_chol_vec_factors
        self.precision_chol_vec_variational = precision_chol_vec_variational
        # self.recognition_auxiliary = recognition_auxiliary
        # self.precision_chol_vec_auxiliary = precision_chol_vec_auxiliary

        self._init_all(observations)

        # Sanity Checks
        assert all([i.shape[0] == self.num_observation for i in observations]), "Inconsistent number of observations"

        # Init Forwarded
        self.forwarded_factors = None

        # Init Distributions and parameters
        self.dist_factors = None
        self.dist_variational = None

        with torch.no_grad():
            batch = self.batches[0][0]
            batched_observations = [obsi[batch] for obsi in observations] \
                if len(batch) < self.num_observation else observations
            self._forward_all(batched_observations)
            self.loss_tot.append(self._get_loss().item())

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
        self._init_precision_factors()

        self._init_variational()
        self._init_precision_variational()

        #self._init_auxiliary(observations)
        #self._init_precision_auxiliary()

    def _forward_all(self, observations):
        """ Forward Neural Networks"""
        
        self._forward_factors(observations)
        self._forward_offset()
        self._forward_variational()
        self._forward_auxiliary(observations)

    def _forward_offset(self):

        if self.fit_params['auxiliary_mode'] == 'constrained_moment_matched':
            natural1_factors, natural2_factors = self.forwarded_factors

            # Mean and Variance (over N) of the first natural parameters
            mean_natural1_factors = natural1_factors.mean(1)
            vari_natural1_factors = (
                    matmul(natural1_factors.unsqueeze(-1), natural1_factors.unsqueeze(-2)).mean(dim=1)
                    - matmul(mean_natural1_factors.unsqueeze(-1), mean_natural1_factors.unsqueeze(-2))
            )

            VarIm1 = torch.linalg.inv(
                torch.eye(self.dim_latent, dtype=self.dtype, device=self.device)
                - 0.5 * torch.matmul(vari_natural1_factors, torch.linalg.inv(natural2_factors))
            )

            # Natural Parameters ~ J x 1 x K and J x K x K
            natural1_moment_matched = matmul(VarIm1, mean_natural1_factors.unsqueeze(-1)).squeeze(-1).unsqueeze(1)
            natural2_moment_matched = matmul(VarIm1, natural2_factors)

            # Store
            self.forwarded_offset = [natural1_moment_matched, natural2_moment_matched]

        elif self.fit_params['auxiliary_mode'] == 'constrained_prior':

            if self.prior.num_centroids == 1:

                # Natural Parameters ~ 1 x 1 x K and 1 x K x K
                natural1_prior = self.prior.natural1.unsqueeze(0)
                natural2_prior = self.prior.natural2()

                # Store
                self.forwarded_offset = [natural1_prior, natural2_prior]

            else:
                # TODO: Maybe moment match the prior in case of a mixture ?
                raise NotImplementedError()

        else:
            raise NotImplementedError()

    def _forward_variational(self):

        if self.prior.num_centroids == 1:
            # The variational Update is closed Form

            # Prior Params ~ 1 x K and K x K
            natural1_prior = self.prior.natural1
            natural2_prior = self.prior.natural2().squeeze(dim=0)

            # Offset Params
            natural1_offset, natural2_offset = self.forwarded_offset

            # Recognition Factors Params
            natural1_factors, natural2_factors = self.forwarded_factors

            # Variational Parameters N x K and N x K x K
            natural1_variational = natural1_prior + (natural1_factors - natural1_offset).sum(dim=0)
            natural2_variational = natural2_prior + (natural2_factors - natural2_offset).sum(dim=0)


        else:

            # Recognised factors ~ J x N x K
            natural1_factors, _ = self.forwarded_factors

            # New shape ~ N x JK
            reshaped_input_size = [natural1_factors.shape[1], self.dim_latent * self.num_factors]
            natural1_input = natural1_factors.permute(1, 0, 2).reshape(reshaped_input_size)

            # Combine recognition distribution into variational N x K and K x K
            natural1_variational = self.recognition_variational(natural1_input)
            natural2_variational = self.precision_chol_vec_variational.precision()

        self.forwarded_variational = [natural1_variational, natural2_variational]

    def _forward_factors(self, observations):
        """ Recognition Factors"""

        # 1st natural parameter: Forward recognition networks ~ J x N x K
        recognition_factors = self.recognition_factors
        natural1_factors = torch.cat(
            [facti(obsi).unsqueeze(0) for facti, obsi in zip(recognition_factors, observations)],
            axis=0
        )

        # 2nd natural parameter ~ J x K x K # TODO: use the class method
        natural2_factors_tril = vector_to_tril(self.precision_chol_vec_factors.chol_vec)
        natural2_factors = - torch.matmul(natural2_factors_tril, natural2_factors_tril.transpose(-1, -2))

        auxiliary_mode = self.fit_params['auxiliary_mode']
        if auxiliary_mode in ['flexible', 'constrained_prior'] and self.prior.num_centroids == 1:
            # Prior Distributions
            natural1_prior = self.prior.natural1
            natural2_prior = self.prior.natural2()

            natural1_factors += natural1_prior.unsqueeze(0)
            natural2_factors += natural2_prior

        # Store
        self.forwarded_factors = [natural1_factors, natural2_factors]

    def _forward_auxiliary(self, observations):

        # Prior or moment match denominator ~ 1 x 1 x K or J x 1 x K  and 1 x K x K
        natural1_offset, natural2_offset = self.forwarded_offset

        # Variational ~ N x K and K x K
        natural1_variational, natural2_variational = self.forwarded_variational

        # Auxiliary ~ J x N x K (or 1 x N x K) and J x K x K (or 1 x K x K)
        natural1_auxiliary = natural1_variational.unsqueeze(0) - natural1_offset
        natural2_auxiliary = natural2_variational.unsqueeze(0) - natural2_offset

        self.forwarded_auxiliary = [natural1_auxiliary, natural2_auxiliary]


    def get_posteriors(self, observations):
        """ Approximate Posterior (Variational) and Recognition Posterior Distributions"""

        with torch.no_grad():

            # Dimension of the problem
            num_factors = self.num_factors

            for reci in self.recognition_factors:
                reci.eval()

            if self.recognition_variational is not None:
                self.recognition_variational.eval()

            # Model Distribution
            self._forward_all(observations)

            # Forwarded parameters
            natural1_factors, natural2_factors = self.forwarded_factors
            natural1_variational, natural2_variational = self.forwarded_variational

            # Some Reshaping
            natural2_factors = natural2_factors.unsqueeze(1).repeat(1, natural1_factors.shape[1], 1, 1)
            natural2_variational = natural2_variational.unsqueeze(0).repeat(natural1_variational.shape[0], 1, 1)

            # Variational Distribution
            distribution_variational = FlexibleMultivariateNormal(
                natural1_variational,
                natural2_variational,
                init_natural=True,
                init_cholesky=False,
                store_suff_stat_mean=True,
            )

            # Factors Distribution
            distribution_factors = FlexibleMultivariateNormal(
                natural1_factors,
                natural2_factors,
                init_natural=True,
                init_cholesky=False,
                store_suff_stat_mean=True,
            )
            
            self.dist_factors = distribution_factors
            self.dist_variational = distribution_variational

            for reci in self.recognition_factors:
                reci.train()

            if self.recognition_variational is not None:
                self.recognition_variational.train()

        return distribution_variational, distribution_factors

    def _get_loss(self):

        # Problem Dimensions
        dim_latent = self.dim_latent
        num_factors = self.num_factors
        num_observation = self.num_observation_batch
        normalizer = self.num_observation_batch

        # Natural parameters
        natural1_factors, natural2_factors = self.forwarded_factors
        natural1_auxiliary, natural2_auxiliary = self.forwarded_auxiliary

        # Delta natural1 for each factor ~ J x N x K
        delta_natural1_jn = natural1_factors + natural1_auxiliary

        # ~ N x K
        delta_natural1_n = delta_natural1_jn.sum(0)

        # Delta natural2 for each factor ~ J x K x K
        delta_natural2_j = natural2_factors + natural2_auxiliary

        # Delta natural2 ~ K x K
        delta_natural2 = delta_natural2_j.sum(dim=0)

        # ~ K x K
        var_natural1 = matmul(delta_natural1_n.unsqueeze(-1), delta_natural1_n.unsqueeze(-2)).sum(dim=0)

        # ~ J x K x K
        var_natural1_j = matmul(delta_natural1_jn.unsqueeze(-1), delta_natural1_jn.unsqueeze(-2)).sum(dim=1)

        # Invert and Determinent ~ J x K x K and J x 1 x 1
        delta2_inv_j, delta2_logdet_j = chol_inv_det(delta_natural2_j)
        
        # ~ J x K x K
        nautral2_inv_j, _ = chol_inv_det(natural2_factors)

        # Responsabilities tmp1 ~ J x M x 1 (M = N)
        prod1 = torch.matmul((delta2_inv_j - nautral2_inv_j).unsqueeze(1), natural1_factors.unsqueeze(-1))
        prod1 = torch.matmul(natural1_factors.unsqueeze(-2), prod1).squeeze(-1)

        # Responsabilities tmp1 ~ J x M x N (M = N)
        prod2 = torch.matmul(delta2_inv_j.unsqueeze(1), natural1_auxiliary.unsqueeze(-1))
        prod2 = torch.matmul(natural1_factors.unsqueeze(2).unsqueeze(-2), prod2.unsqueeze(1)).squeeze(-1).squeeze(-1)
        sj_mn = - prod1 / 4 - prod2 / 2

        # Log Gamma ~ J x N
        log_numerator = sj_mn.diagonal(dim1=-1, dim2=-2)
        log_denominator = torch.logsumexp(sj_mn, dim=1)
        log_gamma = log_numerator - log_denominator
        log_gamma = log_gamma.sum()

        # KL terms
        if self.prior.num_centroids == 1:
            # Closed Form Updates for the variational distribution

            # Prior paramters ~ K (x K)
            # natural1_prior = self.prior.natural1.squeeze(0)
            natural2_prior = self.prior.natural2().squeeze(0)

            # Invert and Determinent ~ K x K and 1 x 1
            # Id = - 1e10 * torch.eye(self.dim_latent, dtype = self.dtype, device=self.device)
            delta2_inv, delta2_logdet = chol_inv_det(delta_natural2 + natural2_prior)

            # Log Normalizer of the average
            constant1 = torch.tensor(
                0.5 * dim_latent * num_observation * (1 + num_factors) * np.log(np.pi * (1 + num_factors)),
                device = self.device, dtype=self.dtype
            )
            logdet1 = - 0.5 * num_observation * (1 + num_factors) * delta2_logdet
            trace1 = - (delta2_inv * var_natural1).sum() / 4
            phi1 = constant1 + logdet1 + trace1

            # Average Of the log normalizer
            constant2 = torch.tensor(
                0.5 * dim_latent * num_observation * np.log(np.pi),
                device=self.device, dtype=self.dtype
            )
            logdet2 = - 0.5 * num_observation * torch.log(torch.linalg.det(-natural2_prior))
            trace2 = 0
            phi2 = constant2 + logdet2 + trace2

            constant3 = torch.tensor(
                0.5 * dim_latent * num_observation * num_factors * np.log(np.pi),
                device=self.device, dtype=self.dtype
            )
            trace3 = - (delta2_inv_j * var_natural1_j).sum() / 4
            logdet3 = - 0.5 * delta2_logdet_j.sum() * num_observation
            phi3 = constant3 + logdet3 + trace3

            # KL normalizer
            kl_divergences = phi1 - phi2 - phi3


        else:
            # TODO: check that the 2 methods agree

            # Helpful constants
            sqrt2 = torch.tensor(np.sqrt(2), device=self.device, dtype=self.dtype)
            log2pi = torch.tensor(np.log(2 * np.pi), device=self.device, dtype=self.dtype)

            # Prior Distributions
            # Mixture weights
            mixture_weights = self.prior.responsabilities()
            # 1st and 2nd Natural Parameter
            natural1_prior = self.prior.natural1
            natural2_prior = self.prior.natural2()
            # log normaliser
            log_normaliser_prior = FlexibleMultivariateNormal(
                natural1_prior,
                natural2_prior,
                init_natural=True,
                init_cholesky=False
            ).log_normalizer

            # Variational Distribution
            # 1st and 2nd Natural Parameter
            natural1_variational, natural2_variational = self.forwarded_variational
            # TODO PUT BACK
            # natural2_variational_chol = vector_to_tril(self.precision_chol_vec_variational.chol_vec)
            natural2_variational_chol = torch.linalg.cholesky(-natural2_variational)
            # Efficient Inversion and determinant
            natural2_variational_inv = - torch.cholesky_inverse(natural2_variational_chol)
            half_logdet_variational = torch.log(sqrt2 * natural2_variational_chol.diagonal(dim1=-1, dim2=-2)).sum(-1)
            # Expectation of the Sufficient statistics
            variational_vari = - 0.5 * natural2_variational_inv
            variational_mean = matmul(variational_vari, natural1_variational.unsqueeze(-1)).squeeze(-1)
            variational_meanmean_vari = variational_vari + matmul(
                variational_mean.unsqueeze(-1),
                variational_mean.unsqueeze(-2)
            )
            # log normalizer of the variational
            malhobilis = matmul(
                variational_mean.unsqueeze(-2),
                natural1_variational.unsqueeze(-1)
            ).squeeze(-1).squeeze(-1)
            log_normalizer_variational = - half_logdet_variational + 0.5 * malhobilis + 0.5 * self.dim_latent * log2pi

            # Natural Parameters of fhat
            natural1_delta, natural2_delta = [
                fac + aux for fac, aux in zip(self.forwarded_factors, self.forwarded_auxiliary)
            ]
            natural2_delta_chol = torch.linalg.cholesky(-natural2_delta)
            # Efficient Inversion and determinant
            natural2_delta_inv = - torch.cholesky_inverse(natural2_delta_chol)
            natural2_delta_cov = - 0.5 * natural2_delta_inv
            delta_mean = matmul(natural2_delta_cov.unsqueeze(1), natural1_delta.unsqueeze(-1)).squeeze(-1)
            half_logdet_delta = torch.log(sqrt2 * natural2_delta_chol.diagonal(dim1=-1, dim2=-2)).sum(dim=-1)
            # log normalizer of fhats ~ J x N
            malhobilis_delta = matmul(
                delta_mean.unsqueeze(-2),
                natural1_delta.unsqueeze(-1)
            ).squeeze(-1).squeeze(-1)
            log_normalizer_delta = - half_logdet_delta.unsqueeze(-1) + 0.5 * malhobilis_delta + 0.5 * self.dim_latent * log2pi

            # KL[variational || Prior] ~ N x U
            delta_log_normaliser = log_normaliser_prior.unsqueeze(0) - log_normalizer_variational.unsqueeze(1)
            delta_natural1 = natural1_prior.unsqueeze(0) -  natural1_variational.unsqueeze(1)
            delta_natural2 = natural2_prior.unsqueeze(0) - natural2_variational.unsqueeze(0).unsqueeze(1)
            delta_product = (
                    (variational_mean.unsqueeze(1) * delta_natural1).sum(-1)
                    + (variational_meanmean_vari.unsqueeze(1) * delta_natural2).sum(dim=(-1, -2))
            )
            KLqpu = delta_log_normaliser - delta_product

            # KL[variational || fhat] ~ J x N
            delta_log_normaliser = log_normalizer_delta - log_normalizer_variational.unsqueeze(0)
            delta_natural1 = natural1_delta - natural1_variational.unsqueeze(0)
            delta_natural2 = natural2_delta - natural2_variational.unsqueeze(0)
            delta_product = (
                    (variational_mean.unsqueeze(0) * delta_natural1).sum(-1)
                    + (variational_meanmean_vari.unsqueeze(0) * delta_natural2.unsqueeze(1)).sum(dim=(-1, -2))
            )
            KLqfhat = delta_log_normaliser - delta_product

            # Approximate
            log_beta = torch.logsumexp(torch.log(mixture_weights).unsqueeze(0) - KLqpu, dim=-1)

            # KL divergences - KL[q || prior] - KL[q || fhat]
            kl_divergences = log_beta.sum() - KLqfhat.sum()

            # tmp1 = natural1_variational.unsqueeze(1).repeat(1, self.prior.num_centroids, 1)
            # tmp2 = natural2_variational.unsqueeze(0).unsqueeze(1).repeat(tmp1.shape[0], self.prior.num_centroids, 1, 1)
            # tmp = FlexibleMultivariateNormal(
            #     tmp1,
            #     tmp2,
            #     init_natural =True,
            #     init_cholesky=False,
            #     store_suff_stat_mean=True,
            #     jitter=0.0,
            # )
            # pri1 = natural1_prior.unsqueeze(0).repeat(tmp1.shape[0], 1, 1)
            # pri2 = natural2_prior.unsqueeze(0).repeat(tmp1.shape[0], 1, 1, 1)
            # prio = FlexibleMultivariateNormal(
            #     pri1,
            #     pri2,
            #     init_natural=True,
            #     init_cholesky=False,
            #     store_suff_stat_mean=True,
            #     jitter=0.0,
            # )
            # delta_dist = FlexibleMultivariateNormal(
            #     natural1_delta,
            #     natural2_delta.unsqueeze(1).repeat(1, natural1_delta.shape[1], 1, 1),
            #     init_natural=True,
            #     init_cholesky=False,
            # )
            # qdelta_dist = FlexibleMultivariateNormal(
            #     natural1_variational.unsqueeze(0).repeat(self.num_factors, 1, 1),
            #     natural2_variational.unsqueeze(0).unsqueeze(0).repeat(self.num_factors, natural1_delta.shape[1], 1, 1),
            #     init_natural=True,
            #     init_cholesky=False,
            #     store_suff_stat_mean=True,
            # )
            #
            # KLqpu_alt = flexible_kl(tmp, prio)
            # KLqfh_alt = flexible_kl(qdelta_dist, delta_dist)
            #
            # torch.allclose(KLqpu, KLqpu_alt)
            # torch.allclose(KLqfhat, KLqfh_alt)

        # Lower (lower) Bound to the log likelihood
        free_energy = (kl_divergences + log_gamma) / normalizer

        return - free_energy

    def fit(self, observations):
        """Fit the model to the observations"""

        # Transform Observation in list if necessary
        observations = [observations] \
            if not (type(observations) is list) and not (type(observations) is tuple) else observations

        # Fit params
        fit_params = self.fit_params
        num_epoch = fit_params['num_epoch']
        
        # Recognition Factors Parameters
        factors_param = [*self.precision_chol_vec_factors.parameters()]
        for cur_factor in self.recognition_factors:
            factors_param += cur_factor.parameters()

        prior_params = self.prior.parameters()

        all_params = [
            [factors_param, fit_params['factors_params']],
            [prior_params, fit_params['prior_params']]
        ]

        if self.recognition_variational is not None:
            variational_params = [*self.precision_chol_vec_variational.parameters()]
            variational_params += self.recognition_variational.parameters()

            all_params.append(
                [variational_params, fit_params['variational_params']]
            )

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
                loss.backward(retain_graph=True)

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



    def _toggle_auxilary(self, observations, optimizers, schedulers):

        if self.fit_params['auxiliary_mode'] == 'flexible':

            # Recognition Factors Parameters
            auxiliary_param = [*self.precision_chol_vec_auxiliary.parameters()]
            for cur_factor in self.recognition_auxiliary:
                auxiliary_param += cur_factor.parameters()

            optimizer = self.fit_params['auxiliary_params']['optimizer'](auxiliary_param)
            scheduler = self.fit_params['auxiliary_params']['scheduler'](optimizer)

            optimizers.append(optimizer)
            schedulers.append(scheduler)

        return optimizers, schedulers


        # with torch.no_grad():
        #     natural1_prior, natural2_prior = self.forwarded_prior
        #     natural1_factors, natural2_factors = self.forwarded_factors
        #     natural1_auxiliary, natural2_auxiliary = self.forwarded_auxiliary
        #
        #     # Variational Distribution
        #     natural1_variational = (natural1_prior.unsqueeze(0) + (natural1_factors - natural1_auxiliary).sum(dim=0)) / (1 + self.num_factors)
        #     natural2_variational = (natural2_prior + (natural2_factors - natural2_auxiliary).sum(dim=0)) / (1 + self.num_factors)
        #     natural2_variational = natural2_variational.unsqueeze(0).repeat(natural1_variational.shape[0], 1, 1)
        #     forwarded_variational = FlexibleMultivariateNormal(
        #         natural1_variational,
        #         natural2_variational,
        #         init_natural=True,
        #         init_cholesky=False,
        #         jitter = 0.0,
        #         store_suff_stat_mean=True,
        #     )
        #
        #     # Prior Distribution
        #     forwarded_prior = FlexibleMultivariateNormal(
        #         natural1_prior,
        #         natural2_prior,
        #         init_natural=True,
        #         init_cholesky=False,
        #         jitter=0.0,
        #     )
        #
        #     # Delta Distribution
        #     natural1_delta = natural1_factors - natural1_auxiliary
        #     natural2_delta = natural2_factors - natural2_auxiliary
        #     natural2_delta = natural2_delta.unsqueeze(1).repeat(1, natural1_delta.shape[1], 1, 1)
        #     forwarded_delta = FlexibleMultivariateNormal(
        #         natural1_delta,
        #         natural2_delta,
        #         init_natural=True,
        #         init_cholesky=False,
        #         jitter = 0.0,
        #     )
        #
        #     self.phi_variational = forwarded_variational.log_normalizer * (self.num_factors + 1)
        #     self.phi_prior = forwarded_prior.log_normalizer * self.num_observation_batch
        #     self.phi_delta = forwarded_delta.log_normalizer
        #     self.kl_normalizer = self.phi_variational.sum() - (self.phi_prior.sum() + self.phi_delta.sum())
        #
        #     alt_natural1_variational = natural1_variational.unsqueeze(0).repeat(self.num_factors, 1, 1)
        #     alt_natural2_variational = natural2_variational.unsqueeze(0).repeat(self.num_factors, 1, 1, 1)
        #     alt_forwarded_variational = FlexibleMultivariateNormal(
        #         alt_natural1_variational,
        #         alt_natural2_variational,
        #         init_natural=True,
        #         init_cholesky=False,
        #         store_suff_stat_mean = True,
        #     )
        #     KLqf = flexible_kl(alt_forwarded_variational, forwarded_delta).sum()
        #
        #     alt_natural1_prior = natural1_prior.unsqueeze(0).repeat(natural1_variational.shape[0], 1)
        #     alt_natural2_prior = natural2_prior.unsqueeze(0).repeat(natural1_variational.shape[0], 1, 1)
        #     alt_forwarded_prior = FlexibleMultivariateNormal(
        #         alt_natural1_prior,
        #         alt_natural2_prior,
        #         init_natural=True,
        #         init_cholesky=False,
        #     )
        #     KLqp = flexible_kl(forwarded_variational, alt_forwarded_prior).sum()
        #     self.KL = -(KLqf + KLqp)

    # TODO: Maybe reinstantiate this latter
    # elif self.fit_params['auxiliary_mode'] == 'flexible':
    #     # 1st Natural Parameter
    #     recognition_auxiliary = self.recognition_auxiliary
    #     natural1_auxiliary_tmp = torch.cat(
    #         [facti(obsi).unsqueeze(0) for facti, obsi in zip(recognition_auxiliary, observations)],
    #         axis=0
    #     )
    #
    #     # 2nd natural parameter
    #     precision_chol_auxiliary = vector_to_tril(self.precision_chol_vec_auxiliary.chol_vec)
    #     natural2_auxiliary_tmp = - matmul(precision_chol_auxiliary, precision_chol_auxiliary.transpose(-1, -2))
    #
    #     natural1_auxiliary_offset = (natural1_factors - natural1_prior).sum(dim=0, keepdims=True).repeat(num_factors, 1, 1)
    #     natural2_auxiliary_offset = (natural2_factors - natural2_prior).sum(dim=0, keepdims=True).repeat(num_factors, 1, 1)
    #
    #     natural1_auxiliary = natural1_auxiliary_offset + natural1_auxiliary_tmp - natural1_factors
    #     natural2_auxiliary = natural2_auxiliary_offset + natural2_auxiliary_tmp - (natural2_factors - natural2_prior)
    # # Auxiliary Distributions
    # natural1_auxiliary = (
    #         natural1_prior - natural1_moment_matched
    #         + (natural1_factors - natural1_moment_matched).sum(dim=0, keepdims=True)
    # )
    # natural2_auxiliary = (
    #         natural2_prior - natural2_moment_matched
    #         + (natural2_factors - natural2_moment_matched).sum(dim=0, keepdims=True)
    # )

    # delta = self.phi_variational.sum().item() - phi1.item()
    # print('Efficient Method Phi1 = ' + str(phi1.item()))
    # print('Full Dist Method Phi1 = ' + str(self.phi_variational.sum().item()))
    # print('Deltas    Method Phi1 = ' + str(delta))
    # print('NormDelta Method Phi1 = ' + str(delta / phi1.item()))
    # print('')
    #
    # delta = self.phi_prior.item() - phi2.item()
    # print('Efficient Method Phi2 = ' + str(phi2.item()))
    # print('Full Dist Method Phi2 = ' + str(self.phi_prior.item()))
    # print('Deltas    Method Phi2 = ' + str(delta))
    # print('NormDelta Method Phi2 = ' + str(delta / phi2.item()))
    # print('')
    #
    #
    # delta = self.phi_delta.sum().item() - phi3.item()
    # print('Efficient Method Phi3 = ' + str(phi3.item()))
    # print('Full Dist Method Phi3 = ' + str(self.phi_delta.sum().item()))
    # print('Deltas    Method Phi3 = ' + str(delta))
    # print('NormDelta Method Phi3 = ' + str(delta / phi3.item()))
    # print('')
    #
    # delta = self.kl_normalizer.item() - kl_normalizer.item()
    # print('Efficient Method KLn = ' + str(kl_normalizer.item()))
    # print('Full Dist Method KLn = ' + str(self.kl_normalizer.item()))
    # print('Deltas    Method KLn = ' + str(delta))
    # print('NormDelta Method KLn = ' + str(delta / kl_normalizer.item()))
    # print('')
    #
    # delta = self.KL.item() - kl_normalizer.item()
    # print('Efficient Method KL2 = ' + str(kl_normalizer.item()))
    # print('Full Dist Method KL2 = ' + str(self.KL.item()))
    # print('Deltas    Method KL2 = ' + str(delta))
    # print('NormDelta Method KL2 = ' + str(delta / kl_normalizer.item()))
    # print('')

    # if self.recognition_auxiliary is not None:
    #     for reci in self.recognition_auxiliary:
    #         reci.eval()