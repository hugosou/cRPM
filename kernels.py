# Imports

import torch
import numpy as np
import torch.nn as nn
from torch import matmul
from utils import diagonalize

__all__ = [
    'Kernel',
    'RBFKernel',
]


def squared_euclidian_distance(locations1, locations2):
    """Distances between locations"""
    # locations1 ~ N1 x D locations2 ~ N2 x D

    # locations1 - locations2 ~ N1 x N2 x D
    diff = locations1.unsqueeze(-2) - locations2.unsqueeze(-3)

    return matmul(diff.unsqueeze(-2), diff.unsqueeze(-1)).squeeze(-1).squeeze(-1)


class Kernel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, locations1, locations2):
        raise NotImplementedError()

    def posteriors(self, locations1, locations2):
        """ Kernel Posterior helpers """

        # Cov_k(M, M) ~ K x M x M
        K_MM = self.forward(locations1, locations1)

        # Identity matrix ~ K x M x M
        Id = 1e-6 * diagonalize(torch.ones(K_MM.shape[:-1], device=locations1.device, dtype=locations1.dtype))

        # inv Cov_k(M, M) ~ K x M x M
        K_MM_inv = torch.linalg.inv(K_MM + Id)

        # Cov_k(T, M) ~ K x T x M
        K_TM = self.forward(locations2, locations1)

        # Cov_k(t,t) ~ K x T (we only keep the diagonal elements)
        K_T = self.forward(locations2, locations2).diagonal(dim1=-1, dim2=-2)

        # Cov_k(T, M) inv( Cov_k(M, M) ) unsqueezed to ~ K x T x M
        K_TM_K_MM_inv = matmul(K_TM, K_MM_inv)

        return K_T, K_MM, K_MM_inv, K_TM, K_TM_K_MM_inv


class RBFKernel(Kernel):
    """Exponentiated quadratic kernel"""

    def __init__(
            self,
            scale,
            lengthscale,
            fit_scale=False,
            fit_lengthscale=True
    ):
        super().__init__()

        self.scale = nn.Parameter(scale, requires_grad=fit_scale)
        self.lengthscale = nn.Parameter(lengthscale, requires_grad=fit_lengthscale)

    def forward(self, locations1, locations2):

        # ||locations1 - locations2||^2 ~ 1 x N1 x N2
        sdist = squared_euclidian_distance(locations1, locations2).unsqueeze(0)

        # Expand and square
        scale_expanded = (self.scale ** 2).unsqueeze(-1).unsqueeze(-1)
        lengthscale_expanded = (self.lengthscale ** 2).unsqueeze(-1).unsqueeze(-1)

        # K(locations1, locations2)
        K = scale_expanded * torch.exp(- 0.5 * sdist / lengthscale_expanded)

        return K
