import torch
import torch.nn as nn
from torch import matmul
from kernels import Kernel, RBFKernel
from typing import List, Union


class GPPrior(nn.Module):

    def __init__(
            self,
            mean0: List[Union[torch.Tensor, bool]],
            scale0: List[Union[torch.Tensor, bool]],
            scale1: List[Union[torch.Tensor, bool]],
            lengthscale0: List[Union[torch.Tensor, bool]],
            lengthscale1: List[Union[torch.Tensor, bool]],
            covariance_type0: str = 'RBF',
            covariance_type1: str = 'RBF',
            **kwargs,
    ):
        super().__init__()

        # Mean Prior Prior Param
        self.mean0 = _init_nn_param(mean0)

        # Prior (Prior) Scale
        self.scale0 = _init_nn_param(scale0)
        self.scale1 = _init_nn_param(scale1)

        # Prior (Prior) Length Scales
        self.lengthscale0 = _init_nn_param(lengthscale0)
        self.lengthscale1 = _init_nn_param(lengthscale1)

        if covariance_type0 == 'RBF':
            covariance0 = RBFKernel(
                self.scale0,
                self.lengthscale0
            )
        else:
            raise NotImplementedError()

        if covariance_type1 == 'RBF':
            self.covariance = RBFKernel(
                self.scale1,
                self.lengthscale1
            )
        else:
            raise NotImplementedError()

        self.mean = MeanPrior(
            self.mean0,
            covariance0,
        )


class MeanPrior(nn.Module):
    def __init__(
            self,
            mean: torch.Tensor,
            covariance: Kernel
    ):

        super().__init__()

        self.mean = mean
        self.covariance = covariance

    def forward(self, locations1, locations2):

        mean = self.mean
        cova = self.covariance(locations1, locations2)

        return matmul(cova, mean.unsqueeze(-1)).squeeze(-1)


def _init_nn_param(
        params: List[Union[torch.Tensor, bool]]
):
    return torch.nn.Parameter(params[0], requires_grad=params[1])





