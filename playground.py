#%%
import torch
from recognition import MultiInputNet, Net
from flexible_multivariate_normal import vector_to_tril


num_observations = 10
len_observations = 13

# 'default', 'ergodic'
empirical_mixture = 'default'

dim_latent = 3

dim_observations = [
    [100, 20],
    [100],
    [10]
]

observations = [
    torch.rand([num_observations, len_observations, *dims])
    for dims in dim_observations
]

#%% Test Single Inputs

J = 0
dim_input = dim_observations[J]
covariance = 'diag'
kernel_conv = (2,)
kernel_pool = (2,)
channels = (1, 2)
dim_hidden = (10, 10)
dropout = 0.0

test = Net(
    dim_input=dim_input,
    dim_latent=dim_latent,
    dim_hidden=dim_hidden,
    dropout=dropout,
    kernel_conv=kernel_conv,
    kernel_pool=kernel_pool,
    channels=channels,
    covariance=covariance,
)

res = test(observations[J])

J = 1
dim_input = dim_observations[J]
covariance = 'diag'
dim_hidden = (10, 10)
dropout = 0.0

test = Net(
    dim_input=dim_input,
    dim_latent=dim_latent,
    dim_hidden=dim_hidden,
    dropout=dropout,
    covariance=covariance,
    zero_init=True,
)

res = test(observations[J])


print(9)
#%%


dim_input = dim_observations
covariance = 'diag'
kernel_conv = ((2,), (), ())
kernel_pool = ((2,), (), ())
channels = ((1, 2), (), ())
dim_hidden = ((10, 10), (10,), (10,))
dropout = 0.0

dim_hidden_merged = (10, )


test_multi = MultiInputNet(
    dim_input=dim_input,
    dim_latent=dim_latent,
    dim_hidden=dim_hidden,
    dim_hidden_merged=dim_hidden_merged,
    dropout=dropout,
    kernel_conv=kernel_conv,
    kernel_pool=kernel_pool,
    channels=channels,
    covariance=covariance,
    zero_init=True,
)

#test = test_multi(observations)


#%%

# TODO: add an init to zero for all !!!!!!!!!!!

from recognition import FullyParametrised

J = 0

test = FullyParametrised(
                 dim_latent,
                 dim_observations[J],
                 covariance='fixed',
                 init=None,
                 zero_init=True
                )


test(observations[J])

print(9)


#%%

from recognition_parametrised_model import RPM
import torch.nn.functional as F

observation_locations = torch.linspace(0, 1, len_observations).unsqueeze(-1)


factors_params = {
    'channels': [[1, 20, 20], [], []],
    'kernel_conv': [[2, 2], [], []],
    'kernel_pool': [[1, 1], [], []],
    'dim_hidden': [[10, 10], [10, 10], [10, 10]],
    'non_linearity': [F.relu, F.relu, F.relu],
    'covariance': ['fixed', 'fixed', 'fixed'],
}

auxiliary_params = {
    'channels': [[1, 20, 20], [], []],
    'kernel_conv': [[2, 2], [], []],
    'kernel_pool': [[1, 1], [], []],
    'dim_hidden': [[10, 10], [10, 10], [10, 10]],
    'non_linearity': [F.relu, F.relu, F.relu],
    'covariance': ['full', 'full', 'full'],
}

variational_params = {
    'inference_mode': 'amortized',  # 'amortized', 'parametrized'
    'channels': [[1, 20, 20], [], []],
    'kernel_conv': [[2, 2], [], []],
    'kernel_pool': [[1, 1], [], []],
    'dim_hidden': [[10, 10], [10, 10], [10, 10]],
    'dim_hidden_merged': [10, 10],
    'non_linearity': [F.relu, F.relu, F.relu],
    'non_linearity_merged': F.relu,
    'covariance': 'diag',
}

fit_params = {
    'dim_latent': dim_latent,
    'gp_kernel': 'RBF',
    'num_epoch': 10,
    'factors_params': factors_params,
    'auxiliary_params': auxiliary_params,
    'variational_params': variational_params,
}



rpm = RPM(
    observations=observations,
    observation_locations=observation_locations,
    inducing_locations= observation_locations[[0, 2, 4, 6, 8]], # torch.linspace(0, 1, 6).unsqueeze(-1),
    fit_params = fit_params,
)



#%%

import kernels
import torch

aa = kernels.RBFKernel(torch.tensor([0.0]), torch.tensor([1.0]))



