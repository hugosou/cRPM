# %%
import torch
from recognition import MultiInputNet, Net
from flexible_multivariate_normal import vector_to_tril

num_observations = 10
len_observations = 1

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

#%%


import torch
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print (x)
else:
    print ("MPS device not found.")


from time import time
import numpy as np

#%%


x1 = torch.rand(10000, 10000)

x2 = torch.rand((1000, 1000), device= mps_device)

#%%

t0 = time()
torch.linalg.matmul(x1, x1)
t1 = time()
print(str(np.round(t1 - t0, 2)) + 'sec')

# %% Test Single Inputs

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

# %%


dim_input = dim_observations
covariance = 'diag'
kernel_conv = ((2,), (), ())
kernel_pool = ((2,), (), ())
channels = ((1, 2), (), ())
dim_hidden = ((10, 10), (10,), (10,))
dropout = 0.0

dim_hidden_merged = (10,)

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

# test = test_multi(observations)


# %%

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

# %%

from recognition_parametrised_model import RPM
import torch.nn.functional as F

observation_locations = torch.linspace(0, 1, len_observations).unsqueeze(-1)

prior_params = {
    'gp_kernel': 'RBF',
    'optimizer': {'name': 'Adam', 'param': {'lr': 1e-3}}
}

factors_params = {
    'channels': [[1, 20, 20], [], []],
    'kernel_conv': [[2, 2], [], []],
    'kernel_pool': [[1, 1], [], []],
    'dim_hidden': [[10, 10], [10, 10], [10, 10]],
    'non_linearity': [F.relu, F.relu, F.relu],
    'covariance': ['fixed', 'fixed', 'fixed'],
    'optimizer': {'name': 'Adam', 'param': {'lr': 1e-3}},
    'dropout': 0.05,
}

auxiliary_params = {
    'channels': [[1, 20, 20], [], []],
    'kernel_conv': [[2, 2], [], []],
    'kernel_pool': [[1, 1], [], []],
    'dim_hidden': [[10, 10], [10, 10], [10, 10]],
    'non_linearity': [F.relu, F.relu, F.relu],
    'covariance': ['fixed', 'fixed', 'fixed'],
    'optimizer': {'name': 'Adam', 'param': {'lr': 1e-3}},
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
    'optimizer': {'name': 'Adam', 'param': {'lr': 1e-3}},
    'dropout': 0.05,
}

fit_params = {
    'num_epoch': 13,
    'dim_latent': dim_latent,
    'prior_params': prior_params,
    'factors_params': factors_params,
    'auxiliary_params': auxiliary_params,
    'variational_params': variational_params,
}

rpm = RPM(
    observations=observations,
    observation_locations=observation_locations,
    fit_params=fit_params,
)

rpm.fit(observations)

print(0)

from matplotlib import pyplot as plt

plt.figure()
plt.plot(rpm.loss_tot)
plt.show()
# %%


from save_load import rpm_load, rpm_save



import pickle

true_latent = None
model = rpm


rpm_save(model, './tmp.pickle', device="cpu")

#%%

la = rpm_load(
        './tmp.pickle',
        device="cpu",
        observations=observations,
        observation_locations=observation_locations
)



#%%

import torch
from prior import GPPrior


dim_latent = 3
num_inducing_points = 10

mean0 = torch.rand(dim_latent, num_inducing_points)
scale0 = torch.rand(dim_latent)
lengthscale0 = torch.rand(dim_latent)

scale1 = torch.rand(dim_latent)
lengthscale1 = torch.rand(dim_latent)


prior = GPPrior(
    [mean0, True],
    [scale0, True],
    [scale1, True],
    [lengthscale0, True],
    [lengthscale1, False],
)

inducing_locations = torch.linspace(0, 1, num_inducing_points).unsqueeze(-1)


prior.covariance(inducing_locations, inducing_locations).shape

prior.scale1


