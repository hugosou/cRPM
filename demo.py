import pickle
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from utils_process import plot_loss

import torch
from torchvision import datasets
from torchvision.transforms import ToTensor

from utils_demo import rearrange_mnist

from fast_rpm import RPM
import torch.nn.functional as F

data_folder = './../MNIST'

# Load MNIST
train_data = datasets.MNIST(
    root=data_folder,
    train=True,
    transform=ToTensor(),
    download=True,
)

test_data = datasets.MNIST(
    root=data_folder,
    train=False,
    transform=ToTensor()
)

# Random seeds
#torch.manual_seed(0)

# Number of Conditionally independent Factors
num_factors = 2

# Sub-Sample original dataset
train_length = 60000

# Keep Only some digits (for efficiency)
sub_ids = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
sub_ids = torch.tensor([0, 1, 2, 3, 4])
num_digits = len(sub_ids)

# Rearrange MNIST by grouping num_factors Conditionally independent Observations together
observations, train_images, train_labels = rearrange_mnist(
    train_data.train_data,
    train_data.train_labels,
    num_factors,
    train_length=train_length,
    sub_ids=sub_ids
)

# Rearrange MNIST by grouping num_factors Conditionally independent Observations together
observations_test, test_images, test_labels = rearrange_mnist(
    test_data.test_data,
    test_data.test_labels,
    num_factors,
    train_length=test_data.test_labels.shape[0],
    sub_ids=sub_ids
)
test_labels = test_labels.reshape(observations_test[0].shape[0], num_factors)


num_plot = np.arange(6)
plt.figure(figsize=(len(num_plot)*2, num_factors*2))
for obsi in range(len(num_plot)):
    for facti in range(num_factors):
        plt.subplot(num_factors, len(num_plot), (1+obsi) + facti * len(num_plot))
        plt.imshow(observations[facti][num_plot[obsi], :, :], cmap='gray')
        plt.xticks([])
        plt.yticks([])
        if facti == 0:
            plt.title('Obs. n=' + str(obsi))
        if obsi == 0:
            plt.ylabel('Factor. j=' + str(facti))



# GPUs ?
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data type: float64 / float32
data_type = torch.float64
torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_default_dtype(data_type)

# Training Move to GPU
obs = [(obsi / obsi.max()).to(device).to(data_type) for obsi in observations]

# Testing Move to GPU
obs_test = [(obsi / obsi.max()).to(device).to(data_type) for obsi in observations_test]

factors_params = {
    'channels': [[1, 10, 20], [1, 10, 20]],
    'kernel_conv': [[5, 5], [5, 5]],
    'kernel_pool': [[2, 2], [2, 2]],
    'dim_hidden': [[50], [50]],
    'nonlinearity': [F.relu, F.relu],
    'covariance': ['fixed_diag', 'fixed_diag'],
    'optimizer': lambda params: torch.optim.Adam(params=params, lr=1e-3),
}

auxiliary_params = {
    'optimizer': lambda params: torch.optim.Adam(params=params, lr=1e-3),
    'update_bool': False,
}

prior_params = {
    'num_centroids': 1,
}


fit_params = {
    'num_epoch': 1,
    'batch_size': 1000,
    'auxiliary_update': True,
    'auxiliary_toggle': lambda x: x.epoch > 0,
    'auxiliary_mode': 'constrained_moment_matched', # flexible, constrained_prior, constrained_moment_matched
    'dim_latent': 3,
    'factors_params': factors_params,
    'auxiliary_params': factors_params,
    'prior_params': prior_params,
    'ergodic': False,
    'pct': 0.1
}

rpm = RPM(
    observations=obs,
    fit_params=fit_params,
)


rpm.fit(obs)


rpm.get_posteriors(obs)












rpm.get_posteriors(obs)

import fast_save_load
fast_save_load.rpm_save(rpm, 'tmp.pickle')

aa = fast_save_load.rpm_load('tmp.pickle', observations=obs)

print(9)





