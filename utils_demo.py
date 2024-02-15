# Imports
import torch
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal
from scipy.stats import norm, bernoulli



def rearrange_mnist(train_images, train_labels, num_factors,
                    train_length=60000, sub_ids=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])):
    # Rearange MNIST dataset by grouping num_factors images of with identical labels together

    # Keep Only some digits
    num_digits = len(sub_ids)
    sub_samples_1 = torch.isin(train_labels, sub_ids)
    train_images = train_images[sub_samples_1]
    train_labels = train_labels[sub_samples_1]

    # Sub-Sample and shuffle original dataset
    perm = torch.randperm(len(train_images))
    train_images = train_images[perm[:train_length]]
    train_labels = train_labels[perm[:train_length]]

    # Dimension of each image
    image_size = train_images.shape[-1]

    # Minimum digit occurrence
    num_reps = torch.min(torch.sum(sub_ids.unsqueeze(dim=0) == train_labels.unsqueeze(dim=1), dim=0))
    num_reps = int((np.floor(num_reps / num_factors) * num_factors).squeeze().numpy())

    # Rearranged Datasets: num_reps x num_digits x image_size x image_size
    train_images_factors = torch.zeros((num_reps, num_digits, image_size, image_size))
    train_labels_factors = torch.zeros(num_reps, num_digits)
    for ii in range(len(sub_ids)):
        kept_images = (train_labels == sub_ids[ii])
        train_images_factors[:, ii, :, :] = train_images[kept_images.nonzero()[:num_reps]].squeeze()
        train_labels_factors[:, ii] = train_labels[kept_images.nonzero()[:num_reps]].squeeze()

    # Number of observation per digits
    num_obs_tmp = int(num_reps / num_factors)

    # Rearrange Datasets: num_obs_tmp x num_factors x num_digits x image_size x image_size
    train_images_factors.resize_(num_obs_tmp, num_factors, num_digits, image_size, image_size)
    train_labels_factors.resize_(num_obs_tmp, num_factors, num_digits)

    # Rearrange Datasets: num_obs x num_factors x image_size x image_size
    num_obs = num_obs_tmp * num_digits
    train_images_factors = torch.permute(train_images_factors, (0, 2, 1, 3, 4))
    train_labels_factors = torch.permute(train_labels_factors, (0, 2, 1))

    # train_images_factors.resize_(num_obs, num_factors, image_size, image_size)

    train_images_factors = reshape_fortran(train_images_factors, (num_obs, num_factors, image_size, image_size))
    train_labels_factors = reshape_fortran(train_labels_factors, (num_obs, num_factors))
    train_labels_factors = train_labels_factors[:, 0]

    # Use another Permutation to mix digits
    perm2 = torch.randperm(num_obs)
    train_images_factors = train_images_factors[perm2]
    train_labels_factors = train_labels_factors[perm2]

    observations = [train_images_factors[:, ii] for ii in range(num_factors)]

    # Reshape Training Labels
    train_images_new = train_images_factors.reshape(num_obs * num_factors, image_size, image_size)
    train_labels_new = (train_labels_factors.unsqueeze(dim=1).repeat(1, num_factors)).reshape(
        num_obs * num_factors)

    return observations, train_images_new, train_labels_new

def reshape_fortran(x, shape):
    # Fortran/ Matlab like tensor  reshaping
    if len(x.shape) > 0:
        x = x.permute(*reversed(range(len(x.shape))))
    return x.reshape(*reversed(shape)).permute(*reversed(range(len(shape))))


def generate_lorenz(num_cond, num_steps, dt=0.01, init_simulation = [2.3274,  3.8649, 18.2295], vari_simulation=0.5):
    xyzs = np.empty((num_cond, num_steps + 1, 3))  # Need one more for the initial values

    for n in range(num_cond):
        xyzs[n, 0] = 1*np.random.randn(3)  # Set initial values

        xyzs[n, 0] = 1 * np.array(init_simulation) + vari_simulation * np.random.randn(3)

        for i in range(num_steps):
            xyzs[n, i + 1] = xyzs[n, i] + lorenzz(xyzs[n, i]) * dt

    return xyzs


def lorenzz(xyz, *, s=10, r=28, b=2.667):

    x, y, z = xyz
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z

    return np.array([x_dot, y_dot, z_dot])


def generate_2D_latent(T, F, omega, z0, noise=0.0):
    # Generate a 2D oscillation

    # Number of Time point
    L = T * F

    # Number of trajectories
    N = z0.shape[0]

    # Time Vector
    t = np.arange(L) / F

    # Rotation angle
    Omega = torch.tensor([2*np.pi * omega / F])

    # Rotation Matrix
    rotation = torch.tensor(
        [[torch.cos(Omega), -torch.sin(Omega)],
         [torch.sin(Omega), torch.cos(Omega)]])
    zt = torch.zeros(N, L + 1, 2)

    noise_mvn = MultivariateNormal(torch.zeros(2),
                                   (noise+1e-20) * torch.eye(2))

    # Loop over init
    for n in range(N):

        # Init
        zt[n, 0] = z0[n]

        # Loop Over time point
        for tt in range(L):
            zc = zt[n, tt]
            zz = torch.matmul(rotation, zc)

            if noise>0:
                zz += 0*noise_mvn.sample()

            zt[n, tt+1] = zz

    return zt, t


def generate_skewed_pixel_from_latent(true_latent, dim_observation, scale_th=0.15, sigma2=0.01, shape_max_0=1000):

    # Max and min value of the latent
    latent_max = true_latent.max()
    latent_min = true_latent.min()

    # Build Observation from 1st latent
    pixel_loc = torch.linspace(latent_min, latent_max, dim_observation).unsqueeze(0).unsqueeze(0)

    # Distance Pixel - Ball
    distance_pixel_ball = (torch.exp(-(pixel_loc - true_latent) ** 2 / scale_th ** 2)).numpy()

    # From Rate to shape parameter
    shape_min = np.sqrt(1 - sigma2)
    shape_max = shape_max_0 - shape_min
    shape_parameter = shape_max * distance_pixel_ball + shape_min

    # From shape to samples
    loc0 = shape_parameter
    var0 = np.ones(shape_parameter.shape) * sigma2

    # Bernouilli Parameter
    ber0 = (1 - var0) / (1 - var0 + loc0 ** 2)

    # Mean of the First Peak
    loc1 = loc0

    # Mean of The Second Peak
    loc2 = - loc1 * ber0 / (1 - ber0)

    # Bernouilli Sample
    pp = bernoulli.rvs(ber0)

    # Assign to one distribution
    loc_cur = pp * loc1 + (1 - pp) * loc2

    # Sample feom the distribution
    observation_samples = norm.rvs(loc_cur, np.sqrt(var0))

    return observation_samples


