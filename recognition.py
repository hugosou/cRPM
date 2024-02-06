# Imports
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from flexible_multivariate_normal import vector_to_tril_diag_idx

__all__ = [
    'FullyParametrised',
    'MultiInputNet',
    'Net',
]

"""
    Multiple Recognition Function Architectures for RPM.
    
    goal: map observation(s) (dimension: feature_size) to latent space (dimension: dim_latent) parameters. 
    
    input: observation(s)
        torch.Tensor or List[torch.Tensor] of size [*batch_size, *feature_size] 

    output: [param1, param2]
        concatenated torch.Tensor of size [*batch_size, dim_latent + dim_latent * (dim_latent + 1) / 2]
        param1 and param2 are then used to parameterize flexible multivariate normal distribution
        param1 [*batch_size, dim_latent] is either the mean or 1st natural parameter
        param2 [*batch_size, dim_latent * (dim_latent + 1) / 2] is either the vectorized Cholesky factor 
            of the variance or -2nd natural parameter   
             
"""


class Encoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x):
        raise NotImplementedError()


class MultiInputNet(Encoder):
    def __init__(self,
                 dim_input,
                 dim_latent,
                 covariance='full',
                 kernel_conv=(),
                 kernel_pool=(),
                 channels=(),
                 dim_hidden=(),
                 dim_hidden_merged=(),
                 non_linearity=(),
                 non_linearity_merged=F.relu,
                 dropout=0.0,
                 zero_init=False):

        """
        Multi-Input Neural Network that outputs natural parameters of Flexible Multivariate Normal distributions.

        :param dim_input: List of dimensions of each input
        :param dim_latent: Dimension of the latent space
        :param covariance: Type of covariance matrix ('full', 'diag', 'fixed' or 'fixed_diag')
        :param kernel_conv: List of kernel sizes for convolutional layers
        :param kernel_pool: List of kernel sizes for pooling layers
        :param channels: List of number of channels for convolutional layers
        :param dim_hidden: List of dimensions of each hidden fully connected layer
        :param dim_hidden_merged: List of dimensions of each hidden fully connected layer after merging inputs
        :param non_linearity: Non linearity function
        :param dropout: Dropout proportion
        """

        super(MultiInputNet, self).__init__()

        self.zero_init = zero_init

        # Convolutional layers
        self.kernel_conv = kernel_conv
        self.kernel_pool = kernel_pool
        self.channels = channels

        # Fully connected layers
        self.dim_hidden = dim_hidden
        self.dim_hidden_merged = dim_hidden_merged
        self.non_linearity = non_linearity
        self.non_linearity_merged = non_linearity_merged

        # Define proportion or neurons to dropout
        self.dropout = nn.Dropout(dropout)

        # Indices for precision matrix
        self.idx_diag = dim_latent + vector_to_tril_diag_idx(dim_latent)

        # Input/Output dimensions
        self.dim_input = dim_input
        self.mlp_input = dim_input
        self.dim_latent = dim_latent
        self.covariance = covariance
        self.dim_output, self.dim_output_sufficient = get_dim_output(covariance, dim_latent)

        # Number of input
        self.num_input = len(self.dim_input)

        # Proper Parameter size
        self.check_sizes()

        # Initialize precision matrix if necessary
        self.param2 = init_param2(covariance, dim_latent, zero_init)

        # Init layers
        self.layers = nn.ModuleList()
        self.mlp_input = []
        self.mlp_merged_input = []

        # Init layers private to each inputs
        for ii in range(self.num_input):
            layers_ii = nn.ModuleList()
            mlp_input_ii = append_cnn(
                layers_ii,
                self.dim_input[ii],
                self.kernel_conv[ii],
                self.kernel_pool[ii],
                self.channels[ii]
            )
            append_mlp(layers_ii, mlp_input_ii, self.dim_hidden[ii])
            mlp_input_ii_merged = self.dim_hidden[ii][-1] if len(self.dim_hidden[ii]) > 0 else mlp_input_ii
            self.mlp_input.append(mlp_input_ii)
            self.mlp_merged_input.append(mlp_input_ii_merged)
            self.layers.append(layers_ii)

        # Init layers shared by all inputs
        mlp_input_merged = np.sum(self.mlp_merged_input)
        layers_merged = nn.ModuleList()
        append_mlp(layers_merged, mlp_input_merged, self.dim_hidden_merged,
                   dim_output=self.dim_output_sufficient, zero_init=self.zero_init)
        self.layers.append(layers_merged)

    def forward(self, x):

        # Init pre-merged input
        y = []

        # Process each input
        for ii in range(self.num_input):

            xi = x[ii]
            layers_ii = self.layers[ii]
            mlp_input_ii = self.mlp_input[ii]
            kernel_conv_ii = self.kernel_conv[ii]
            kernel_pool_ii = self.kernel_pool[ii]

            # Convolutional layers
            if len(kernel_conv_ii) > 0:

                # Handle multi-batches and assume 1 input channel
                batch_dim = xi.shape[:-2]
                xi = xi.reshape(-1, 1, *self.dim_input[ii])

                for cl in range(len(kernel_conv_ii)):
                    xi = self.non_linearity[ii](F.max_pool2d(layers_ii[cl](xi), kernel_pool_ii[cl]))

                xi = xi.reshape(*batch_dim, mlp_input_ii)

            # Feedforward layers
            if len(layers_ii[len(kernel_conv_ii):]) > 0:
                for layer in layers_ii[len(kernel_conv_ii):-1]:
                    xi = layer(xi)
                    xi = self.dropout(xi)
                    xi = self.non_linearity[ii](xi)
                xi = layers_ii[-1](xi)

            # Append pre-processed input
            y.append(xi)

        # Concatenate pre-processed modalities
        y = torch.cat(y, dim=-1)

        # Merge Feedforward layers
        layers_merged = self.layers[self.num_input]
        for layer in layers_merged[:-1]:
            y = layer(y)
            y = self.dropout(y)
            y = self.non_linearity_merged(y)
        y = layers_merged[-1](y)

        # Reorganize sufficient statistics depending on covariance mode
        z = reorganize_sufficient_statistics(
            y,
            self.dim_latent,
            self.covariance,
            self.idx_diag,
            self.dim_output,
            self.param2
        )

        return z

    def check_sizes(self):
        """
        Check that all sizes of subnetworks are consistent
        """

        num_input = self.num_input

        # Assert that all subnetwork are provided
        assert len(self.dim_hidden) == num_input
        assert len(self.kernel_conv) == num_input
        assert len(self.kernel_pool) == num_input
        assert len(self.channels) == num_input

        for ii in range(num_input):
            num_cnn_layer = len(self.kernel_conv[ii])
            if num_cnn_layer > 0:
                assert num_cnn_layer == len(self.kernel_pool[ii])
                assert num_cnn_layer == len(self.channels[ii]) - 1


class Net(Encoder):

    def __init__(self,
                 dim_input,
                 dim_latent,
                 covariance='full',
                 kernel_conv=(),
                 kernel_pool=(),
                 channels=(),
                 dim_hidden=(),
                 non_linearity=F.relu,
                 dropout=0.0,
                 zero_init=False,
                 ):

        """
        Neural Network that outputs natural parameters of Flexible Multivariate Normal distributions.

        :param dim_input: List of dimensions of each input
        :param dim_latent: Dimension of the latent space
        :param covariance: Type of covariance matrix (full, diag, fixed, fixed_diag)
        :param kernel_conv: Kernel sizes for convolutional layers
        :param kernel_pool: Kernel sizes for pooling layers
        :param channels: Number of channels for convolutional layers
        :param dim_hidden: Dimensions of each hidden fully connected layer
        :param non_linearity: Non linearity function
        :param dropout: Dropout proportion
        :param zero_init: Initialize last layer to output zeros
        """

        super(Net, self).__init__()

        self.zero_init = zero_init

        # Convolutional layers
        self.kernel_conv = kernel_conv
        self.kernel_pool = kernel_pool
        self.channels = channels

        # Fully connected layers
        self.dim_hidden = dim_hidden
        self.non_linearity = non_linearity

        # Define proportion or neurons to dropout
        self.dropout = nn.Dropout(dropout)

        # Input/Output dimensions
        self.dim_input = dim_input
        self.mlp_input = dim_input
        self.dim_latent = dim_latent
        self.covariance = covariance
        self.dim_output, self.dim_output_sufficient = get_dim_output(covariance, dim_latent)

        # Indices for precision matrix
        self.idx_diag = dim_latent + vector_to_tril_diag_idx(dim_latent)

        # Initialize precision matrix if necessary
        self.param2 = init_param2(covariance, dim_latent, zero_init)

        # Init layers
        self.layers = nn.ModuleList()
        self.mlp_input = append_cnn(self.layers, self.dim_input, self.kernel_conv, self.kernel_pool, self.channels)
        append_mlp(self.layers, self.mlp_input, self.dim_hidden, self.dim_output_sufficient, zero_init=self.zero_init)

    def forward(self, x):

        # Convolutional layers
        if len(self.kernel_conv) > 0:

            # Handle multi-batches and assume 1 input channel
            batch_dim = x.shape[:-2]
            x = x.reshape(-1, 1, *self.dim_input)

            for cl in range(len(self.kernel_conv)):
                x = self.non_linearity(F.max_pool2d(self.layers[cl](x), self.kernel_pool[cl]))

            x = x.reshape(*batch_dim, self.mlp_input)

        # Feedforward layers
        for layer in self.layers[len(self.kernel_conv):-1]:
            x = layer(x)
            x = self.dropout(x)
            x = self.non_linearity(x)
        x = self.layers[-1](x)

        y = reorganize_sufficient_statistics(
            x,
            self.dim_latent,
            self.covariance,
            self.idx_diag,
            self.dim_output,
            self.param2
        )

        return y


class FullyParametrised(Encoder):

    def __init__(self,
                 dim_latent,
                 batch_shape,
                 covariance='fixed',
                 init=None,
                 zero_init=False,
                 ):

        super(FullyParametrised, self).__init__()

        self.dim_latent = dim_latent
        self.batch_shape = batch_shape
        self.covariance = covariance
        self.zero_init = zero_init

        if init is None:
            self.param1, self.param2 = self.init_all()
        else:
            self.param1, self.param2 = init

    def init_all(self):

        dim_latent = self.dim_latent
        batch_shape = self.batch_shape

        param1_init = torch.zeros(*self.batch_shape, self.dim_latent, requires_grad=False)
        param1 = torch.nn.Parameter(param1_init, requires_grad=True)

        if self.covariance == 'full':
            diag_idx = vector_to_tril_diag_idx(dim_latent)
            param2_init = torch.zeros(*batch_shape, int(dim_latent * (dim_latent + 1) / 2), requires_grad=False)

            if self.zero_init:
                param2_init[..., diag_idx] = 0.0
            else:
                param2_init[..., diag_idx] = 0.5

            param2 = torch.nn.Parameter(param2_init, requires_grad=True)

        elif self.covariance == 'diag':
            if self.zero_init:
                param2_init = 0.0 * torch.ones(*batch_shape, dim_latent, requires_grad=False)
            else:
                param2_init = 0.5 * torch.ones(*batch_shape, dim_latent, requires_grad=False)

            param2 = torch.nn.Parameter(param2_init, requires_grad=True)

        elif self.covariance == 'fixed':
            diag_idx = vector_to_tril_diag_idx(dim_latent)
            param2_init = torch.zeros(int(dim_latent * (dim_latent + 1) / 2), requires_grad=False)

            if self.zero_init:
                param2_init[diag_idx] = 0.0
            else:
                param2_init[diag_idx] = 0.5

            param2 = torch.nn.Parameter(param2_init, requires_grad=True)

        elif self.covariance == 'fixed_diag':

            if self.zero_init:
                param2_init = 0.0 * torch.ones(dim_latent, requires_grad=False)
            else:
                param2_init = 0.5 * torch.ones(dim_latent, requires_grad=False)

            param2 = torch.nn.Parameter(param2_init, requires_grad=True)

        else:
            raise NotImplementedError()

        return param1, param2

    def forward(self, x):

        dim_latent = self.dim_latent
        batch_shape = self.batch_shape
        param1 = self.param1

        if self.covariance == 'full':
            param2 = self.param2

        elif self.covariance == 'diag' or self.covariance == 'fixed_diag':
            diag_idx = vector_to_tril_diag_idx(dim_latent)
            param2 = torch.zeros(
                *batch_shape, int(dim_latent * (dim_latent + 1) / 2),
                device=self.param2.device,
                dtype=self.param2.device
            )
            param2[..., diag_idx] = self.param2

        elif self.covariance == 'fixed':

            param2 = torch.zeros(
                *batch_shape, int(dim_latent * (dim_latent + 1) / 2),
                device=self.param2.device,
                dtype=self.param2.device
            )
            param2[..., :] = self.param2

        else:
            raise NotImplementedError()

        return torch.cat([param1, param2], dim=-1)


def init_param2(covariance, dim_latent, zero_init):
    """
    Initialize fixed param2 matrix if necessary
    :param covariance: Type of covariance matrix (full, diag, fixed, fixed_diag)
    :param dim_latent: Distribution dimension
    :return: param2 matrix
    """
     
    if covariance == 'fixed':
        diag_idx = vector_to_tril_diag_idx(dim_latent)
        param2_init = torch.zeros(int(dim_latent * (dim_latent + 1) / 2), requires_grad=False)
        if zero_init:
            param2_init[diag_idx] = 0.0
        else:
            param2_init[diag_idx] = 0.5  # TODO check sign
        param2 = torch.nn.Parameter(param2_init, requires_grad=True)

    elif covariance == 'fixed_diag':
        if zero_init:
            param2_init = 0.0 * torch.ones(dim_latent, requires_grad=False)  # TODO check sign
        else:
            param2_init = 0.5 * torch.ones(dim_latent, requires_grad=False)  # TODO check sign
        param2 = torch.nn.Parameter(param2_init, requires_grad=True)

    else:
        param2 = None
        
    return param2


def get_dim_output(covariance, dim_latent):
    """
    :param dim_latent: Distribution dimension
    :param covariance: Type of covariance matrix (full, diag, fixed, fixed_diag)
    :return: dimension of the output of the network depending of the covariance mode
    """
    # Full dimension of the sufficient statistics
    dim_output = int(dim_latent * (dim_latent + 3) / 2)

    if covariance == 'full':
        # Network outputs mean and a full covariance Cholesky Decomposition vector
        dim_output_sufficient = dim_output

    elif covariance == 'diag':
        # Network outputs mean and a diagonal Cholesky Decomposition vector
        dim_output_sufficient = 2 * dim_latent

    elif covariance == 'fixed' or covariance == 'fixed_diag':
        # Network outputs mean and fixed covariance
        dim_output_sufficient = dim_latent

    else:
        raise NotImplementedError

    dim_output = dim_output
    dim_output_sufficient = dim_output_sufficient

    return dim_output, dim_output_sufficient


def conv_pool_dim(dim_input, kernel_conv, kernel_pool):
    """
    Convolutional and pooling layer output dimension
    :param dim_input: Input dimension
    :param kernel_conv: Kernel size for convolutional layer
    :param kernel_pool: Kernel size for pooling layer
    :return: Output dimension
    """

    out_conv = dim_input - kernel_conv + 1
    out_pool = out_conv // kernel_pool

    return out_pool


def append_cnn(layers, dim_input, kernel_conv, kernel_pool, channels):
    """
    Append convolutional Layers with a pooling layer
    :param layers: List of layers
    :param dim_input: Input dimension
    :param kernel_conv: Kernel sizes for convolutional layers
    :param kernel_pool: Kernel sizes for pooling layers
    :param channels: Number of channels for convolutional layers
    :return: Output dimension
    """

    # Use and append convolutional and pooling layer
    if len(kernel_conv) > 0:

        # Check sizes
        assert (len(kernel_conv) == len(kernel_pool))
        assert (len(kernel_conv) == len(channels) - 1)

        # Init channel number
        channels_ou = channels[0]

        # Init output size
        conv_output_x = dim_input[0]
        conv_output_y = dim_input[1]

        for cl in range(len(kernel_conv)):
            # Current channels
            channels_in = channels_ou
            channels_ou = channels[cl + 1]

            # Current output size
            conv_output_x = conv_pool_dim(conv_output_x, kernel_conv[cl], kernel_pool[cl])
            conv_output_y = conv_pool_dim(conv_output_y, kernel_conv[cl], kernel_pool[cl])

            # Append convolutional layer
            layers.append(nn.Conv2d(channels_in, channels_ou, kernel_size=kernel_conv[cl]))

        # CNN Output: linearized and collapsed across channels
        dim_output = int(channels_ou * conv_output_x * conv_output_y)

    else:
        dim_output = dim_input[0]

    return dim_output


def append_mlp(layers, dim_input, dim_hidden, dim_output=None, zero_init=False):
    """
    Append fully connected layers
    :param layers: List of layers
    :param dim_input: Input dimension
    :param dim_hidden: List of hidden dimensions
    :param dim_output: Output dimension
    :param zero_init: zero out the last layer
    """

    # Use and append fully connected layers
    for i in range(len(dim_hidden) + 1):
        if len(dim_hidden) > 0:
            if i == 0:
                layers.append(nn.Linear(dim_input, dim_hidden[i]))
            elif i == len(dim_hidden) and dim_output is not None:
                layers.append(nn.Linear(dim_hidden[i - 1], dim_output))
            elif i < len(dim_hidden):
                layers.append(nn.Linear(dim_hidden[i - 1], dim_hidden[i]))
        elif dim_output is not None:
            layers.append(nn.Linear(dim_input, dim_output))

        if zero_init:
            torch.nn.init.constant_(layers[-1].weight, 0)
            torch.nn.init.constant_(layers[-1].bias, 0)

def reorganize_sufficient_statistics(y, dim_latent, covariance, idx_diag, dim_output, param2):
    """
    Reorganize the output of the network into the sufficient statistics of the distribution
    :param y: input
    :param dim_latent: Distribution dimension
    :param covariance: Type of covariance matrix (full, diag, fixed, fixed_diag)
    :param idx_diag: Indices of the diagonal elements of the Cholesky Decomposition vector
    :param dim_output: dimension of the output of the network
    :param param2: fixed precision matrix if necessary
    :return: sufficient statistics
    """

    if covariance == 'full':
        z = y
    else:
        # Init output
        z = torch.zeros(y.shape[:-1] + torch.Size([dim_output]), dtype=y.dtype, device=y.device)

        # First Natural parameter
        z[..., :dim_latent] = y[..., :dim_latent]

        # Cholesky decomposition of the second natural parameter
        if covariance == 'diag':
            z[..., idx_diag] = y[..., dim_latent:]

        elif covariance == 'fixed':
            z[..., dim_latent:] = param2

        elif covariance == 'fixed_diag':
            z[..., idx_diag] = param2

        else:
            raise NotImplementedError

    return z
