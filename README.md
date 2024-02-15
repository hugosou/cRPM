
# (Continuous) Recognition Parametrised Model (cRPM)

See [Walker\*, Soulat\*, Yu, Sahani\* (2023)](https://arxiv.org/abs/2209.05661) for details.

## Model

**Goal**: Explain structure in multimodal high dimensional observations $\mathcal{X}$ (images, videos, time series, etc.) with a set of simpler unobserved variable $\mathcal{Z}$. 

**Notations**: In the most general case:  

$$ \text{ }$$

- Observation: $J$ time-series measured over $T$ timesteps:

$$\mathcal{X} = \{ \mathsf{x}_{jt} : j = 1\dots J, t=1\dots T \}$$

- Latent: $K$-dimensional variable

$$\mathcal{Z}=\{\mathsf{z}_t:t=1 \dots T\}$$

**Hypothesis**: Given $\mathcal{Z}$, observations are conditionally independent across modality and time. The full joint writes:


$$ {P}_{\theta}(\mathcal{X}, \mathcal{Z}) = {p}_{\theta_z}(\mathcal{Z}) \prod_{j,t} \left( {p}_{0,jt}({x}_{jt}) \frac{{f}_{\theta j}({z}_{t} | {x}_{jt})}{{F}_{\theta j}({z}_{t})} \right) $$



$$ \mathsf{P}_{\theta}(\mathcal{X}, \mathcal{Z}) = \mathsf{p}_{\theta_z}(\mathcal{Z}) \prod_{j,t} \left( \mathsf{p}_{0,jt}(\mathsf{x}_{jt}) \frac{\mathsf{f}_{\theta j}(\mathsf{z}_{t} | \mathsf{x}_{jt})}{\mathsf{F}_{\theta j}(\mathsf{z}_{t})} \right) $$


Where each recognition factor $\mathsf{f}_{\theta j}$ is parametrised by a neural network $\theta_j$ that outputs the natural parameters of a multivariate normal distribution. The (empirical) marginals $\mathsf{p}_{0,jt}$ are defined using a set of observations $\{ \mathcal{X}^{(n)} \}_{n=1}^N$ such that: 

$$\mathsf{p}_{0,jt}(\mathsf{x}_{jt}) = \frac1N \sum_{n=1}^N \delta(\mathsf{x}_{jt} - \mathsf{x}_{jt}^{(n)})$$

where $\delta$ is a diract. It comes that:

$$\mathsf{F}_{\theta j}(\mathsf{z}_{t}) = \frac1N \sum_{n=1}^N \mathsf{f}_{\theta j}(\mathsf{z}_{t} | \mathsf{x}_{jt}^{(n)})$$

Finally, the prior $\mathsf{p}_{\theta z}(\mathcal{Z})$ comprises independent Gaussian Process priors (over time, space, etc) for each latent dimension. Optimization uses a lower bound to the ELBO (Variational Free Energy) hence a variational distribution and a set of auxiliary factors.

## Repository

`recognition_parametrised_model.py` contains the main class to instantiate and fit continuous RPM models.

`flexible_multivariate_normal.py` defines and handles Multivariate Normal Distributions using either natural parameters or mean and variance.

## Demo

We provide Jupyter notebooks to illustrate RP-GPFA:

1) Time Series

- Structured bouncing ball experiment: demonstrates the advantages of bypassing the specification of a generative model. `./demo_textured_bouncing_ball.ipynb`


- Multi-Modal Time series: 3D moving ellipsoid. RP-GPFA combines video and range sensor signal to infer the 2D position of an "agent" of interest. (The moving Ellipsoid dataset and stored demo results can be downloaded [here](https://www.dropbox.com/sh/70yc801n3p64ke1/AAC3irVxD9p119N22J1qvqYYa?dl=0)) `./demo_textured_bouncing_ball.ipynb`

2) Static Images

- MNIST Data: Non-linear embedding of images `./demo_peer_supervision.ipynb`

