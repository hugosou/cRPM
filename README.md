
# Continuous Recognition Parametrised Model (cRPM)

See below for a brief model description and our [paper](https://arxiv.org/abs/2209.05661) (Walker\*, Soulat\*, Yu, Sahani\* *AISTATS2023*) for details.

## Repository

`recognition_parametrised_model.py` contains the main class to instantiate and fit continuous RPM models.

`flexible_multivariate_normal.py` defines and handles Multivariate Normal Distributions using either natural or standard parameters.

## Demo

We provide Jupyter notebooks to illustrate cRPM capabilities:

1) Time Series

- Structured bouncing ball experiment: demonstrates the advantages of bypassing the specification of a generative model. `./demo_textured_bouncing_ball.ipynb`


- Multi-Modal Time series: 3D moving ellipsoid. cRPM combines video and range sensor signal to infer the 2D position of an "agent" of interest. (The moving Ellipsoid dataset and stored demo results can be downloaded [here](https://drive.google.com/file/d/1WkCVPcERyMEmTMEJKMPddhwmDvcnV-fN/view?usp=drive_link) `./demo_rpgpfa_multimodal.ipynb`

2) Static Images

- MNIST Data: Non-linear embedding of images `./demo_peer_supervision.ipynb

## Model

**Goal**: Explain structure in multimodal high dimensional observations $\mathcal{X}$ (images, videos, time series, etc.) with a set of simpler unobserved variable $\mathcal{Z}$. 

**Notations**: In the most general case:  

$$ \text{ }$$

- Observation: $J$ time-series measured over $T$ timesteps:

$$\mathcal{X} = \\{ \mathsf{x_{jt}} : j = 1\dots J, t=1\dots T \\}$$

- Latent: $K$-dimensional variable

$$\mathcal{Z}=\\{\mathsf{z}_t:t=1 \dots T\\}$$

**Hypothesis**: Given $\mathcal{Z}$, observations are conditionally independent across modality and time. The full joint writes:


$$ \mathsf{P_{\theta}}(\mathcal{X}, \mathcal{Z}) = \mathsf{p_{\theta_z}}(\mathcal{Z}) \prod_{j,t} \left( \mathsf{p_{0,jt}}(\mathsf{x_{jt}}) \frac{\mathsf{f_{\theta j}}(\mathsf{z_{t}} | \mathsf{x_{jt}})}{\mathsf{F_{\theta j}}(\mathsf{z}_{t})} \right) $$


Where each recognition factor $\mathsf{f_{\theta j}}$ is parametrised by a neural network $\theta_j$ that outputs the natural parameters of a multivariate normal distribution. The (empirical) marginals $\mathsf{p_{0,jt}}$ are defined using a set of observations $\\{ \mathcal{X}^{(n)} \\}_{n=1}^N$ such that: 

$$\mathsf{p_{0,jt}}(\mathsf{x_{jt}}) = \frac1N \sum_{n=1}^N \delta(\mathsf{x_{jt}} - \mathsf{x_{jt}}^{(n)})$$

where $\delta$ is a diract. It comes that:

$$\mathsf{F_{\theta j}}(\mathsf{z_{t}}) = \frac1N \sum_{n=1}^N \mathsf{f_{\theta j}}(\mathsf{z_{t}} | \mathsf{x_{jt}}^{(n)})$$

Finally, the prior $\mathsf{p_{\theta z}}(\mathcal{Z})$ comprises independent Gaussian Process priors (over time, space, etc) for each latent dimension. Optimization uses a lower bound to the ELBO (Variational Free Energy) hence a variational distribution and a set of auxiliary factors.
