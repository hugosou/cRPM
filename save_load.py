import torch
import pickle
from recognition_parametrised_model import RPM

def save_crpm(filename, model, observations=None, true_latent=None):
    """ Helper to save a RP-GPFA model (converts all objects to cpu)"""

    with open(filename, 'wb') as outp:

        # Factors Neural Network
        recognition_factors_cpu = []
        for reci in model.recognition_factors:
            recognition_factors_cpu.append(reci.to("cpu"))

        recognition_variational_cpu = model.recognition_variational.to("cpu")

        # Fit parameters
        dim_latent = model.dim_latent
        fit_params_cpu = model.fit_params

        # Prior
        #log_prior_cpu = model.log_prior.to("cpu")

        # Observations
        observations_cpu = None
        if not (observations is None):
            observations_cpu = []
            for obsi in observations:
                observations_cpu.append(obsi.to("cpu"))

        # True Underlying latent (if known)
        true_latent_cpu = None
        if not (true_latent is None):
            true_latent_cpu = true_latent.to("cpu")

        # Store as Dictionary
        model_save = {'loss_tot': model.loss_tot,
                      'fit_params': fit_params_cpu,
                      'dim_latent': dim_latent,
                      'true_latent': true_latent_cpu,
                      'observations': observations_cpu,
                      'recognition_factors': recognition_factors_cpu,
                      'recognition_variational': recognition_variational_cpu}

        # Save
        pickle.dump(model_save, outp, pickle.HIGHEST_PROTOCOL)


def load_crpm(model_name, observations=None):
    """ Helper to Load a RP-GPFA model """
    with open(model_name, 'rb') as outp:
        model_parameters = pickle.load(outp)

        # Current Loss, fit params, latent and Observations (if provided)
        loss_tot = model_parameters['loss_tot']
        fit_params = model_parameters['fit_params']
        dim_latent = model_parameters['dim_latent']
        true_latent = model_parameters['true_latent']
        observations = model_parameters['observations'] if observations is None else observations

        # Current Parameter Estimates
        prior = None
        recognition_factors = model_parameters['recognition_factors']
        recognition_variational = model_parameters['recognition_variational']

        # Init a new model
        model_loaded = RPM(dim_latent, observations, fit_params=fit_params, prior=prior, loss_tot = loss_tot,
                           recognition_factors=recognition_factors, recognition_variational=recognition_variational)

    return model_loaded, observations, true_latent
