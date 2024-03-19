import torch

import copy
import pickle

from uuid import uuid4
from datetime import date
from typing import Union, List, Tuple

from fast_rpm import RPM


def unique_token():
    return  str(date.today()) + '-' + str(uuid4())


def rpm_load_param(model_name: str):
    with open(model_name, 'rb') as outp:
        rpm = pickle.load(outp)
    return rpm['fit_params']

def rpm_save(
        model: RPM,
        filename: str,
        device: str = "cpu",
        true_latent: torch.Tensor = None,
        observations: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]] = None,
):
    """ Helper to save a RP-GPFA model (converts all objects to cpu)"""

    with open(filename, 'wb') as outp:
        # Store as Dictionary
        model_save = _dictionarize(
            model,
            device=device,
            true_latent=true_latent,
            observations=observations,
        )
        
        model_save = remove_lambda(model_save)

        # Save
        pickle.dump(model_save, outp, pickle.HIGHEST_PROTOCOL)


def rpm_load(
        model_name: str,
        device: str = "cpu",
        observations: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]] = None,
        true_latent=None,
) -> (RPM, List[torch.Tensor]):

    with open(model_name, 'rb') as outp:
        loaded_dict = _dictionarize(pickle.load(outp), device=device)

    observations = loaded_dict['observations'] \
        if observations is None else observations
    observations = loaded_dict['observations'] \
        if observations is None else observations
    true_latent = loaded_dict['true_latent'] \
        if true_latent is None else true_latent

    loaded_rpm = RPM(
        observations,
        fit_params=loaded_dict['fit_params'],
        loss_tot=loaded_dict['loss_tot'],
        recognition_factors=loaded_dict['recognition_factors'],
        precision_chol_vec_factors=loaded_dict['precision_chol_vec_factors'],
    )

    return loaded_rpm, observations, true_latent


def _to_device(x, device):
    """Recursive move to device"""
    if isinstance(x, torch.Tensor) or isinstance(x, torch.nn.Module):
        y = x.to(device)
    elif x is None:
        y = None
    elif isinstance(x, float) or isinstance(x, dict):
        y = x
    elif isinstance(x, list) or isinstance(x, tuple):
        y = [_to_device(xi, device) for xi in x]
    else:
        raise NotImplementedError()
    return y


def _dictionarize(
        model,
        true_latent=None,
        observations=None,
        device="cpu",
):
    """ Take RPM or "dictionarized" model and move it as a dictionary to device"""

    if isinstance(model, RPM):
        
        dict_model = {
            'fit_params': model.fit_params,
            'recognition_factors': model.recognition_factors,
            'precision_chol_vec_factors': model.precision_chol_vec_factors,
            'observations': observations,
            'true_latent': true_latent,
            'loss_tot': model.loss_tot
        }

    elif isinstance(model, dict):
        dict_model = model

    else:
        raise NotImplementedError()

    dict_model = {
        key: _to_device(dict_model[key], device) for key in dict_model.keys()
    }

    return dict_model

def check_lambda(func):
    if callable(func):
        try:
            return func.__name__ == "<lambda>"
        except AttributeError:
            return False
        
    return callable(func) and func.__name__ == "<lambda>"


def remove_lambda(old_dict: dict):
    """Remove unpickable Lambda functions"""

    new_dict = copy.deepcopy(old_dict)

    for key in old_dict.keys():
        if type(old_dict[key]) is dict:
            new_dict[key] = remove_lambda(new_dict[key])
        elif check_lambda(old_dict[key]):
            del new_dict[key]
        else:
            pass
    return new_dict
