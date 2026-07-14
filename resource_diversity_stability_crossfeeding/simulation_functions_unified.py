# -*- coding: utf-8 -*-
"""
Unified, generalised simulation/analysis utilities for the consumer-resource
models in this package.

This merges "resource_diversity_stability(sl)/simulation_functions.py" and
"external_resource_stability/simulation_functions_new.py", which had
diverged into two nearly-identical, ~90%-duplicated copies (the "_new" copy
also happened to fix a handful of bugs the older copy still has - see the
notes throughout). Neither of those two files is edited by this module;
this is a new, from-scratch replacement designed to be a drop-in substitute
for either (every function that either file's callers actually import -
pickle_dump, CRM_across_parameter_space, save_models, load_in_communities,
generate_simulation_df, simulation_df_from_communities, community_dynamics_df,
prop_chaotic, le_pivot, le_pivot_r, agg_pivot, generic_heatmaps,
generic_heatmaps_multi - keeps the same name and signature/behaviour here).

Compared to the originals, this version:
    - supports every consumer-resource model class in models.py (both
      originals were missing "Self-limiting resource supply, leached" and
      "Metabolic pathways" entirely - calling CRM_across_parameter_space
      with either would raise or silently do the wrong thing)
    - lets every model-specific rate (death, resource growth/influx/outflux,
      self-inhibition, resource interaction, metabolic production, ...) be
      generated as 'normal', 'constant', or 'user-supplied', inferred from
      which keys are present in a parameter set (mu_<x>/sigma_<x> -> normal,
      a bare array -> user-supplied, a bare scalar -> constant), rather than
      every model hardcoding 'constant' - see infer_rate_spec()
    - replaces the ~6-way copy-pasted per-model dict construction in
      model_specific_args() with one data-driven implementation, driven by
      the MODEL_RATE_PARAMS registry below - adding a new model to this
      pipeline is now a one-line registry entry rather than a new match-case
      branch duplicated across every function that inspects model-specific
      parameters
    - merges consumer_resource_model_dynamics/complex_ecosystem_model_dynamics,
      which differed only in whether growth_consumption_rates_args was a
      single dict or a list of dicts (both are still exposed as names, for
      compatibility, but now share one implementation)
    - fixes a few pre-existing bugs found while merging: a missing 'method'
      argument in CRM_df's inner load_data_create_df call (this made
      generate_simulation_df raise a TypeError whenever it was actually
      called), a missing 'rho' key in the older file's rho_coupled()
      (would raise "Please supply a value for rho" whenever hit), a
      'sigma_g' typo shadowing 'sigma_c' in the older file's parameter list,
      and model_specific_parameters raising an UnboundLocalError for
      "Hybrid resource supply" (and now also "Self-limiting resource
      supply, leached" and "Metabolic pathways", which just weren't handled
      at all)

@author: jamil
"""

import numpy as np
import pandas as pd
import sys
from copy import deepcopy
import pickle
import seaborn as sns
from matplotlib import pyplot as plt
import os
from tqdm import tqdm
import numpy.typing as npt
from typing import Union, TypedDict, NotRequired, Literal

sys.path.insert(0, 'C:/Users/jamil/Documents/PhD/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models/consumer_resource_modules')
from models import Consumer_Resource_Model
from community_level_properties import max_le

# %%

##################### model-specific parameter registry #####################

# the multi-trophic model is handled by its own code path throughout, since
# its shape (number of trophic levels, one death rate per level) varies -
# every other model is fully described by the registry below
TROPHIC_MODEL = "Self-limiting resource supply, multi-trophic level"

# For every other model: the (kwarg_prefix, lookup_key, model_label) triples
# describing the model-specific-rate arguments accepted by
# Consumer_Resource_Model.model_specific_rates().
#   kwarg_prefix - combined with '_method'/'_args' to build the keyword
#       arguments passed to model_specific_rates()
#   lookup_key - the base name used to find this parameter in a parameter
#       set dict, e.g. 'mu_<lookup_key>'/'sigma_<lookup_key>'/'<lookup_key>'
#   model_label - the parameter label ParametersInterface.other_parameter_methods()
#       actually expects as a dict key for 'constant'/'user-supplied' args
#       (usually the same as lookup_key, except where the model class's
#       internal label doesn't match the friendlier name used in parameter
#       sets, e.g. SL_CRPM/SL_TL_CRM's resource-interaction matrix is 'Aij'
#       internally but referred to as 'A' in parameter sets, matching the
#       'mu_A'/'sigma_A' convention already used for the trophic model)
MODEL_RATE_PARAMS = {
    "Self-limiting resource supply":
        [("death", "d", "d"), ("resource_growth", "b", "b")],
    "Self-limiting resource supply, self-inhibition":
        [("death", "d", "d"), ("resource_growth", "b", "b"), ("si", "si", "si")],
    "Self-limiting resource supply, leached":
        [("death", "d", "d"), ("resource_growth", "b", "b"),
         ("resource_interaction", "A", "Aij")],
    "Externally-supplied resources":
        [("death", "d", "d"), ("influx", "b", "b"), ("outflux", "o", "o")],
    "Hybrid resource supply":
        [("death", "d", "d"), ("influx", "b", "b"), ("outflux", "o", "o"),
         ("resource_inhibition", "a", "a")],
    "Metabolic pathways":
        [("death", "d", "d"), ("outflux", "o", "o"), ("resource_growth", "b", "b"),
         ("resource_inhibition", "A", "A")],
}

# models with a required call beyond growth_consumption_rates()/
# model_specific_rates() - currently just MP_CRM's structured metabolic
# network. Maps model name -> function building that call's kwargs from a
# parameter set (see _metabolic_network_args below).
def _metabolic_network_args(parm_set : dict):

    '''

    Build kwargs for MP_CRM's metabolic_network() call from a parameter set.

    network_method is resolved first (an explicit 'network_method' key if
    given, else inferred from which keys are present: 'p_s' -> 'step',
    'mean_q'/'variance_q' -> 'gamma', neither -> left unset so
    metabolic_network() falls back to its own default of 'gamma' with
    mean=1, variance=1). resource_conversions is then built to match
    whichever network_method was resolved (an explicit 'resource_conversions'
    dict always wins, if supplied).

    '''

    kwargs = {}

    if 'w' in parm_set:

        kwargs['energies'] = np.asarray(parm_set['w'])

    network_method = parm_set.get('network_method')

    if network_method is None:

        if 'p_s' in parm_set:

            network_method = 'step'

        elif 'mean_q' in parm_set or 'variance_q' in parm_set:

            network_method = 'gamma'

    if network_method is not None:

        kwargs['network_method'] = network_method

        if 'resource_conversions' in parm_set:

            kwargs['resource_conversions'] = parm_set['resource_conversions']

        elif network_method == 'step':

            kwargs['resource_conversions'] = {'p_s' : parm_set['p_s']}

        elif network_method == 'gamma':

            kwargs['resource_conversions'] = {'mean' : parm_set.get('mean_q', 1),
                                              'variance' : parm_set.get('variance_q', 1)}

    method, method_args = infer_rate_spec(parm_set, 'p')
    kwargs['production_method'] = method
    kwargs['production_args'] = method_args

    return kwargs

EXTRA_MODEL_CALLS = {
    "Metabolic pathways" : [("metabolic_network", _metabolic_network_args)],
}

# %%

def infer_rate_spec(parm_set : dict,
                    lookup_key : str,
                    model_label : Union[str, None] = None):

    '''

    Work out how a model-specific rate parameter should be generated from a
    parameter-set dict, so that every rate parameter in every model can be
    'normal', 'constant', or 'user-supplied', rather than every model
    hardcoding 'constant'.

    Resolution order:
        1. an explicit '<lookup_key>_method' (+ optional '<lookup_key>_args')
           override in the parameter set, for full manual control
        2. 'mu_<lookup_key>' and 'sigma_<lookup_key>' both present -> 'normal'
        3. '<lookup_key>' present as a list/np.ndarray -> 'user-supplied'
        4. '<lookup_key>' present as a scalar -> 'constant'

    Parameters
    ----------
    parm_set : dict
        A single parameter set.
    lookup_key : str
        Base name used to find this parameter in parm_set.
    model_label : str or None
        The parameter label expected by the model class itself for
        'constant'/'user-supplied' args (defaults to lookup_key).

    Returns
    -------
    method : str
        'normal', 'constant', or 'user-supplied'.
    args : dict
        Arguments for that method, ready to pass as e.g. death_args.

    '''

    if model_label is None:

        model_label = lookup_key

    method_override_key = lookup_key + '_method'

    if method_override_key in parm_set:

        return parm_set[method_override_key], parm_set.get(lookup_key + '_args', {})

    mu_key, sigma_key = 'mu_' + lookup_key, 'sigma_' + lookup_key

    if mu_key in parm_set and sigma_key in parm_set:

        return 'normal', {'mu' : parm_set[mu_key], 'sigma' : parm_set[sigma_key]}

    if lookup_key in parm_set:

        value = parm_set[lookup_key]

        if isinstance(value, (list, np.ndarray)):

            return 'user-supplied', {model_label : np.asarray(value)}

        return 'constant', {model_label : value}

    raise KeyError("Parameter set is missing '" + lookup_key + "' (or 'mu_" + \
                   lookup_key + "'/'sigma_" + lookup_key + \
                   "') needed to generate a value for '" + lookup_key + "'.")

# %%

def model_specific_args(parameter_sets : list[dict], model : str):

    '''

    Build the Consumer_Resource_Model initialisation kwargs, model_specific_rates
    kwargs, and any extra required setup-call kwargs (e.g. MP_CRM's
    metabolic_network) for every parameter set, for any supported model.

    Parameters
    ----------
    parameter_sets : list[dict]
        List of parameter sets.
    model : str
        Model name, as accepted by Consumer_Resource_Model.

    Returns
    -------
    initialisation_list : list[dict]
        Per parameter set, kwargs for Consumer_Resource_Model(**...).
    m_s_rates_args_list : list[dict]
        Per parameter set, kwargs for community.model_specific_rates(**...).
    extra_calls_list : list[dict[str, dict]]
        Per parameter set, {method_name : kwargs} for any additional calls
        required after model_specific_rates() (empty dict if none).

    '''

    if model == TROPHIC_MODEL:

        return _trophic_model_specific_args(parameter_sets)

    if model not in MODEL_RATE_PARAMS:

        raise ValueError("Unsupported model '" + model + "'. Supported models: " + \
                         str(list(MODEL_RATE_PARAMS) + [TROPHIC_MODEL]))

    rate_params = MODEL_RATE_PARAMS[model]

    initialisation_list = [dict(model = model,
                                pool_sizes = [parm_set['M'], parm_set['S']])
                           for parm_set in parameter_sets]

    m_s_rates_args_list = [_rate_args(parm_set, rate_params)
                           for parm_set in parameter_sets]

    extra_calls_builders = EXTRA_MODEL_CALLS.get(model, [])

    extra_calls_list = [{method_name : args_builder(parm_set)
                         for method_name, args_builder in extra_calls_builders}
                        for parm_set in parameter_sets]

    return initialisation_list, m_s_rates_args_list, extra_calls_list

def _rate_args(parm_set : dict, rate_params : list[tuple]):

    args = {}

    for prefix, lookup_key, model_label in rate_params:

        method, method_args = infer_rate_spec(parm_set, lookup_key, model_label)
        args[prefix + '_method'] = method
        args[prefix + '_args'] = method_args

    return args

def _trophic_model_specific_args(parameter_sets : list[dict]):

    initialisation_list = [dict(model = TROPHIC_MODEL,
                                pool_sizes = parm_set['pool_sizes'])
                           for parm_set in parameter_sets]

    def per_level_death(parm_set):

        death_methods, death_args = [], []

        for i in np.arange(2, parm_set['trophic_levels'] + 1):

            level_label = 'd_' + str(i)
            method, method_args = infer_rate_spec(parm_set, level_label, level_label)
            death_methods.append(method)
            death_args.append(method_args)

        return death_methods, death_args

    m_s_rates_args_list = []

    for parm_set in parameter_sets:

        death_methods, death_args = per_level_death(parm_set)
        resource_growth_method, resource_growth_args = infer_rate_spec(parm_set, 'b')
        resource_interaction_method, resource_interaction_args = \
            infer_rate_spec(parm_set, 'A', 'Aij')

        m_s_rates_args_list.append(dict(death_methods = death_methods,
                                        death_args = death_args,
                                        resource_growth_method = resource_growth_method,
                                        resource_growth_args = resource_growth_args,
                                        resource_interaction_method = resource_interaction_method,
                                        resource_interaction_args = resource_interaction_args))

    extra_calls_list = [{} for _ in parameter_sets]

    return initialisation_list, m_s_rates_args_list, extra_calls_list

# %%

def pickle_dump(filename : str, data : any):

    '''

    Pickle data

    Parameters
    ----------
    filename : str
        file directory.
    data : any
        data.

    Returns
    -------
    None.

    '''


    with open(filename, 'wb') as fp:

        pickle.dump(data, fp)

# %%

def CRM_across_parameter_space(parameter_sets : list[dict],
                               subdirectory : str,
                               parms_for_filenames : list[str],
                               model = "Self-limiting resource supply",
                               save_method : Literal['v1', 'v2', 'v3'] = 'v1',
                               **simulation_kwargs : dict[str, any]):

    '''

    Create and simulate communities across parameter space using the
    Consumer_Resource_Model class. Communities are then pickled.

    Parameters
    ----------
    parameter_sets : list[dict]
        List of parameter sets for the Consumer_Resource_Model class.
    subdirectory : str
        Directory to save community data in.
    parms_for_filenames : list[str]
        Parameters used to name each file.
    model : str
        Any model supported by Consumer_Resource_Model - see MODEL_RATE_PARAMS
        and TROPHIC_MODEL above.
    simulation_kwargs : dict[str, any], optional
        Optional arguments for the Consumer_Resource_Model.
        The default is dict(no_communities = 20, t_end = 7000).

    Returns
    -------

    None

    '''

    # create the directory where the communities should be saved (if the directory
    #   doesn't already exist)
    full_directory = "C:/Users/jamil/Documents/PhD/Data/" \
                        + subdirectory

    if not os.path.exists(full_directory):

        os.makedirs(full_directory)

    # make list of filenames
    if 'pool_sizes' in parms_for_filenames:

        names_list = ["_".join([str(np.round(parm_set[key], 4))
                                if key != 'pool_sizes'
                                else str(np.round(parm_set[key][0], 4))
                                for key in parms_for_filenames])
                      for parm_set in parameter_sets]

    else:

        names_list = ["_".join([str(np.round(parm_set[key], 4))
                                for key in parms_for_filenames])
                      for parm_set in parameter_sets]

    # compile all simulation arguments together with default args (if none are given)
    complete_sim_kwargs = dict(no_communities = 20,
                               t_end = 7000) | simulation_kwargs

    # create list of arguments for generating model specific rates (+ any
    # extra required calls, e.g. MP_CRM's metabolic_network)
    initialisation_list, m_s_rates_args_list, extra_calls_list = \
        model_specific_args(parameter_sets, model)

    g_c_rates_args_list = growth_consumption_rates_args(parameter_sets, model)

    # Iterate through the parameter space, creating and simulating community dynamics
    for name, init_class, gc_rates_args, ms_rates_args, extra_calls in \
        tqdm(zip(names_list, initialisation_list, g_c_rates_args_list,
                m_s_rates_args_list, extra_calls_list),
             position = 0, leave = True, total = len(names_list)):

        CRMs_create_and_save(subdirectory,
                             "simulations_" + name,
                             init_class,
                             gc_rates_args,
                             ms_rates_args,
                             save_method,
                             extra_calls = extra_calls,
                             **complete_sim_kwargs)

# %%

def growth_consumption_rates_args(parameter_sets : list[dict], model : str):

    '''

    Build growth_consumption_rates() kwargs for every parameter set. Chooses
    between 'coupled by rho' and 'growth function of consumption' based on
    whether the parameter sets have a 'mu_y'/'mu_y_<i>' key (yield-coupled)
    or not (rho-coupled).

    '''

    def rho_coupled(parameter_sets):

        return [dict(method = 'coupled by rho',
                     mu_c = parm_set['mu_c'],
                     sigma_c = parm_set['sigma_c'],
                     mu_g = parm_set['mu_g'],
                     sigma_g = parm_set['sigma_g'],
                     rho = parm_set['rho'])
                for parm_set in parameter_sets]

    def rue_coupled(parameter_sets):

        return [dict(method = 'growth function of consumption',
                     mu_c = parm_set['mu_c'],
                     sigma_c = parm_set['sigma_c'],
                     mu_g = parm_set['mu_y'],
                     sigma_g = parm_set['sigma_y'])
                for parm_set in parameter_sets]

    def rho_coupled_tl(parameter_sets, trophic_levels):

        return [[dict(method = 'coupled by rho',
                      trophic_level = i,
                      mu_c = parm_set[f'mu_c_{i}'],
                      sigma_c = parm_set[f'sigma_c_{i}'],
                      mu_g = parm_set[f'mu_g_{i}'],
                      sigma_g = parm_set[f'sigma_g_{i}'],
                      rho = parm_set[f'rho_{i}'])
                 for i in np.arange(2, trophic_levels + 1)]
                for parm_set in parameter_sets]

    def rue_coupled_tl(parameter_sets, trophic_levels):

        return [[dict(method = 'growth function of consumption',
                      trophic_level = i,
                      mu_c = parm_set[f'mu_c_{i}'],
                      sigma_c = parm_set[f'sigma_c_{i}'],
                      mu_g = parm_set[f'mu_y_{i}'],
                      sigma_g = parm_set[f'sigma_y_{i}'])
                 for i in np.arange(2, trophic_levels + 1)]
                for parm_set in parameter_sets]

    if model == TROPHIC_MODEL:

        trophic_levels = parameter_sets[0]['trophic_levels']

        if 'mu_y_2' in parameter_sets[0]:

            return rue_coupled_tl(parameter_sets, trophic_levels)

        return rho_coupled_tl(parameter_sets, trophic_levels)

    if 'mu_y' in parameter_sets[0]:

        return rue_coupled(parameter_sets)

    return rho_coupled(parameter_sets)

#%%

def CRMs_create_and_save(subdirectory : str,
                         filename : str,
                         init_class : dict,
                         growth_consumption_rates_args : Union[list,
                                                               TypedDict('gc_args',
                                                                   {'method' : str,
                                                                    'mu_c' : float,
                                                                    'sigma_c' : float,
                                                                    'mu_g' : float,
                                                                    'sigma_g' : float,
                                                                    'conserve_mass' : NotRequired[bool],
                                                                    'kwargs' : NotRequired[any]})],
                         model_specific_rates_args : dict[str, any],
                         save_method : Literal['v1', 'v2', 'v3'],
                         extra_calls : Union[dict[str, dict], None] = None,
                         **kwargs : any):

    '''

    Simulate communites where model parameters are sampled from the same distributions,
    then save the data

    Parameters
    ----------
    filepath : str
        filepath to save community data.
    init_class : dict
        kwargs for Consumer_Resource_Model.
    growth_consumption_rates_args : TypedDict('gc_args', {'method' : str,
                                                          'mu_c' : float,
                                                          'sigma_c' : float,
                                                          'mu_g' : float,
                                                          'sigma_g' : float,
                                                          'conserve_mass' : NotRequired[bool],
                                                          'kwargs' : NotRequired[any]}) or list of these
        Arguments used to generate growth and consumption rates.
    model_specific_rates_args : Dict(str, any)
        Arguments used to generate model-specific rates.
    save_method : str
        'v1' - pickle the raw community objects.
        'v2' - pickle a dataframe of community attributes (see save_models).
        'v3' - save a csv of community parameters/properties.
    extra_calls : dict[str, dict], optional
        {method_name : kwargs} for any extra calls required after
        model_specific_rates() (e.g. MP_CRM's metabolic_network).
    **kwargs : any
        Optional arguments (no_communities, no_init_conds, t_end).

    Returns
    -------
    None.

    '''

    communities = consumer_resource_model_dynamics(init_class,
                                                   growth_consumption_rates_args,
                                                   model_specific_rates_args,
                                                   extra_calls = extra_calls,
                                                   **kwargs)

    match save_method:

        case 'v1':

            pickle_dump("C:/Users/jamil/Documents/PhD/Data/" + \
                         subdirectory + "/" + filename + ".pkl",
                         communities)

        case 'v2':

            save_models(communities,
                        "C:/Users/jamil/Documents/PhD/Data/" + subdirectory,
                        filename)

        case 'v3':

            gc_method = growth_consumption_rates_args[0]['method'] \
                if isinstance(growth_consumption_rates_args, list) \
                else growth_consumption_rates_args['method']

            df = simulation_df_from_communities(communities,
                                                init_class['model'],
                                                gc_method)

            df.to_csv("C:/Users/jamil/Documents/PhD/Data/" + \
                         subdirectory + "/" + filename + ".csv")

    del communities


# %%

def consumer_resource_model_dynamics(init_class : dict,
                                     growth_consumption_rates_args : Union[list,
                                                                           TypedDict('gc_args',
                                                                               {'method' : str,
                                                                                'mu_c' : float,
                                                                                'sigma_c' : float,
                                                                                'mu_g' : float,
                                                                                'sigma_g' : float,
                                                                                'conserve_mass' : NotRequired[bool],
                                                                                'kwargs' : NotRequired[any]})],
                                     model_specific_rates_args : dict[str, any],
                                     no_communities : int = 5,
                                     no_init_conds : int = 2,
                                     t_end : float = 3500,
                                     extra_calls : Union[dict[str, dict], None] = None):

    '''

    Simulate communites where model parameters are sampled from the same
    distributions. Works for every model - growth_consumption_rates_args can
    be a single dict (one growth_consumption_rates() call) or a list of dicts
    (one call per trophic level, for the multi-trophic model); extra_calls
    covers any required calls beyond growth_consumption_rates()/
    model_specific_rates() (e.g. MP_CRM's metabolic_network).

    Parameters
    ----------
    init_class : dict
        kwargs for Consumer_Resource_Model.
    growth_consumption_rates_args : dict or list[dict]
        Arguments used to generate growth and consumption rates.
    model_specific_rates_args : Dict(str, any)
        Arguments used to generate model-specific rates.
    no_communities : int, optional
        Number of communities to create. The default is 5.
    no_init_conds : int, optional
        Number of initial abundances dynamics are simulated from. The default is 2.
    t_end : float, optional
        Simulation end time. The default is 3500.
    extra_calls : dict[str, dict], optional
        {method_name : kwargs} for any extra calls required after
        model_specific_rates().

    Returns
    -------
    communities : list
        List of simulated Consumer_Resource_Model instances.

    '''

    def community_dynamics():

        # initialise consumer-resource model class
        community = Consumer_Resource_Model(**init_class)

        # generate growth/consumption rates - one call, or one per trophic level
        if isinstance(growth_consumption_rates_args, list):

            for gc_rates_args in growth_consumption_rates_args:

                community.growth_consumption_rates(**gc_rates_args)

        else:

            community.growth_consumption_rates(**growth_consumption_rates_args)

        community.model_specific_rates(**model_specific_rates_args)

        # any extra required setup, e.g. MP_CRM's metabolic_network
        for method_name, method_kwargs in (extra_calls or {}).items():

            getattr(community, method_name)(**method_kwargs)

        # simulate commmunity dynamics
        community.simulate_community(t_end, no_init_conds)

        # estimate community properties, including the max. lyapunov exponent
        community.calculate_community_properties()
        community.lyapunov_exponent = max_le(community, community.ODE_sols[0].y[:, -1],
                                             T = 1000, perturbation = 1e-6)

        return community

    # generate n communities, where n = no_communities
    communities = [deepcopy(community_dynamics()) for _ in range(no_communities)]

    return communities

# kept as a separate name for compatibility with old callers - the trophic
# and non-trophic cases are now handled by one shared implementation above
complex_ecosystem_model_dynamics = consumer_resource_model_dynamics

# %%

def save_models(communities : list,
                directory : str,
                filename: str):

    def model_attr_to_dict(community):

        attributes = list(community.__dict__.keys())

        attributes_dict = {attr : getattr(community, attr)
                           for attr in attributes
                           if attr != "ODE_sols"}

        attributes_dict['simulation_t'] = [rep.t for rep in community.ODE_sols]
        attributes_dict['simulation_y'] = [rep.y.T for rep in community.ODE_sols]

        ####

        all_models = {"SL_CRM" : "Self-limiting resource supply",
                      "SL_CRPM" : "Self-limiting resource supply, leached",
                      "SL_SI_CRM" : "Self-limiting resource supply, self-inhibition",
                      "SL_TL_CRM" : "Self-limiting resource supply, multi-trophic level",
                      "ES_CRM" : "Externally-supplied resources",
                      "Hybrid_CRM" : "Hybrid resource supply",
                      "MP_CRM" : "Metabolic pathways",
                      "eLV" : "eLV"}

        attributes_dict['model'] = all_models[type(community).__name__]

        return attributes_dict

    attr_dicts = [model_attr_to_dict(community) for community in communities]

    attr_df = pd.DataFrame(attr_dicts)

    if not os.path.exists(directory):

        os.makedirs(directory)

    attr_df.to_pickle(directory + "/" + filename + ".bz2",
                      compression='bz2')


# %%

def load_in_communities(filepath : str):

    def dict_to_model(community_dict):

        if community_dict['model'].endswith("multi-trophic level"):

            community = Consumer_Resource_Model(community_dict['model'],
                                                community_dict['pool_sizes'])

        else:

            community = Consumer_Resource_Model(community_dict['model'],
                                                pool_sizes = [community_dict['no_resources'],
                                                              community_dict['no_species']])

        for attr, val in community_dict.items():

            if attr not in ['model', 'no_species', 'no_resources', 'pool_sizes',
                            'simulation_t', 'simulation_y']:

                setattr(community, attr, val)

        community.ODE_sols = [ReloadedODEs(t, y.T)
                              for t, y in zip(community_dict['simulation_t'],
                                              community_dict['simulation_y'])]

        return community

    attr_df = pd.read_pickle(filepath)

    if isinstance(attr_df, list):

        return attr_df

    else:

        attr_dicts = attr_df.to_dict("records")

        communities = [dict_to_model(attr_dict) for attr_dict in attr_dicts]

        return communities

class ReloadedODEs:

    def __init__(self, t : npt.NDArray, y : npt.NDArray):

        self.t = t
        self.y = y

# %%

def generate_simulation_df(directory : str):

    '''

    Load data from consumer-resource models and create a dataframe containing
    their parameters and community properties.

    Parameters
    ----------
    directory : str
        filepath for community data.

    Returns
    -------
    df : pd.DataFrame
        Dataframe of model parameters and community properties.

    '''

    # parameters to include in the dataframe
    parameters = ['no_species', 'no_resources', 'mu_c', 'sigma_c', 'mu_y',
                  'sigma_y', 'd_val', 'b_val']

    # load data and create dataframe
    df = CRM_df(directory, parameters)

    # rename columns to useful names for our analysis (e.g., taking into account M-scaling)
    df.rename(columns = {'mu_c' : 'mu_c/M', 'sigma_c' : 'sigma_c/root_M',
                         'mu_g' : 'mu_y', 'sigma_g' : 'sigma_y',
                         'no_resources' : 'M', 'no_species' : 'S'},
                        inplace = True)

    # calculate actual mean and std. deviation in consumption coefficients
    df['mu_c'] = df['mu_c/M'] * df['M']
    df['sigma_c'] = df['sigma_c/root_M'] * np.sqrt(df['M'])

    # calculate the correlation between growth and consumption
    df['rho'] = np.sqrt(1 / (1 + \
                             ((df['sigma_y']/df['mu_y'])**2 * (1 + \
                                                               ((df['mu_c']**2)/(df['M'] * df['sigma_c']**2))))))

    # calculate the distance from the stability threshold
    df['Instability distance'] = df['rho']**2 - df['Species packing']

    # calculate the distance from the infeasibility threshold
    df['Infeasibility distance'] = df['phi_R'] - df['phi_N']/(df['M']/df['S'])

    # remove any numerical inaccuracies by rounding the parameters
    for var in ['rho', 'mu_c', 'mu_y', 'sigma_c', 'sigma_y', 'mu_c/M',
                'sigma_c/root_M']:

        df[var] = np.round(df[var], 6)

    # set species and reosurce pool size to the correct type - int
    df['M'] = np.int32(df['M'])
    df['S'] = np.int32(df['S'])

    return df

# %%

def simulation_df_from_communities(communities, model, gc_method):

    '''

    Load data from consumer-resource models and create a dataframe containing
    their parameters and community properties.

    Parameters
    ----------
    communities : list
        List of Consumer_Resource_Model instances (all the same model type).
    model : str
        Model name.
    gc_method : str
        growth_consumption_rates() method used ('coupled by rho' or
        'growth function of consumption'; for the multi-trophic model, pass
        the method used for the bottom trophic level).

    Returns
    -------
    df : pd.DataFrame
        Dataframe of model parameters and community properties.

    '''

    # parameters to include in the dataframe

    if model == TROPHIC_MODEL:

        parameters = extract_trophic_level_parms(communities[0].trophic_levels) + \
                        model_specific_parameters(model, communities)

    else:

        parameters = extract_growth_consumption_parms() + \
                    model_specific_parameters(model, communities)

    # load data and create dataframe
    df = community_dynamics_df(communities, parameters)

    df = (
            df.pipe(parameter_rename_and_calc,
                    model = model, gc_method = gc_method)
              .pipe(model_specific_emergent_properties,
                    model = model)
              .pipe(np.round, 6)
          )

    return df

# %%

def _resolve_existing_attrs(reference_object : any, candidates : list[str]):

    '''

    Return whichever of `candidates` actually exist as attributes on
    `reference_object`, preserving order. Used so the dataframe-generating
    functions below work regardless of whether a model-specific rate was
    generated as 'normal' (producing mu_<x>/sigma_<x> attributes) or
    'constant' (producing <label>_val) - see infer_rate_spec().

    '''

    return [attr for attr in candidates if hasattr(reference_object, attr)]

def _rate_param_attr_candidates(model_label : str):

    '''

    The class-attribute name(s) a model-specific rate parameter could end up
    with, depending on which distribution ParametersInterface.other_parameter_methods()
    used to generate it. Note 'normal' only keys off the label's first
    character (a quirk of other_parameter_methods itself), so e.g. every
    per-trophic-level death rate ('d_2', 'd_3', ...) shares 'mu_d'/'sigma_d'
    if generated as 'normal'.

    '''

    return [model_label + '_val', 'mu_' + model_label[0], 'sigma_' + model_label[0]]

def model_specific_parameters(model : str, communities : list):

    '''

    Work out which model-specific-rate summary attributes actually exist on
    a batch of communities (dynamically, so this works whether those rates
    were generated as 'normal' or 'constant' - see infer_rate_spec()). For
    the multi-trophic model, this covers the bottom level's resource growth
    rate ('b'), the resource-interaction matrix ('Aij'), and every trophic
    level's death rate ('d_2', 'd_3', ...).

    '''

    representative = communities[0]

    if model == TROPHIC_MODEL:

        model_labels = ['b', 'Aij'] + \
            ['d_' + str(i) for i in np.arange(2, representative.trophic_levels + 1)]

    elif model in MODEL_RATE_PARAMS:

        model_labels = [model_label for _, _, model_label in MODEL_RATE_PARAMS[model]]

    else:

        raise ValueError("Unsupported model '" + model + "' for model_specific_parameters. " + \
                         "Supported models: " + str(list(MODEL_RATE_PARAMS) + [TROPHIC_MODEL]))

    candidates = [attr for model_label in model_labels
                 for attr in _rate_param_attr_candidates(model_label)]

    return _resolve_existing_attrs(representative, candidates)

# %%

def extract_trophic_level_parms(trophic_levels : int):

    poolsize_parms = ['pool_sizes']

    growth_consumption_parms = [string + str(i) for string in ['mu_g_', 'sigma_g_',
                                                                'mu_c_', 'sigma_c_',
                                                                'rho_']
                                for i in np.arange(2, trophic_levels + 1)]

    return poolsize_parms + growth_consumption_parms

def extract_growth_consumption_parms():

    poolsize_parms = ['no_resources', 'no_species']
    growth_consumption_parms = ['mu_g', 'sigma_g', 'mu_c', 'sigma_c', 'rho']

    return poolsize_parms + growth_consumption_parms

# %%

def parameter_rename_and_calc(df, model, gc_method):

    def coupled_rho(df, i = None):

        if i == None:

            df.rename(columns = {'mu_c' : 'mu_c/M', 'sigma_c' : 'sigma_c/root_M',
                                 'mu_g' : 'mu_g/M', 'sigma_g' : 'sigma_g/root_M',
                                 'no_resources' : 'M', 'no_species' : 'S'},
                                inplace = True)

            # calculate actual mean and std. deviation in consumption coefficients
            df['mu_c'] = df['mu_c/M'] * df['M']
            df['sigma_c'] = df['sigma_c/root_M'] * np.sqrt(df['M'])
            df['mu_g'] = df['mu_g/M'] * df['M']
            df['sigma_g'] = df['sigma_g/root_M'] * np.sqrt(df['M'])

        else:

            df.rename(columns = {'mu_c_' + str(i) : 'mu_c_' + str(i) + '/PS_'  + str(i),
                                 'sigma_c_' + str(i) : 'sigma_c_' + str(i) + '/root_PS_' + str(i),
                                 'mu_g_' + str(i) : 'mu_g_' + str(i) + '/PS_' + str(i),
                                 'sigma_g_' + str(i) : 'sigma_g_' + str(i) + '/root_PS_' + str(i)},
                      inplace = True)

            # calculate actual mean and std. deviation in consumption coefficients
            df['mu_c_' + str(i)] = df['mu_c_' + str(i) + '/PS_'  + str(i)] * df['PS_' + str(i)]
            df['sigma_c_' + str(i)] = df['sigma_c_' + str(i) + '/root_PS_'  + str(i)] * np.sqrt(df['PS_' + str(i)])
            df['mu_g_' + str(i)] = df['mu_g_' + str(i) + '/PS_'  + str(i)] * df['PS_' + str(i)]
            df['sigma_g_' + str(i)] = df['sigma_g_' + str(i) + '/root_PS_'  + str(i)] * np.sqrt(df['PS_' + str(i)])

        return df

    def coupled_rue(df, i = None):

        if i == None:

            df.rename(columns = {'mu_c' : 'mu_c/M', 'sigma_c' : 'sigma_c/root_M',
                                 'mu_g' : 'mu_y', 'sigma_g' : 'sigma_y',
                                 'no_resources' : 'M', 'no_species' : 'S'},
                                inplace = True)

            # calculate actual mean and std. deviation in consumption coefficients
            df['mu_c'] = df['mu_c/M'] * df['M']
            df['sigma_c'] = df['sigma_c/root_M'] * np.sqrt(df['M'])

            # calculate the correlation between growth and consumption
            df['rho'] = np.sqrt(1 / (1 + \
                                     ((df['sigma_y']/df['mu_y'])**2 * (1 + \
                                                                       ((df['mu_c']**2)/(df['M'] * df['sigma_c']**2))))))

        else:

            df.rename(columns = {'mu_c_' + str(i) : 'mu_c_' + str(i) + '/PS_'  + str(i),
                                 'sigma_c_' + str(i) : 'sigma_c_' + str(i) + '/root_PS_' + str(i),
                                 'mu_g_' + str(i) : 'mu_y_' + str(i),
                                 'sigma_g_' + str(i) : 'sigma_y_' + str(i)},
                      inplace = True)

            # calculate actual mean and std. deviation in consumption coefficients
            df['mu_c_' + str(i)] = df['mu_c_' + str(i) + '/PS_'  + str(i)] * df['PS_' + str(i)]
            df['sigma_c_' + str(i)] = df['sigma_c_' + str(i) + '/root_PS_'  + str(i)] * np.sqrt(df['PS_' + str(i)])

            df['rho_' + str(i)] = np.sqrt(1 / (1 + \
                                     ((df['sigma_y_' + str(i)]/df['mu_y_' + str(i)])**2 * (1 + \
                                                                       ((df['mu_c_' + str(i)]**2)/(df['PS_' + str(i)] * df['sigma_c_' + str(i)]**2))))))

        return df

    if model == TROPHIC_MODEL:

        match gc_method:

            case 'coupled by rho':

                # trophic level 1 is the resource pool - it has no growth/
                # consumption rates of its own, so start at level 2
                for trophic_level in np.arange(2, df.loc[0, 'trophic_levels'] + 1):

                    df = df.pipe(coupled_rho, i = trophic_level)

            case 'growth function of consumption':

                for trophic_level in np.arange(2, df.loc[0, 'trophic_levels'] + 1):

                    df = df.pipe(coupled_rue, i = trophic_level)

        # resource-interaction (Aij) is only 'mu_A'/'sigma_A'-scaled if it was
        # actually generated as 'normal' - if it was 'constant'/'user-supplied'
        # instead, 'Aij_val' (or nothing, for an array) is already the raw value
        if 'mu_A' in df.columns and 'sigma_A' in df.columns:

            df.rename(columns = {'mu_A' : 'mu_A/PS_1',
                                 'sigma_A' : 'sigma_A/root_PS_1'},
                                inplace = True)

            df['mu_A'] = df['mu_A/PS_1'] * df['PS_1']
            df['sigma_A'] = df['sigma_A/root_PS_1'] * np.sqrt(df['PS_1'])


    else:

        match gc_method:

            case 'coupled by rho':

              df = df.pipe(coupled_rho)

            case 'growth function of consumption':

                df = df.pipe(coupled_rue)

    return df

# %%

def model_specific_emergent_properties(df, model):

    if model == TROPHIC_MODEL:

        for i in np.arange(2, df.loc[0, 'trophic_levels'] + 1):

            df['Packing_' + str(i)] = \
                (df['phi_TL' + str(i)]*df['PS_' + str(i)])/(df['phi_TL' + str(i-1)]*df['PS_' + str(i-1)])

        for i in np.arange(2, df.loc[0, 'trophic_levels'] + 1):

            df['PS_' + str(i)] = np.int32(df['PS_' + str(i)])

    else:

        # calculate the species packing ratio
        df['Species packing'] = (df['phi_N']*df['S'])/(df['phi_R']*df['M'])

        df['M'] = np.int32(df['M'])
        df['S'] = np.int32(df['S'])

        if model.startswith("Self-limiting resource supply") is True:

            # calculate the distance from the stability threshold
            df['Instability distance'] = df['rho']**2 - df['Species packing']

            # calculate the distance from the infeasibility threshold
            df['Infeasibility distance'] = df['phi_R'] - df['phi_N']/(df['M']/df['S'])

        elif model.startswith("Externally-supplied resources") is True or \
            model.startswith("Hybrid resource supply") is True:

            # calculate the distance from the infeasibility threshold
            df['Infeasibility distance'] = 0.5 - df['phi_N']/(df['M']/df['S'])

    return df

# %%

def CRM_df(directory : str, parameters : list, method : Literal['v1', 'v2'] = 'v1'):

    '''

    Load consumer resource model data and generate dataframe

    Parameters
    ----------
    directory : str
        filepath for community data.
    parameters : list
        model parameters to include in the dataframe.
    method : str
        Kept for backward compatibility - no longer needed. load_in_communities
        already auto-detects whether a file holds pickled raw community
        objects (save_method 'v1') or a pickled community-attribute dataframe
        (save_method 'v2'/save_models).

    Returns
    -------
    df : pd.DataFrame
        Dataframe of model parameters and community properties.

    '''

    def load_data_create_df(filepath):

        communities = load_in_communities(filepath)

        return community_dynamics_df(communities, parameters)

    df = pd.concat([load_data_create_df(directory + "/" + file)
                    for file in os.listdir(directory)],
                   axis = 0, ignore_index = True)

    return df

# %%

def community_dynamics_df(communities : list,
                          parameters : list):

    '''

    Generate dataframe on some communities' model parameters and properties

    Parameters
    ----------
    communities : list
        List of objects of the consumer_resource_model class.
    parameters : list
        model parameters to include in the dataframe.

    Returns
    -------
    df : pd.DataFrame
        Dataframe of model parameters and community properties.
    '''

    parameters = np.array(parameters)

    if (trophic_levels := getattr(communities[0], "trophic_levels", None)) is None:

        # extract community properties
        properties_df = pd.DataFrame.from_dict({'phi_N' : [phi_N for community in communities for phi_N in community.species_survival_fraction],
                                                'N_mean' : [N_mean for community in communities for N_mean in community.species_avg_abundance],
                                                'q_N' : [q_N for community in communities for q_N in community.species_abundance_fluctuations],
                                                'phi_R' : [phi_R for community in communities for phi_R in community.resource_survival_fraction],
                                                'R_mean' : [R_mean for community in communities for R_mean in community.resource_avg_abundance],
                                                'q_R' : [q_R for community in communities for q_R in community.resource_abundance_fluctuations],
                                                'Max. lyapunov exponent' : np.concatenate([np.repeat(community.lyapunov_exponent, len(community.ODE_sols))
                                                                                           for community in communities]),
                                                'Divergence measure' : [simulation.t[-1] for community in communities for simulation in community.ODE_sols]})

            # extract model parameters
        parameter_df = pd.DataFrame.from_dict({parameter : \
                                               np.concatenate([np.repeat(getattr(community, parameter),
                                                                         len(community.ODE_sols))
                                                               for community in communities])
                                               for parameter in parameters})

        df = pd.concat([parameter_df, properties_df], axis = 1)

    else:

        abundance_distribution = [{'phi_TL' + str(i) : [phi_L for community in communities for phi_L in getattr(community, "TL_" + str(i) + "_survival_fraction")],
                                   'mean_TL' + str(i) : [phi_L for community in communities for phi_L in getattr(community, "TL_" + str(i) + "_avg_abundance")],
                                   'fluct_TL' + str(i) : [phi_L for community in communities for phi_L in getattr(community, "TL_" + str(i) + "_abundance_fluctuations")]}
                                  for i in np.arange(1, trophic_levels + 1)]

        abundance_dict = {key : val for dist_list in abundance_distribution for key, val in dist_list.items()}

        properties_dict = abundance_dict | \
                            {'trophic_levels' : np.repeat(trophic_levels, len(communities[0].ODE_sols) * len(communities)),
                             'Max. lyapunov exponent' : np.concatenate([np.repeat(community.lyapunov_exponent, len(community.ODE_sols))
                                                                        for community in communities]),
                             'Divergence measure' : [simulation.t[-1] for community in communities for simulation in community.ODE_sols]}

        parameter_dict_wps = {parameter : np.concatenate([np.repeat(getattr(community, parameter),
                                                                    len(community.ODE_sols))
                                                          for community in communities])
                              for parameter in parameters if parameter != "pool_sizes"}

        pool_size_dict = {'PS_' + str(i+1) : np.concatenate([np.repeat(community.pool_sizes[i],
                                                                       len(community.ODE_sols))
                                                             for community in communities])
                          for i in np.arange(trophic_levels)}

        parameter_dict = parameter_dict_wps | pool_size_dict

        df = pd.DataFrame.from_dict(parameter_dict | properties_dict)

    return df

# %%

#### pivot table functions ####

def prop_chaotic(x,
                instability_threshold = 0):

    '''

    calculate the proportion of communities with max. lyapunov exponents > 0

    '''

    return 1 - np.count_nonzero(x < instability_threshold)/len(x)

####

def le_pivot(df, index = 'sigma_c', columns = 'mu_c', values = 'Max. lyapunov exponent'):

    '''

    generate pivot table containing the proportion of unstable communities,
    grouped by 2 different model parameters

    '''

    return [pd.pivot_table(df, index = index, columns = columns,
                          values = 'Max. lyapunov exponent', aggfunc = prop_chaotic)]

def le_pivot_r(df, index = 'sigma_c', columns = 'mu_c', values = 'Max. lyapunov exponent'):

    '''

    generate pivot table containing the proportion of stable communities,
    grouped by 2 different model parameters


    '''

    return [1 - pd.pivot_table(df, index = index, columns = columns,
                               values = 'Max. lyapunov exponent', aggfunc = prop_chaotic)]

####################

def agg_pivot(df, values, index = 'sigma_c', columns = 'mu_c', aggfunc = 'mean'):

    '''

    Generate pivot table grouped by any 2 parameters for any aggregation functionS

    '''

    return [pd.pivot_table(df, index = index, columns = columns,
                           values = values, aggfunc = aggfunc)]

# %%

def generic_heatmaps(df, x, y, xlabel, ylabel, variables, cmaps, titles,
                     fig_dims, figsize,
                     pivot_functions = None, is_logged = None, specify_min_max = None,
                     mosaic = None, gridspec_kw = None, **kwargs):

    '''

    Useful function for plotting multiple heatmaps quickly

    '''

    if pivot_functions is None:

        pivot_tables = {variable : df.pivot(index = y, columns = x, values = variable)
                        for variable in variables}

    else:

        pivot_tables = {variable : (df.pivot(index = x, columns = y, values = variable)
                                    if pivot_functions[variable] is None
                                    else
                                    pivot_functions[variable](df, index = y,
                                                              columns = x,
                                                              values = variable)[0])
                        for variable in variables}

    if is_logged is None:

        pivot_tables_plot = pivot_tables

    else:

        pivot_tables_plot = pivot_tables | \
                            {variable : np.log10(np.abs(pivot_tables[variable]))
                             for variable in is_logged}

    start_v_min_max = {variable : [np.min(pivot_table), np.max(pivot_table)]
                       for variable, pivot_table in pivot_tables_plot.items()}

    if specify_min_max:

        v_min_max = start_v_min_max | specify_min_max

    else:

        v_min_max = start_v_min_max

    sns.set_style('white')

    if mosaic:

        fig, axs = plt.subplot_mosaic(mosaic, figsize = figsize,
                                      gridspec_kw = gridspec_kw)

    else:

        fig, axs = plt.subplots(fig_dims[0], fig_dims[1], figsize = figsize,
                                sharex = True, sharey = True, layout = 'constrained')

    fig.supxlabel(xlabel, fontsize = 16, weight = 'bold')
    fig.supylabel(ylabel, fontsize = 16, weight = 'bold', horizontalalignment = 'center',
                  verticalalignment = 'center')

    if fig_dims == (1,1):

        axs.set_facecolor('grey')

        subfig = sns.heatmap(pivot_tables_plot[variables[0]], ax = axs,
                    vmin = v_min_max[variables[0]][0], vmax = v_min_max[variables[0]][1],
                    cbar = True, cmap = cmaps, **kwargs)

        subfig.axhline(0, 0, 1, color = 'black', linewidth = 2)
        subfig.axhline(pivot_tables_plot[variables[0]].shape[0], 0, 1,
                       color = 'black', linewidth = 2)
        subfig.axvline(0, 0, 1, color = 'black', linewidth = 2)
        subfig.axvline(pivot_tables_plot[variables[0]].shape[1], 0, 1,
                       color = 'black', linewidth = 2)

        axs.set_yticks([0.5, len(np.unique(df[y])) - 0.5],
                      labels = [np.round(np.min(df[y]), 3),
                                np.round(np.max(df[y]), 3)], fontsize = 14)
        axs.set_xticks([0.5, len(np.unique(df[x])) - 0.5],
                      labels = [np.round(np.min(df[x]), 3),
                                np.round(np.max(df[x]), 3)],
                      fontsize = 14, rotation = 0)
        axs.set_xlabel('')
        axs.set_ylabel('')
        axs.invert_yaxis()
        axs.set_title(titles, fontsize = 16, weight = 'bold')

    else:

        if mosaic:

            iterator = axs.values()

        else:

            iterator = axs.flatten()

        for ax, variable, cmap, title in zip(iterator, variables, cmaps, titles):

            ax.set_facecolor('grey')

            subfig = sns.heatmap(pivot_tables_plot[variable], ax = ax,
                        vmin = v_min_max[variable][0], vmax = v_min_max[variable][1],
                        cbar = True, cmap = cmap, **kwargs)

            subfig.axhline(0, 0, 1, color = 'black', linewidth = 2)
            subfig.axhline(pivot_tables_plot[variable].shape[0], 0, 1,
                           color = 'black', linewidth = 2)
            subfig.axvline(0, 0, 1, color = 'black', linewidth = 2)
            subfig.axvline(pivot_tables_plot[variable].shape[1], 0, 1,
                           color = 'black', linewidth = 2)

            ax.set_yticks(np.arange(0.5, len(pivot_tables_plot[variable].index.to_numpy()) + 0.5, 2),
                          labels = pivot_tables_plot[variable].index.to_numpy()[::2], fontsize = 8)
            ax.set_xticks(np.arange(0.5, len(pivot_tables_plot[variable].columns.to_numpy()) + 0.5, 2),
                          labels = pivot_tables_plot[variable].columns.to_numpy()[::2],
                          fontsize = 8, rotation = 0)
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.invert_yaxis()
            ax.set_title(title, fontsize = 16, weight = 'bold')

    return fig, axs

# %%

def generic_heatmaps_multi(dfs, xs, ys, xlabels, ylabels, variable, cmap,
                           fig_dims, figsize,
                           pivot_functions = None, is_logged = None, specify_min_max = None,
                           mosaic = None, gridspec_kw = None, cbar_pos = 0, **kwargs):

    if isinstance(xs, str):

        x_iterator = np.repeat(xs, len(dfs))

    else:

        x_iterator = xs

    if pivot_functions is None:

        pivot_tables = [df.pivot(index = y, columns = x, values = variable)
                        for x, y, df in zip(x_iterator, ys, dfs)]

    else:

        pivot_tables = [(df.pivot(index = x, columns = y, values = variable)
                                    if pivot_functions[variable] is None
                                    else
                                    pivot_functions[variable](df, index = y,
                                                              columns = x,
                                                              values = variable)[0])
                        for x, y, df in zip(x_iterator, ys, dfs)]


    if is_logged is None:

        pivot_tables_plot = pivot_tables

    else:

        pivot_tables_plot = [np.log10(np.abs(pivot_table))
                             for pivot_table in pivot_tables]

    if specify_min_max is None:

        v_min_max = [[np.min(pivot_table), np.max(pivot_table)]
                     for pivot_table in pivot_tables_plot]

    else:

        v_min_max = specify_min_max


    def plot_ax(i, ax, pivot_table, v_mm, ylabel, cbar_pos):

        ax.set_facecolor('grey')

        if i == cbar_pos:

            subfig = sns.heatmap(pivot_table, ax = ax,
                        vmin = v_mm[0], vmax = v_mm[1],
                        cbar = True, cmap = cmap, **kwargs)

        else:

            subfig = sns.heatmap(pivot_table, ax = ax,
                        vmin = v_mm[0], vmax = v_mm[1],
                        cbar = False, cmap = cmap, **kwargs)

        subfig.axhline(0, 0, 1, color = 'black', linewidth = 2)
        subfig.axhline(pivot_table.shape[0], 0, 1,
                       color = 'black', linewidth = 2)
        subfig.axvline(0, 0, 1, color = 'black', linewidth = 2)
        subfig.axvline(pivot_table.shape[1], 0, 1,
                       color = 'black', linewidth = 2)

        ax.set_yticks(np.arange(0.5, len(pivot_table.index.to_numpy()) + 0.5, 2),
                      labels = pivot_table.index.to_numpy()[::2], fontsize = 8)
        ax.set_xticks(np.arange(0.5, len(pivot_table.columns.to_numpy()) + 0.5, 2),
                      labels = pivot_table.columns.to_numpy()[::2],
                      fontsize = 8, rotation = 0)
        ax.set_ylabel(ylabel, fontsize = 10, weight = 'bold')
        ax.invert_yaxis()

    sns.set_style('ticks')

    if isinstance(xs, str):

        fig, axs = plt.subplots(fig_dims[0], fig_dims[1], figsize = figsize,
                                layout = 'constrained', sharex = True)

        fig.supxlabel(xlabels, fontsize = 10, weight = 'bold')

        for i, (ax, pivot_table, ylabel, v_mm) in enumerate(zip(axs.flatten(),
                                                                pivot_tables_plot,
                                                                ylabels,
                                                                v_min_max)):

            plot_ax(i, ax, pivot_table, v_mm, ylabel, cbar_pos)
            ax.set_xlabel('')

    else:

        fig, axs = plt.subplots(fig_dims[0], fig_dims[1], figsize = figsize,
                                layout = 'constrained')

        for i, (ax, pivot_table, xlabel, ylabel, v_mm) in enumerate(zip(axs.flatten(),
                                                                        pivot_tables_plot,
                                                                        xlabels,
                                                                        ylabels,
                                                                        v_min_max)):

            plot_ax(i, ax, pivot_table, v_mm, ylabel, cbar_pos)
            ax.set_xlabel(xlabel, fontsize = 10, weight = 'bold')

    return fig, axs
