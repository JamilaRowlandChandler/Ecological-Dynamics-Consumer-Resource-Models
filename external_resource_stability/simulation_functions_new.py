# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 10:19:33 2025

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
from scipy.stats import pearsonr
import numpy.typing as npt
from typing import TYPE_CHECKING, Union, TypedDict, NotRequired, Literal
from ast import literal_eval
import re
from json import dumps, loads

sys.path.insert(0, 'C:/Users/jamil/Documents/PhD/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models/consumer_resource_modules')
from models import Consumer_Resource_Model
from community_level_properties import max_le

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
        
    ######################################
        
    def model_specific_args(parameter_sets : list[dict],
                            model):
        
        '''
        
        Correctly format model-specific parameters for the consumer-resource 
        model class 

        Parameters
        ----------
        parameter_sets : list[Dict]
            List of parameter sets for the Consumer_Resource_Model class.
        simulation_kwargs : Dict(str, any)
            Optional arguments for the Consumer_Resource_Model.

        Returns
        -------
        m_s_rates_args_list : list(Dict)
            Correctly-formatted model-specific arguments

        '''
        
        # extract model type
        match model:
            
            case "Self-limiting resource supply":
                
                # make list of resource and species pool sizes
                initialisation_list = [dict(model = model,
                                            pool_sizes = [parm_set['M'], parm_set['S']])
                                       for parm_set in parameter_sets]
                
                # extract and rename arguments for death rates and instrinsic
                # resource growth rates
                m_s_rates_args_list = [dict(death_method = 'constant',
                                            death_args =  {'d' : parm_set['d']},
                                            resource_growth_method = 'constant',
                                            resource_growth_args = {'b' : parm_set['b']})
                                       for parm_set in parameter_sets]
                
            case "Self-limiting resource supply, self-inhibition":
                
                # make list of resource and species pool sizes
                initialisation_list = [dict(model = model,
                                            pool_sizes = [parm_set['M'], parm_set['S']])
                                       for parm_set in parameter_sets]
                
                # extract and rename arguments for death rates, instrinsic 
                # resource growth rates, and consumer self-inhibition
                m_s_rates_args_list = [dict(death_method = 'constant',
                                            death_args =  {'d' : parm_set['d']},
                                            resource_growth_method = 'constant',
                                            resource_growth_args = {'b' : parm_set['b']},
                                            si_method = 'constant',
                                            si_args = {'si' : parm_set['si']})
                                       for parm_set in parameter_sets]
                
                # create list of arguments for generating growth and consumption rates
               
                
            case "Externally-supplied resources":
                
                # make list of resource and species pool sizes
                initialisation_list = [dict(model = model,
                                            pool_sizes = [parm_set['M'], parm_set['S']])
                                       for parm_set in parameter_sets]
                
                m_s_rates_args_list = [dict(death_method = 'constant',
                                            death_args =  {'d' : parm_set['d']},
                                            influx_method = 'constant',
                                            influx_args = {'b' : parm_set['b']},
                                            outflux_method = 'constant',
                                            outflux_args = {'o' : parm_set['o']})
                                       for parm_set in parameter_sets]
                
            case "Hybrid resource supply":
                
                # make list of resource and species pool sizes
                initialisation_list = [dict(model = model,
                                            pool_sizes = [parm_set['M'], parm_set['S']])
                                       for parm_set in parameter_sets]
                
                m_s_rates_args_list = [dict(death_method = 'constant',
                                            death_args =  {'d' : parm_set['d']},
                                            influx_method = 'constant',
                                            influx_args = {'b' : parm_set['b']},
                                            outflux_method = 'constant',
                                            outflux_args = {'o' : parm_set['o']},
                                            resource_inhibition_method = 'constant',
                                            resource_inhibition_args = {'a' : parm_set['a']})
                                       for parm_set in parameter_sets]
                    
            case "Self-limiting resource supply, multi-trophic level":
                
                # make list of resource and species pool sizes
                initialisation_list = [dict(model = model,
                                            pool_sizes = parm_set['pool_sizes'])
                                       for parm_set in parameter_sets]
                
                m_s_rates_args_list = [dict(death_methods = np.repeat('constant',
                                                                      parm_set['trophic_levels'] - 1).tolist(),
                                            death_args =  [{'d_' + str(i) : parm_set['d_' + str(i)]}
                                                           for i in np.arange(2, parm_set['trophic_levels'] + 1)],
                                            resource_growth_method = 'constant',
                                            resource_growth_args = {'b' : parm_set['b']},
                                            resource_interaction_method = 'normal',
                                            resource_interaction_args = {'mu' : parm_set['mu_A'], 
                                                                         'sigma' : parm_set['sigma_A']})
                                       for parm_set in parameter_sets]
                
        return initialisation_list, m_s_rates_args_list
    
    def growth_consumption_rates(parameter_sets, model):
        
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
        
        if model.endswith("multi-trophic level") is True:
            
            if 'mu_y_2' in parameter_sets[0].keys():
                
                g_c_rates_args_list = rue_coupled_tl(parameter_sets,
                                                     parameter_sets[0]['trophic_levels'])
            
            else:
                
                g_c_rates_args_list = rho_coupled_tl(parameter_sets,
                                                     parameter_sets[0]['trophic_levels'])
        
        else:
            
            if 'mu_y' in parameter_sets[0].keys():
                
                g_c_rates_args_list = rue_coupled(parameter_sets)
            
            else:
                
                g_c_rates_args_list = rho_coupled(parameter_sets)
        
        return g_c_rates_args_list

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
    
    # create list of arguments for generating model specific rates
    initialisation_list, m_s_rates_args_list = model_specific_args(parameter_sets,
                                                                   model)

    g_c_rates_args_list = growth_consumption_rates(parameter_sets,
                                                   model)
    
    # Iterate through the parameter space, creating and simulating community dynamics
    for name, init_class, growth_consumption_rates_args, model_specific_rates_args in \
        tqdm(zip(names_list, initialisation_list, g_c_rates_args_list, m_s_rates_args_list),
             position = 0, leave = True, total = len(names_list)):
            
        CRMs_create_and_save(subdirectory,
                             "simulations_" + name,
                             init_class,
                             growth_consumption_rates_args,
                             model_specific_rates_args,
                             save_method,
                             **complete_sim_kwargs)
    
#%%

def CRMs_create_and_save(subdirectory : str,
                         filename : str,
                         init_class : list[dict[str : int]],
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
                         **kwargs : any):
    
    '''
    
    Simulate communites where model parameters are sampled from the same distributions,
    then save the data
    
    Parameters
    ----------
    filepath : str
        filepath to save community data.
    no_species : int
        species pool size.
    no_resources : int
        resource pool size.
    growth_consumption_rates_args : TypedDict('gc_args', {'method' : str,
                                                          'mu_c' : float,
                                                          'sigma_c' : float,
                                                          'mu_g' : float,
                                                          'sigma_g' : float,
                                                          'conserve_mass' : NotRequired[bool],
                                                          'kwargs' : NotRequired[any]})
        Arguments used to generate growth and consumption rates.
    model_specific_rates_args : Dict(str, any)
        Arguments used to generate model-specific rates.
    **kwargs : any
        Optional arguments.

    Returns
    -------
    None.

    '''
    
    if init_class['model'].endswith("multi-trophic level"):
    
        communities = complex_ecosystem_model_dynamics(init_class,
                                                       growth_consumption_rates_args,
                                                       model_specific_rates_args,
                                                       **kwargs)
        
    else:
        
        communities = consumer_resource_model_dynamics(init_class,
                                                       growth_consumption_rates_args,
                                                       model_specific_rates_args,
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
            
            if init_class['model'].endswith("multi-trophic level"):
                
                df = simulation_df_from_communities(communities,
                                                    init_class['model'],
                                                    growth_consumption_rates_args[0]['method'])
                
            else:
                
                df = simulation_df_from_communities(communities,
                                                    init_class['model'],
                                                    growth_consumption_rates_args['method'])
            
            df.to_csv("C:/Users/jamil/Documents/PhD/Data/" + \
                         subdirectory + "/" + filename + ".csv")
            
    del communities 
   

# %%

def consumer_resource_model_dynamics(init_class,
                                     growth_consumption_rates_args : TypedDict('gc_args',
                                                                               {'method' : str,
                                                                                'mu_c' : float,
                                                                                'sigma_c' : float,
                                                                                'mu_g' : float,
                                                                                'sigma_g' : float,
                                                                                'conserve_mass' : NotRequired[bool],
                                                                                'kwargs' : NotRequired[any]}),
                                     model_specific_rates_args : dict[str, any],
                                     no_communities : int = 5,
                                     no_init_conds : int = 2,
                                     t_end : float = 3500):
    
    '''
    
    Simulate communites where model parameters are sampled from the same distributions

    Parameters
    ----------
    no_species : int
        species pool size.
    no_resources : int
        resource pool size.
    growth_consumption_rates_args : TypedDict('gc_args', {'method' : str,
                                                          'mu_c' : float,
                                                          'sigma_c' : float,
                                                          'mu_g' : float,
                                                          'sigma_g' : float,
                                                          'conserve_mass' : NotRequired[bool],
                                                          'kwargs' : NotRequired[any]})
        Arguments used to generate growth and consumption rates.
    model_specific_rates_args : Dict(str, any)
        Arguments used to generate model-specific rates.
    model : Literal["Self-limiting resource supply", "Self-limiting resource supply, self-inhibition"], optional
        Models. The default is "Self-limiting resource supply".
    no_communities : int, optional
        Number of communities to create. The default is 5.
    no_init_conds : int, optional
        Number of initial abundances dynamics are simulated from. The default is 2.
    t_end : float, optional
        Simulation end time. The default is 3500.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    
    def community_dynamics(init_class,
                           growth_consumption_rates_args : TypedDict('gc_args',
                                                                     {'method' : str,
                                                                      'mu_c' : float,
                                                                      'sigma_c' : float,
                                                                      'mu_g' : float,
                                                                      'sigma_g' : float,
                                                                      'conserve_mass' : NotRequired[bool],
                                                                      'kwargs' : NotRequired[any]}),
                           model_specific_rates_args : dict[str, any],
                           no_init_conds : int,
                           t_end : float):
        
        '''
        
        create and simulate community dynamics 
        
        '''
        
        #breakpoint()
       
        # initialise consumer-resource model class
        community = Consumer_Resource_Model(**init_class)

        # generate model parameters
        community.growth_consumption_rates(**growth_consumption_rates_args)
        community.model_specific_rates(**model_specific_rates_args)
        
        # simulate commmunity dynamics
        community.simulate_community(t_end, no_init_conds)
        
        # estimate community properties, including the max. lyapunov exponent
        community.calculate_community_properties()
        community.lyapunov_exponent = max_le(community, community.ODE_sols[0].y[:, -1],
                                             T = 1000, perturbation = 1e-6)

        return community 

    # generate n communities, where n = no_communities
    communities = [deepcopy(community_dynamics(init_class,
                                               growth_consumption_rates_args,
                                               model_specific_rates_args,
                                               no_init_conds,
                                               t_end))
                                  for _ in range(no_communities)]
    
    return communities

# %%

def complex_ecosystem_model_dynamics(init_class,
                                     growth_consumption_rates_args : list[dict],
                                     model_specific_rates_args : dict[str, any],
                                     no_communities : int = 5,
                                     no_init_conds : int = 2,
                                     t_end : float = 3500):
    
    '''
    
    Simulate communites where model parameters are sampled from the same distributions

    Parameters
    ----------
    no_species : int
        species pool size.
    no_resources : int
        resource pool size.
    growth_consumption_rates_args : TypedDict('gc_args', {'method' : str,
                                                          'mu_c' : float,
                                                          'sigma_c' : float,
                                                          'mu_g' : float,
                                                          'sigma_g' : float,
                                                          'conserve_mass' : NotRequired[bool],
                                                          'kwargs' : NotRequired[any]})
        Arguments used to generate growth and consumption rates.
    model_specific_rates_args : Dict(str, any)
        Arguments used to generate model-specific rates.
    model : Literal["Self-limiting resource supply", "Self-limiting resource supply, self-inhibition"], optional
        Models. The default is "Self-limiting resource supply".
    no_communities : int, optional
        Number of communities to create. The default is 5.
    no_init_conds : int, optional
        Number of initial abundances dynamics are simulated from. The default is 2.
    t_end : float, optional
        Simulation end time. The default is 3500.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    
    def community_dynamics(init_class,
                           growth_consumption_rates_args : list[dict],
                           model_specific_rates_args : dict[str, any],
                           no_init_conds : int,
                           t_end : float):
        
        '''
        
        create and simulate community dynamics 
        
        '''
        
        #breakpoint()
    
        # initialise consumer-resource model class
        community = Consumer_Resource_Model(**init_class)

        # generate model parameters
        
        for gc_rates_args in growth_consumption_rates_args:
            
            community.growth_consumption_rates(**gc_rates_args)
        
        community.model_specific_rates(**model_specific_rates_args)
        
        # simulate commmunity dynamics
        community.simulate_community(t_end, no_init_conds)
        
        # estimate community properties, including the max. lyapunov exponent
        community.calculate_community_properties()
        community.lyapunov_exponent = max_le(community, community.ODE_sols[0].y[:, -1],
                                             T = 1000, perturbation = 1e-6)

        return community 

    # generate n communities, where n = no_communities
    communities = [deepcopy(community_dynamics(init_class,
                                               growth_consumption_rates_args,
                                               model_specific_rates_args,
                                               no_init_conds,
                                               t_end))
                                  for _ in range(no_communities)]
    
    return communities

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
                      "SL_SI_CRM" : "Self-limiting resource supply, self-inhibition",
                      "SL_TL_CRM" : "Self-limiting resource supply, multi-trophic level",
                      "ES_CRM" : "Externally-supplied resources",
                      "Hybrid_CRM" : "Hybrid resource supply",
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
    directory : str
        filepath for community data.

    Returns
    -------
    df : pd.DataFrame
        Dataframe of model parameters and community properties.

    '''
    
    # parameters to include in the dataframe
    
    if model.endswith("multi-trophic level"):
    
        parameters = extract_trophic_level_parms(model,
                                                 communities[0].trophic_levels) + \
                        model_specific_parameters(model) 
            
    else:
        
        parameters = extract_growth_consumption_parms() + \
                    model_specific_parameters(model) 
    
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

def model_specific_parameters(model):
    
    if model.startswith("Self-limiting resource supply") is True:
        
        ms_parm_list = ['b_val']
    
    elif model.startswith("Externally-supplied resources") is True:
        
        ms_parm_list = ['b_val', 'o_val']
        
    elif model.startswith("Hybrid resource supply") is True:
        
        ms_parm_list = ['b_val', 'o_val', 'a_val']
        
    return ms_parm_list

# %%

def extract_trophic_level_parms(trophic_levels):
        
    poolsize_parms = ['pool_sizes']
    
    m_parms = [string + str(i) for string in ['mu_g_', 'sigma_g_',
                                              'mu_c_', 'sigma_c_',
                                              'rho_']
                for i in np.arange(2, trophic_levels + 1)] + \
                ['d_' + str(i) + "_val"
                 for i in np.arange(2, trophic_levels + 1)] + \
                ['mu_A', 'sigma_A']
        
    return poolsize_parms + m_parms


# %% 

def extract_growth_consumption_parms():

    poolsize_parms = ['no_resources', 'no_species']
    m_parms = ['mu_g', 'sigma_g', 'mu_c', 'sigma_c', 'rho', 'd_val']
        
    return poolsize_parms + m_parms
    
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
                                 'mu_g_' + str(i) : 'mu_g_' + str(i) + '/PS_' +  + str(i),
                                 'sigma_g_' + str(i) : 'sigma_g_' + str(i) + '/root_PS' + str(i)},
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
    
    if model.endswith("multi-trophic level") is True:
        
        match gc_method:
            
            case 'coupled by rho':
        
                for trophic_level in np.arange(1, df.loc[0, 'trophic_levels'] + 1):
            
                    df = df.pipe(coupled_rho, i = trophic_level)
                    
            case 'growth function of consumption':
                
                for trophic_level in np.arange(2, df.loc[0, 'trophic_levels'] + 1):
            
                    df = df.pipe(coupled_rue, i = trophic_level)
        
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
    
    if model.endswith("multi-trophic level") is True:
        
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
        
        elif model.startswith("Externally-supplied resources") is True:
            
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

    Returns
    -------
    df : pd.DataFrame
        Dataframe of model parameters and community properties.

    '''
    
    def load_data_create_df(filepath, method):
        
        match method:
            
            case 'v1':
    
                communities = load_in_communities(filepath)
            
            case 'v2':
            
                communities = pd.read_pickle(filepath)
        
        df = community_dynamics_df(communities, parameters)
    
        return df
    
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
        
        #breakpoint()
    
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
