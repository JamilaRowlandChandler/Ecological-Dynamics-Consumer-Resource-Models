# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 16:22:19 2026

@author: jamil
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import os
import sys
from tqdm import tqdm
from typing import Literal, Union
import numpy.typing as npt
from copy import deepcopy

os.chdir('C:/Users/jamil/Documents/PhD/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models/resource_diversity_stability(sl)/effective_gLV')

sys.path.insert(0, "C:/Users/jamil/Documents/PhD/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models" + \
                    "/consumer_resource_modules")
from models import Consumer_Resource_Model
from community_level_properties import max_le, eigenspectrum
    
sys.path.insert(0, "C:/Users/jamil/Documents/PhD/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models" + \
                    "/resource_diversity_stability(sl)")
from complete_simulation_functions import pickle_dump, simulation_df_from_communities

# %%

def extract_parameters(base_community : Literal["SL_CRM"]) -> dict:
    
    parameters = {attr : getattr(base_community, 
                                 attr, 
                                 None)
                  for attr in ['no_resources', 'mu_c', 'sigma_c',
                               'mu_g', 'sigma_g',
                               'consumption',
                               'growth',
                               'b', 'd']}
    
    return parameters

# %%

def separate_timescales(parameters : dict,
                        epsilon : float) -> dict:
    
    separated_parameters = deepcopy(parameters)
    
    separated_parameters['epsilon'] = epsilon
    
    return separated_parameters

# %%

def separate_parameter_timescales(base_community : Literal["SL_CRM"],
                                  epsilons : Union[list[float], npt.NDArray]):
    
    base_parameters = extract_parameters(base_community)
    
    parameters_sep_by_eps = [separate_timescales(base_parameters,
                                                 epsilon)
                             for epsilon in epsilons]
    
    return parameters_sep_by_eps

# %%

def resimulate_CRM_timescale(parameters : dict,
                             fast_variable : Literal["resources",
                                                     "consumers"]) -> None:
    
    M = parameters['no_resources']
    
    mu_c = parameters['mu_c']
    sigma_c = parameters['sigma_c']
    mu_y = parameters['mu_g']
    sigma_y = parameters['sigma_g']
    consumption = parameters['consumption']
    growth = parameters['growth']
    
    intrinsic_resource_growth = parameters['b']
    death = parameters['d']
    
    epsilon = parameters['epsilon']
    
    community = Consumer_Resource_Model("Self-limiting resource supply",
                                        M,
                                        M)
    
    community.growth_consumption_rates('user-supplied',
                                       mu_c = mu_c,
                                       sigma_c = sigma_c,
                                       mu_g = mu_y,
                                       sigma_g = sigma_y,
                                       consumption = consumption,
                                       growth = growth)
    community.model_specific_rates(death_method = "user-supplied",
                                   death_args = {'d' : death},
                                   resource_growth_method = "user-supplied",
                                   resource_growth_args = {'b' : intrinsic_resource_growth})
    
    if fast_variable == "resources":
    
        community.timescale_separation(epsilon_r = epsilon)
        
    elif fast_variable == "species":
        
        community.timescale_separation(epsilon_s = epsilon)
        
    # run simulations from randomly generated initial abundances
    community.simulate_community(t_end = 7000,
                                 no_init_cond = 2)
    
    community.calculate_community_properties()
       
    # numerically estimate the max. lyapunov exponent
    community.lyapunov_exponent = max_le(community,
                                         community.ODE_sols[0].y[:, -1],
                                         T = 1000,
                                         perturbation = 1e-6)
    
    eigenspec_stats = [eigenspectrum(community,
                                     ode_sol.y[:, -1])
                       for ode_sol in community.ODE_sols]
    
    community.eigenvec_resource_mag = [eig_stat['magnitude_ratios']['resources'] 
                                       for eig_stat in eigenspec_stats]
    
    return community
    
# %%

def CRM_timescale_separation(CRM_directory : str,
                             CRM_ts_directory : str,
                             epsilons,
                             resource_pool_sizes,
                             mu_c,
                             extra_name : str = "",
                             fast_variable : Literal["resources",
                                                     "species"] = "resources"):


    def read_call_timescale_separate(full_CRM_directory : str,
                                     full_CRM_ts_directory : str,
                                     epsilons):
        
        # read in consumer-resource model (CRM) communities
        CRM_communities = pd.read_pickle(full_CRM_directory)
        
        parameters_sep_by_eps = np.array([separate_parameter_timescales(CRM_community,
                                                                        epsilons)
                                          for CRM_community in CRM_communities]).flatten()
        
        CRM_ts_communities = [resimulate_CRM_timescale(parameters,
                                                       fast_variable)
                              for parameters in
                              tqdm(parameters_sep_by_eps,
                                   leave = True,
                                   position = 1,
                                   total = len(parameters_sep_by_eps))]
        
        df = simulation_df_from_communities(CRM_ts_communities,
                                            "Self-limiting resource supply",
                                            "growth function of consumption",
                                            extra_parameters = ["timescalar"])
        
        df.to_csv(full_CRM_ts_directory)
    
    ###################################################################################
    
    full_CRM_directory = "C:/Users/jamil/Documents/PhD/Data/resource_diversity_stability/simulations/" + \
                           CRM_directory
                           
    full_CRM_ts_directory = "C:/Users/jamil/Documents/PhD/Data/resource_diversity_stability/simulations/" + \
                          CRM_ts_directory
    
    # make file directory for eLVs
    if not os.path.exists(full_CRM_ts_directory):
        
        os.makedirs(full_CRM_ts_directory)
            
    # generate filenames based on mu_c
    filenames = ["simulations_" + \
                 str(M) + "_" + str(np.round(mu_c/M, 4)) + extra_name
                 for M in resource_pool_sizes]
        
    for filename in tqdm(filenames,
                         leave = True,
                         position = 0,
                         total = len(filenames)):
        
        read_call_timescale_separate(full_CRM_directory + "/" + filename + ".pkl",
                                     full_CRM_ts_directory + "/" + filename + ".csv",
                                     epsilons)

# %%

epsilons = 10.0**np.arange(-5.0, 1.0, 1.0)

CRM_timescale_separation(CRM_directory = "M_vs_mu_c",
                         CRM_ts_directory = "CRM_TS/Fast_Consumer/M_vs_mu_c",
                         epsilons = epsilons,
                         resource_pool_sizes = np.arange(50, 150, 25), # np.arange(50, 275, 25),
                         mu_c = 145,
                         fast_variable="species")

# %%

epsilons = np.arange(0.3, 1.1, 0.2)

CRM_timescale_separation(CRM_directory = "M_vs_mu_c",
                         CRM_ts_directory = "CRM_TS/Fast_Consumer/small_separation",
                         epsilons = epsilons,
                         resource_pool_sizes = np.arange(50, 275, 25),
                         mu_c = 145,
                         fast_variable="species")
