# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 11:23:47 2026

@author: jamil
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import os
import sys
from tqdm import tqdm
from typing import Literal
import numpy.typing as npt

os.chdir('C:/Users/jamil/Documents/PhD/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models/external_resource_stability/stability_transitions')

sys.path.insert(0, "C:/Users/jamil/Documents/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models" + \
                    "/consumer_resource_modules")
from effective_LV_models import Effective_LV_model
from community_level_properties import max_le
    
sys.path.insert(0, "C:/Users/jamil/Documents/PhD/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models" + \
                    "/external_resource_stability")
from simulation_functions import simulation_df_from_communities

sys.path.insert(0, 'C:/Users/jamil/Documents/PhD/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models/cavity_method_functions')
from self_consistency_equation_functions import parameter_combinations, variable_fixed_parameters

# %%

def egLVs(model, 
          rhos, 
          sigmas,
          fixed_parameters,
          directory : str = "egLV/rho_vs_sigma"):
    
    '''
    
    ...

    Parameters
    ----------
    resource_pool_sizes : npt.NDArray, optional
        The default is np.arange(50, 275, 25).
    mu_c : float, optional
        The default is 145.
    gLV_directory : str, optional
        File directory to save elVs in. The default is "egLV/M_vs_mu_c".
    all_resource_survive : bool, optional
        Do we assume all resources survive or not. The default is False.

    Returns
    -------
    dict
        resource pool size vs max. lyapunov exponents for all eLVs.

    '''

    def call_egLV_communities(gc_args : dict, 
                              filename : str):
        
        '''
        
        ...

        '''
        
        # generate eLV from CRM communities, run simulations
        communities = [egLV_communities(model, 
                                        init_class,
                                        gc_args,
                                        model_specific_rates_args)]
       
        # save eLV
        communities_df = simulation_df_from_communities(communities,
                                                        model,
                                                        gc_args_list[0]["method"])
        
        communities_df.to_csv("C:/Users/jamil/Documents/PhD/Data/external_resource_stability/simulations/" + \
                              directory + "/simulations_" + filename + ".csv")
        
    ###################################################################################
    
    # make file directory for eLVs
    if not os.path.exists("C:/Users/jamil/Documents/PhD/Data/external_resource_stability/simulations/" + \
                          directory):
        
        os.makedirs("C:/Users/jamil/Documents/PhD/Data/external_resource_stability/simulations/" + \
                    directory)
            
    init_class, gc_args_list, model_specific_rates_args = generate_parameters(rhos,
                                                                              sigmas,
                                                                              fixed_parameters)
     
    # generate filenames based on mu_c
    filenames = [str(gc_args['rho']) + "_" + str(gc_args['sigma_c']) 
                 for gc_args in gc_args_list]
    
    for gc_args in gc_args_list:
        
        gc_args['mu_c'] = gc_args['mu_c']/init_class['M']
        gc_args['sigma_c'] = gc_args['sigma_c']/np.sqrt(init_class['M'])
        gc_args['mu_g'] = gc_args['mu_g']/init_class['M']
        gc_args['sigma_g'] = gc_args['sigma_g']/np.sqrt(init_class['M'])
    
    for gc_args, filename in tqdm(zip(gc_args_list, filenames),
                                      leave = True,
                                      position = 1,
                                      total = len(filenames)):
    
        call_egLV_communities(gc_args, filename)
                    
# %%

def generate_parameters(rhos,
                        sigmas,
                        fixed_parameters):
    
    rho_sigma_combinations = np.unique(parameter_combinations([rhos,
                                                               sigmas],
                                                                  0),
                                    axis = 1)
    
    gc_args_list = [{'method' : 'coupled by rho',
                     'mu_c' : fixed_parameters['mu_c'],
                     'sigma_c' : rho_sigma_combo[1],
                     'mu_g' : fixed_parameters['mu_g'],
                     'sigma_g' : rho_sigma_combo[1],
                     'rho' : rho_sigma_combo[0]}
                    for rho_sigma_combo in rho_sigma_combinations]
    
    model_specific_rates_args = dict(death_method = 'constant',
                                     death_args =  {'d' : fixed_parameters['d']},
                                     influx_method = 'constant',
                                     influx_args = {'b' : fixed_parameters['b']},
                                     outflux_method = 'constant',
                                     outflux_args = {'o' : fixed_parameters['o']})
    
    init_class = dict(S = np.int64(fixed_parameters['S']),
                      M = np.int64(fixed_parameters['M']))
    
    return init_class, gc_args_list, model_specific_rates_args

# %%

def egLV_communities(model : Literal["Externally-supplied resources",
                                     "Self-limiting resource supply"],
                     init_class : dict, 
                     growth_consumption_rates_args,
                     model_specific_rates_args):
    
    # initialise eLV
    community = Effective_LV_model(model, init_class)
    
    # generate CRM parameters
    community.growth_consumption_rates(**growth_consumption_rates_args)
    community.model_specific_rates(**model_specific_rates_args)
    
    # generate eLV parameters
    community.generate_elv_parameters()
    # calculate mean interaction strength, self-inhibition etc
    community.calculate_interaction_stats()
    
    # run simulations from randomly generated initial abundances
    community.simulation(t_end = 400,
                         initial_abundances=np.random.uniform(1e-8,
                                                              2/community.no_species,
                                                              community.no_species))
    
    # numerically estimate the max. lyapunov exponent
    community.max_lyapunov_exponent = max_le(community,
                                             community.ODE_sol.y[:, -1],
                                             T = 1000,
                                             perturbation = 1e-6)
    
    return community
   
   