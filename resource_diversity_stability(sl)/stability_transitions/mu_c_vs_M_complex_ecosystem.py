# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 19:29:27 2026

@author: jamil
"""

import numpy as np
import sys
import os
import pandas as pd

os.chdir("C:/Users/jamil/Documents/PhD/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models/" + \
         "resource_diversity_stability(sl)/stability_transitions")
    
sys.path.insert(0, "C:/Users/jamil/Documents/PhD/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models/" + \
                    "resource_diversity_stability(sl)")
from simulation_functions import CRM_across_parameter_space, le_pivot_r

sys.path.insert(0, 'C:/Users/jamil/Documents/PhD/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models/cavity_method_functions')
from self_consistency_equation_functions import parameter_combinations

# %%

def poolsize_mu_c(poolsize_range,
                  mu_c_range,
                  fixed_parameters,
                  subdirectory = 'resource_diversity_stability/simulations/M_vs_mu_c_complex_ecosystem_2',
                  save_method = 'v3',
                  **kwargs):
    
    parameters = generate_parameters_M_C(poolsize_range,
                                         mu_c_range,
                                         fixed_parameters)
    
    CRM_across_parameter_space(parameters,
                               subdirectory,
                               ['pool_sizes', 'mu_c_tot'],
                               save_method=save_method,
                               model = 'Self-limiting resource supply, multi-trophic level',
                               **kwargs)
                    
# %%

def generate_parameters_M_C(poolsize_range,
                            mu_c_range,
                            fixed_parameters):
    
    poolsize_mucs = np.unique(parameter_combinations([poolsize_range,
                                                      mu_c_range],
                                                     1),
                              axis = 1)
    
    mucs_sigmacs = [{'mu_c_tot' : poolsize_mucs[1, :],
                     'mu_c_' + str(i) : poolsize_mucs[1, :]/poolsize_mucs[0, :],
                     'sigma_c_' + str(i) : np.repeat(fixed_parameters['sigma_c'], poolsize_mucs.shape[1])/np.sqrt(poolsize_mucs[0, :])}
                      for i in np.arange(2, fixed_parameters['trophic_levels'] + 1)]
                      
    mucs_sigmacs = {key : val 
                    for dictionary in mucs_sigmacs
                    for key, val in dictionary.items()}
    
    parameters = {'pool_sizes' : [np.repeat(np.int64(poolsize),
                                           fixed_parameters['trophic_levels'])
                  for poolsize in poolsize_mucs[0, :]]} | \
                 mucs_sigmacs | \
                 {key : np.repeat(val, poolsize_mucs.shape[1])
                  for key, val in fixed_parameters.items() 
                  if key != ('mu_A' or 'sigma_A' or 'sigma_c')} | \
                {'mu_A' : np.repeat(fixed_parameters['mu_A'], poolsize_mucs.shape[1])/poolsize_mucs[0, :],
                 'sigma_A' : np.repeat(fixed_parameters['sigma_A'], poolsize_mucs.shape[1])/np.sqrt(poolsize_mucs[0, :])}
                
    return pd.DataFrame(parameters).to_dict('records')

# %%

pool_sizes = np.arange(50, 275, 25) # np.arange(200, 260, 20) # np.arange(20, 200, 20)
mu_cs = np.arange(80, 200, 20)

# %%

# data for stability diagram
poolsize_mu_c(pool_sizes, mu_cs,
              {'trophic_levels' : 3, 'sigma_c' : 1.6,
               'mu_y_2' : 1, 'sigma_y_2' : 0.13069,
               'mu_y_3' : 1, 'sigma_y_3' : 0, 'd_2' : 1, 'd_3' : 0.93,
               'b' : 1, 'mu_A' : 10, 'sigma_A' : 0.5})

# %%

# example simulations
poolsize_mu_c([75, 250], [140],
              {'trophic_levels' : 3, 'sigma_c' : 1.6,
               'mu_y_2' : 1, 'sigma_y_2' : 0.13069,
               'mu_y_3' : 1, 'sigma_y_3' : 0, 'd_2' : 1, 'd_3' : 0.93,
               'b' : 1, 'mu_A' : 10, 'sigma_A' : 0.5},
              save_method = 'v2',
              no_communities = 3)

