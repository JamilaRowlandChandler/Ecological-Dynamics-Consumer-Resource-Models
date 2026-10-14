# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 23:43:47 2026

@author: jamil
"""

import numpy as np
import sys
import os

os.chdir("C:/Users/jamil/Documents/PhD/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models/" + \
         "resource_diversity_stability(sl)/stability_transitions")
    
sys.path.insert(0, "C:/Users/jamil/Documents/PhD/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models/" + \
                    "resource_diversity_stability(sl)")
from complete_simulation_functions import CRM_across_parameter_space

sys.path.insert(0, 'C:/Users/jamil/Documents/PhD/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models/cavity_method_functions')
import self_consistency_equation_functions as sce

# %%

def M_effect_resource_exp(M_range,
                          resource_exponents,
                          fixed_parameters,
                          subdirectory,
                          **kwargs):
    
    parameters = generate_parameters_M_exp(M_range,
                                           resource_exponents,
                                           fixed_parameters)
    
    CRM_across_parameter_space(parameters,
                               "resource_diversity_stability/simulations/" + \
                                   subdirectory,
                               ['M', 'resource_exponent'],
                               model = 'Self-limiting resource supply, generic resource type',
                               save_method = 'v3',
                               **kwargs)
                    
# %%

def generate_parameters_M_exp(M_range,
                              resource_exponents,
                              fixed_parameters):
    
    M_exp_combinations = np.unique(sce.parameter_combinations([M_range,
                                                               resource_exponents],
                                                               1),
                                    axis = 1)
    
    Ms = M_exp_combinations[0, :]
    
    mu_cs = fixed_parameters['mu_c_tot']/Ms
    sigma_cs = fixed_parameters['sigma_c']/np.sqrt(Ms)
    
    Ss = Ms/fixed_parameters['gamma']
    
    variable_parameters = np.vstack([M_exp_combinations,
                                     Ss,
                                     mu_cs,
                                     sigma_cs])

    # array of all parameter combinations
    parameters = sce.variable_fixed_parameters(variable_parameters,
                                               fixed_parameters,
                                               ['M',
                                                'resource_exponent',
                                                'S',
                                                'mu_c', 
                                                'sigma_c'])
    
    for parms in parameters:
        
        parms['S'] = np.int32(parms['S'])
        parms['M'] = np.int32(parms['M'])
    
    return parameters

# %%

resource_pool_sizes = np.arange(50, 375, 50)
resource_exponents = np.arange(0.7, 1.1, 0.1)

M_effect_resource_exp(resource_pool_sizes,
                      resource_exponents,
                      {'mu_c_tot' : 50.0,
                       'sigma_c' : 1.0,
                       'mu_y': 1.0, 
                       'sigma_y' : 1.6/np.sqrt(150), 
                       'b' : 1.0,
                       'd' : 1.0,
                       'gamma' : 1},
                      "M_vs_mu_c_generic_resource",
                      no_communities=5,
                      no_init_conds=1,
                      t_end=7000)

