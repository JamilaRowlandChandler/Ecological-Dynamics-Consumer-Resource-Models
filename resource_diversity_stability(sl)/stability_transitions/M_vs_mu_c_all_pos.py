# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 10:28:10 2026

@author: jamil
"""

import numpy as np
import sys
import os
from typing import Literal, Union
import numpy.typing as npt
from copy import deepcopy

os.chdir("C:/Users/jamil/Documents/PhD/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models/" + \
         "resource_diversity_stability(sl)/stability_transitions")
    
sys.path.insert(0, "C:/Users/jamil/Documents/PhD/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models/" + \
                    "resource_diversity_stability(sl)")
from complete_simulation_functions import CRM_across_parameter_space

sys.path.insert(0, 'C:/Users/jamil/Documents/PhD/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models/cavity_method_functions')
import self_consistency_equation_functions as sce

# %%

def M_effect_fixed_C(M_range : Union[npt.NDArray, list],
                     mu_c_range : Union[npt.NDArray, list],
                     fixed_parameters : dict,
                     subdirectory : str,
                     distribution : Literal['gamma',
                                            'uniform'] = 'gamma',
                     **kwargs):
    
    fixed_parameters_copy = deepcopy(fixed_parameters)
    
    match distribution:
        
        case 'gamma':
            
            fixed_parameters_copy['all_positive'] = True
            
            parameters = generate_parameters_M_C(M_range,
                                                 mu_c_range,
                                                 fixed_parameters_copy)
        
        case 'uniform':
            
            parameters_incomplete = generate_parameters_M_C(M_range,
                                                            mu_c_range,
                                                            fixed_parameters_copy)
            
            parameters = generate_uniform_rates(parameters_incomplete)
            
    CRM_across_parameter_space(parameters,
                               "resource_diversity_stability/simulations/" + \
                                   subdirectory,
                               ['M', 'mu_c_tot'],
                               model = 'Self-limiting resource supply',
                               save_method = 'v3',
                               **kwargs)
                    
# %%

def generate_parameters_M_C(M_range,
                            mu_c_range,
                            fixed_parameters):
    
    M_mu_c_combinations = np.unique(sce.parameter_combinations([M_range,
                                                                mu_c_range],
                                                               1),
                                    axis = 1)
    
    Ms = M_mu_c_combinations[0, :]
    mu_c_tots = M_mu_c_combinations[1, :]
    
    mu_cs = mu_c_tots/Ms
    sigma_cs = fixed_parameters['sigma_c']/np.sqrt(Ms)
    
    Ss = Ms/fixed_parameters['gamma']
    
    variable_parameters = np.vstack([M_mu_c_combinations,
                                     Ss,
                                     mu_cs,
                                     sigma_cs])

    # array of all parameter combinations
    parameters = sce.variable_fixed_parameters(variable_parameters,
                                               fixed_parameters,
                                               ['M',
                                                'mu_c_tot',
                                                'S',
                                                'mu_c', 
                                                'sigma_c'])
    
    for parms in parameters:
        
        parms['S'] = np.int32(parms['S'])
        parms['M'] = np.int32(parms['M'])
    
    return parameters

# %%

def generate_uniform_rates(parameters):
    
    def uniform_growth_consumption(M, S,
                                   mu_c, sigma_c,
                                   mu_y, sigma_y):
        
        consumption = np.random.uniform(mu_c - np.sqrt(3)*sigma_c,
                                        mu_c + np.sqrt(3)*sigma_c,
                                        (M, S))
        
        yield_conversion = np.random.uniform(mu_y - np.sqrt(3)*sigma_y,
                                             mu_y + np.sqrt(3)*sigma_y,
                                             (S, M))
        
        growth = yield_conversion * consumption.T
        
        return consumption, growth
    
    for parms in parameters:
        
        parms['consumption'], parms['growth'] = \
            uniform_growth_consumption(parms['M'],
                                       parms['S'],
                                       parms['mu_c'],
                                       parms['sigma_c'],
                                       parms['mu_y'],
                                       parms['sigma_y'])
            
    return parameters

# %%

resource_pool_sizes = np.arange(50, 275, 25)
mu_cs = np.linspace(100, 250, 8)

M_effect_fixed_C(resource_pool_sizes,
                 mu_cs,
                 {'sigma_c' : 1.6,
                  'mu_y': 1.0, 
                  'sigma_y' : 1.6/np.sqrt(150), 
                  'b' : 1.0,
                  'd' : 1.0,
                  'gamma' : 1},
                 "M_vs_mu_c_all_pos")

# %%

resource_pool_sizes = np.arange(50, 275, 25)
mu_cs = np.linspace(100, 205, 8)

M_effect_fixed_C(resource_pool_sizes,
                 mu_cs,
                 {'sigma_c' : 1.6,
                  'mu_y': 1.0, 
                  'sigma_y' : 1.6/np.sqrt(150), 
                  'b' : 1.0,
                  'd' : 1.0,
                  'gamma' : 1},
                 "M_vs_mu_c_uniform",
                 distribution = 'uniform')


