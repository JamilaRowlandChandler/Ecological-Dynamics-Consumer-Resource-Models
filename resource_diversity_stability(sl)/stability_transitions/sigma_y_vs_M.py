# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 23:32:42 2025

@author: jamil
"""

# -*- coding: utf-8 -*-
"""
Created on Fri May  9 15:32:30 2025

@author: jamil
"""

import numpy as np
import numpy.typing as npt
from typing import Union
import sys
import os

os.chdir("C:/Users/jamil/Documents/PhD/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models/" + \
         "resource_diversity_stability(sl)/stability_transitions")
    
sys.path.insert(0, "C:/Users/jamil/Documents/PhD/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models/" + \
                    "resource_diversity_stability(sl)")
from simulation_functions import CRM_across_parameter_space

sys.path.insert(0, 'C:/Users/jamil/Documents/PhD/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models/cavity_method_functions')
import self_consistency_equation_functions as sce

# %%

def M_effect_sigma_y(M_range : Union[npt.NDArray, tuple, list],
                     sigma_y_range : Union[npt.NDArray, tuple, list],
                     n : int,
                     fixed_parameters : dict,
                     subdirectory : str,
                     **simulation_kwargs : any):
    
    parameters = generate_parameters_M_sigma_y(M_range, sigma_y_range, 
                                               n, fixed_parameters)
    
    CRM_across_parameter_space(parameters, subdirectory,
                               ['M', 'sigma_y'], **simulation_kwargs)
                
# %%

def generate_parameters_M_sigma_y(M_range : Union[npt.NDArray, tuple, list],
                                  sigma_y_range : Union[npt.NDArray, tuple, list],
                                  n : int,
                                  fixed_parameters : dict):
    
    M_sigma_y_combinations = np.unique(sce.parameter_combinations([M_range,
                                                                   sigma_y_range],
                                                                  n),
                                    axis = 1)
    
    variable_parameters = np.vstack([M_sigma_y_combinations[0, :]/fixed_parameters['gamma'],
                                     M_sigma_y_combinations,
                                     np.repeat(fixed_parameters['mu_c'],
                                               M_sigma_y_combinations.shape[1])/M_sigma_y_combinations[0, :],
                                     np.repeat(fixed_parameters['sigma_c'],
                                               M_sigma_y_combinations.shape[1])/np.sqrt(M_sigma_y_combinations[0, :])])
    
    # array of all parameter combinations
    parameters = sce.variable_fixed_parameters(variable_parameters,
                                               fixed_parameters,
                                               ['S', 'M', 'sigma_y', 'mu_c', 'sigma_c'])
    
    for parms in parameters:
        
        parms['S'] = np.int32(parms['S'])
        parms['M'] = np.int32(parms['M'])
    
    return parameters
    
# %%

resource_pool_sizes = np.arange(50, 275, 25)
#resource_pool_sizes = 225

# %%

M_effect_sigma_y(resource_pool_sizes,
                 (0.05, 0.25),
                 9,
                 {'mu_c' : 160, 'sigma_c' : 1.6,
                  'mu_y' : 1, 'b' : 1, 'd' : 1, 'gamma' : 1},
                 'resource_diversity_stability/simulations/M_vs_sigma_y_more',
                 simulation_kwargs=dict(no_communities = 40,
                                        t_end = 7000))
