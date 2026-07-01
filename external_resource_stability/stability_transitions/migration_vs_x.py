# -*- coding: utf-8 -*-
"""
Created on Tue May  5 15:52:11 2026

@author: jamil
"""

import numpy as np
import sys
import os
from copy import deepcopy
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

# %%

abspath = os.path.abspath(__file__)
file_directory_name = os.path.dirname(abspath)
os.chdir(file_directory_name)

sys.path.insert(0, file_directory_name.removesuffix("\\stability_transitions"))
from simulation_functions_new import CRM_across_parameter_space, le_pivot_r

sys.path.insert(0,  file_directory_name.removesuffix("\\external_resource_stability\\stability_transitions") + \
                "\\cavity_method_functions")
from self_consistency_equation_functions import variable_fixed_parameters, \
    parameter_combinations
    
# %%
    
def p1_p2(model,
          p1,
          p2,
          fixed_parameters,
          subdirectory,
          save_method = 'v3',
          **kwargs):
    
    parameters = generate_parameters(p1,
                                     p2,
                                     fixed_parameters)
    
    CRM_across_parameter_space(parameters,
                               subdirectory,
                               [p1[0], p2[0]],
                               save_method=save_method,
                               model = model,
                               **kwargs)
                        
# %%

def generate_parameters(p1,
                        p2,
                        fixed_parameters):
    
    def transform(param_array,
                  param_name,
                  fixed_parameters,
                  no_parm = 2):
        
        match param_name:
            
            case 'exponent_b':
                
                return 10**param_array, ['b']
            
            case 'exponent_a':
                
                return 10**param_array, ['a']
        
            case 'sigma':
            
                return np.tile(param_array/np.sqrt(fixed_parameters['M']),
                               2).reshape((len(param_array), no_parm)), \
                        ['sigma_c', 'sigma_g']
        
            case 'mu':
            
                return np.tile(param_array/fixed_parameters['M'],
                               2).reshape((len(param_array), no_parm)), \
                        ['mu_c', 'mu_g']
                        
            case _:
                
                return [], []
                        
    p1_p2_combos = np.unique(parameter_combinations([p1[1],
                                                     p2[1]],
                                                    1),
                                    axis = 1).tolist()
    
    mod_params = []
    mod_names = []
    
    mod_params.append(np.array(p1_p2_combos))
    mod_names += [p1[0], p2[0]]
    
    for param_array, param_name in zip(p1_p2_combos, [p1[0], p2[0]]):
        
        params, names = transform(np.array(param_array),
                                  param_name,
                                  fixed_parameters)
        
        if names != []: 
            
            mod_params.append(params)
            mod_names += names
    
    variable_parameters = np.vstack(mod_params)
    
    fixed_parameters_mod = deepcopy(fixed_parameters)
    
    if 'mu' in fixed_parameters_mod:
        
        fixed_parameters_mod['mu_c'] = fixed_parameters_mod['mu']/fixed_parameters_mod['M']
        fixed_parameters_mod['mu_g'] = fixed_parameters_mod['mu_c']
        
    if 'sigma' in fixed_parameters_mod:
        
        fixed_parameters_mod['sigma_c'] = fixed_parameters_mod['sigma']/np.sqrt(fixed_parameters_mod['M'])
        fixed_parameters_mod['sigma_g'] = fixed_parameters_mod['sigma_c']

    # array of all parameter combinations
    parameters = variable_fixed_parameters(variable_parameters,
                                           fixed_parameters_mod,
                                           mod_names)
    
    return parameters

#############################################################################

# %%

bs = np.arange(-5, 0.5, 0.5)
resource_inhibitions = np.array([-5, 0.0])
rhos = np.arange(0.1, 1.1, 0.1)
sigma = 4.0
mu = 50
d = 1
o = 1
system_size = 150

# %%

p1_p2("Hybrid resource supply",
      ('exponent_b', [-5, 0.0]),
      ('exponent_a', resource_inhibitions),
      dict(mu = mu, sigma = sigma, rho = 0.8,
           d = d,
           o = o,
           M = system_size, S = system_size),
    "external_resource_stability/simulations/hybrid_influx_resourceinhibition",
    no_communities = 20,
    t_end = 1000,
    no_init_conds = 1)

# %%

p1_p2("Hybrid resource supply",
      ('exponent_b', bs),
      ('rho', rhos),
      dict(mu = mu, sigma = sigma,
           d = d,
           o = o,
           a = 1,
           M = system_size, S = system_size),
    "external_resource_stability/simulations/hybrid_influx_rho",
    no_communities = 20,
    t_end = 1000,
    no_init_conds = 1)

####################################################

