# -*- coding: utf-8 -*-
"""
Created on Fri Nov 14 14:24:01 2025

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

def rho_sigma(model,
              rho_range,
              sigma_range,
              fixed_parameters,
              subdirectory,
              save_method = 'v3',
              **kwargs):
    
    parameters = generate_parameters(rho_range, sigma_range, fixed_parameters)
    
    CRM_across_parameter_space(parameters,
                               subdirectory,
                               ['rho', 'sigma_M'],
                               save_method=save_method,
                               model = model,
                               **kwargs)
                    
# %%

def generate_parameters(rho_range, sigma_range, fixed_parameters):
    
    rho_sigma_combos = np.unique(parameter_combinations([rho_range,
                                                         sigma_range],
                                                        1),
                                    axis = 1)
    
    variable_parameters = np.vstack([rho_sigma_combos,
                                     rho_sigma_combos[1, :]/np.sqrt(fixed_parameters['M']),
                                     rho_sigma_combos[1, :]/np.sqrt(fixed_parameters['M'])])
    
    fixed_parameters_mod = deepcopy(fixed_parameters)
    
    fixed_parameters_mod['mu_c'] *= 1/fixed_parameters_mod['M']
    fixed_parameters_mod['mu_g'] *= 1/fixed_parameters_mod['M']

    # array of all parameter combinations
    parameters = variable_fixed_parameters(variable_parameters,
                                           fixed_parameters_mod,
                                           ['rho', 'sigma_M',
                                            'sigma_c', 'sigma_g'])
    
    return parameters



# %%

rhos = np.arange(0.1, 1.1, 0.1)
sigmas = np.arange(2, 13, 1)
mu = 50
d = 1
b = 1
o = 1
a = 1
system_size = 150

# %%

rho_sigma("Externally-supplied resources",
          rhos,
          sigmas,
          dict(mu_c = mu, mu_g = mu,
               d = d, b = b, o = o,
               M = system_size, S = system_size),
          "external_resource_stability/simulations/rho_sigma_mu50_es",
          no_communities = 20,
          t_end = 1000,
          no_init_conds = 1)

rho_sigma("Self-limiting resource supply",
          rhos,
          sigmas,
          dict(mu_c = mu, mu_g = mu,
               d = d, b = b,
               M = system_size, S = system_size),
          "external_resource_stability/simulations/rho_sigma_mu50_sl",
          no_communities = 20,
          t_end = 1000,
          no_init_conds = 1)

rho_sigma("Hybrid resource supply",
          rhos,
          sigmas,
          dict(mu_c = mu, mu_g = mu,
               d = d, b = b, o = o, a = a,
               M = system_size, S = system_size),
          "external_resource_stability/simulations/rho_sigma_mu50_h",
          no_communities = 20,
          t_end = 1000,
          no_init_conds = 1)

rho_sigma("Hybrid resource supply",
          np.arange(0.7, 1.1, 0.1), #rhos,
          sigmas,
          dict(mu_c = mu, mu_g = mu,
               d = d, b = 0.001, o = o, a = a,
               M = system_size, S = system_size),
          "external_resource_stability/simulations/rho_sigma_mu50_h_b0001",
          no_communities = 20,
          t_end = 1000,
          no_init_conds = 1)

# %%

################################################

rho_sigma("Hybrid resource supply",
          [0.6], # np.arange(0.7, 1.0, 0.1), #rhos,
          np.arange(5.0, 13.0, 1.0), #sigmas,
          dict(mu_c = mu, mu_g = mu,
               d = d, b = 0, o = o, a = 10**(-5),
               M = system_size, S = system_size),
          "external_resource_stability/simulations/rho_sigma_noinflux",
          no_communities = 20,
          t_end = 1000,
          no_init_conds = 1)

# %%

rho_sigma("Hybrid resource supply",
          np.arange(0.7, 1.0, 0.1), #rhos,
          sigmas,
          dict(mu_c = mu, mu_g = mu,
               d = d, b = 0.001, o = o, a = 10**(-5),
               M = system_size, S = system_size),
          "external_resource_stability/simulations/rho_sigma_smallinflux",
          no_communities = 20,
          t_end = 1000,
          no_init_conds = 1)

rho_sigma("Hybrid resource supply",
          rhos,
          sigmas,
          dict(mu_c = mu, mu_g = mu,
               d = d, b = b, o = o, a = 10**(-5),
               M = system_size, S = system_size),
          "external_resource_stability/simulations/rho_sigma_largeinflux",
          no_communities = 20,
          t_end = 1000,
          no_init_conds = 1)

