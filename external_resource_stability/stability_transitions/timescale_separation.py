# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 23:45:21 2026

@author: jamil
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 23:34:50 2026

@author: jamil
"""

import numpy as np
import sys
import os
from copy import deepcopy

# %%

abspath = os.path.abspath(__file__)
file_directory_name = os.path.dirname(abspath)
os.chdir(file_directory_name)

sys.path.insert(0, file_directory_name.removesuffix("\\stability_transitions"))
from complete_simulation_functions import CRM_across_parameter_space, le_pivot_r

sys.path.insert(0,  file_directory_name.removesuffix("\\external_resource_stability\\stability_transitions") + \
                "\\cavity_method_functions")
from self_consistency_equation_functions import variable_fixed_parameters, \
    parameter_combinations

# %%

def rho_epsilon(model,
                rho_range,
                epsilons,
                fixed_parameters,
                subdirectory,
                save_method = 'v3',
                **kwargs):
    
    parameters = generate_parameters(rho_range,
                                     epsilons,
                                     fixed_parameters)
    
    CRM_across_parameter_space(parameters,
                               subdirectory,
                               ['rho',
                                'epsilon_exponent'],
                               save_method=save_method,
                               model = model,
                               **kwargs)
                    
# %%

def generate_parameters(rho_range, epsilons, fixed_parameters):
    
    rho_eps_combos = np.unique(parameter_combinations([rho_range,
                                                       epsilons],
                                                        1),
                                    axis = 1)
    
    variable_parameters = np.vstack([rho_eps_combos,
                                     np.round(np.log10(rho_eps_combos[1, :]), 1)])
    
    fixed_parameters_mod = deepcopy(fixed_parameters)
    
    fixed_parameters_mod['mu_c'] *= 1/fixed_parameters_mod['M']
    fixed_parameters_mod['mu_g'] *= 1/fixed_parameters_mod['M']
    fixed_parameters_mod['sigma_c'] *= 1/np.sqrt(fixed_parameters_mod['M'])
    fixed_parameters_mod['sigma_g'] *= 1/np.sqrt(fixed_parameters_mod['M'])

    # array of all parameter combinations
    parameters = variable_fixed_parameters(variable_parameters,
                                           fixed_parameters_mod,
                                           ['rho',
                                            'epsilon',
                                            'epsilon_exponent'])
    
    return parameters



# %%

rhos = np.arange(0.1, 1.1, 0.1)
epsilons = 10.0**np.arange(-4.0, 0.0, 1.0)
mu = 50
sigma = 6
d = 1
b = 1
o = 1
system_size = 150

# %%

rho_epsilon("Externally-supplied resources",
            rhos,
            epsilons,
            dict(mu_c = mu, mu_g = mu,
                 sigma_c = sigma, sigma_g = sigma,
                 d = d, b = b, o = o,
                 M = system_size, S = system_size),
          "external_resource_stability/simulations/rho_timescale_es",
          no_communities = 20,
          t_end = 1000,
          no_init_conds = 1)