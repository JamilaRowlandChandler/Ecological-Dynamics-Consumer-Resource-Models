# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 12:00:00 2026

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
from complete_simulation_functions import CRM_across_parameter_space

sys.path.insert(0,  file_directory_name.removesuffix("\\external_resource_stability\\stability_transitions") + \
                "\\cavity_method_functions")
import self_consistency_equation_functions as sce

# %%

def rho_sigma(rho_range,
              sigma_range,
              fixed_parameters,
              subdirectory,
              **kwargs):
    
    parameters = generate_parameters(rho_range, sigma_range, fixed_parameters)
    
    CRM_across_parameter_space(parameters,
                               subdirectory,
                               ['rho', 'sigma_M'],
                               model = 'Leached biomolecules',
                               save_method = 'v3',
                               **kwargs)
                    
# %%

def generate_parameters(rho_range, sigma_range, fixed_parameters):
    
    rho_sigma_combos = np.unique(sce.parameter_combinations([rho_range,
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
    parameters = sce.variable_fixed_parameters(variable_parameters,
                                               fixed_parameters_mod,
                                               ['rho', 'sigma_M',
                                                'sigma_c', 'sigma_g'])
    
    return parameters

# %%

rhos = np.arange(0.1, 1.1, 0.1)
sigmas = np.arange(2, 13, 1)

# %%

# c = expected number of resources each consumer produces
fixed_parameters = dict(mu_c = 3, mu_g = 3, d = 1, l = 1, c = 10,
                        M = 100, S = 300)

rho_sigma(rhos, sigmas,
          fixed_parameters,
          "LB_CRM_rho_vs_sigma")
