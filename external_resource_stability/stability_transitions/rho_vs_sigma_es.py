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

# %%

abspath = os.path.abspath(__file__)
file_directory_name = os.path.dirname(abspath)
os.chdir(file_directory_name)

sys.path.insert(0, file_directory_name.removesuffix("\\stability_transitions"))
from simulation_functions import CRM_across_parameter_space, \
    generate_simulation_df, le_pivot_r

sys.path.insert(0,  file_directory_name.removesuffix("\\external_resource_stability\\stability_transitions") + \
                "\\cavity_method_functions")
import self_consistency_equation_functions as sce

# %%

def rho_sigma(rho_range, sigma_range, n, fixed_parameters):
    
    parameters = generate_parameters(rho_range, sigma_range, n, fixed_parameters)
    
    CRM_across_parameter_space(parameters,
                               "external_resource_stability/simulations/rho_vs_sigma",
                               ['rho', 'sigma_M'])
                    
# %%

def generate_parameters(rho_range, sigma_range, n, fixed_parameters):
    
    rho_sigma_combos = np.unique(sce.parameter_combinations([rho_range,
                                                             sigma_range],
                                                            n),
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

rhos = np.linspace(0.1, 1, 10) # np.linspace(0, 1, 11)
sigmas = np.linspace(2, 5, 11)

# %%

rho_sigma(rhos, sigmas, 11,
          dict(mu_c = 160, mu_g = 160, d = 1, b = 1, o = 1, M = 150, S = 150))

# %%

df_simulation = generate_simulation_df("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/" \
                                       + 'external_resource_stability/simulations/rho_vs_sigma')
    
# %%

df_simulation.groupby(['rho', 'sigma_c'])['phi_N'].apply('mean')
    
# %%

le_pivot_r(df_simulation, index = 'sigma_c', columns = 'rho')
