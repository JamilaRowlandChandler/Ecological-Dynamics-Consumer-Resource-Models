# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 2026

@author: jamil

Same rough mu_c vs M sweep as mu_c_vs_M_gated_false.py (gated=False,
shared_network=False vs True), but with production rate p=0.999 instead of
p=1 - checking whether even a tiny (0.1%) production leak is enough to
restore stability across the grid, given the exact p=1 knife-edge found
for a single (M=250, mu_C=250, shared_network=True) case.
"""

import numpy as np
import sys
import os
import pandas as pd

# %%

abspath = os.path.abspath(__file__)
file_directory_name = os.path.dirname(abspath)
os.chdir(file_directory_name)

sys.path.insert(0, 'C:/Users/jamil/Documents/PhD/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models/resource_diversity_stability_crossfeeding')
from simulation_functions_unified import CRM_across_parameter_space

import self_consistency_equation_functions as sce

# %%

def M_effect_fixed_C(M_range, mu_C_range, sigma_C, n, fixed_parameters, save_dir):

    parameters = generate_parameters(M_range, mu_C_range, sigma_C, n, fixed_parameters)

    CRM_across_parameter_space(parameters,
                               save_dir,
                               ['M', 'mu_c'],
                               model = "Metabolic pathways",
                               save_method = 'v3',
                               no_communities = 2,
                               t_end = 7000,
                               no_init_conds = 1)

# %%

def generate_parameters(M_range, mu_C_range, sigma_C, n, fixed_parameters):

    M_mu_C_combinations = np.unique(sce.parameter_combinations([M_range,
                                                                mu_C_range],
                                                               n),
                                    axis = 1)

    variable_parameters = np.vstack([M_mu_C_combinations[0, :]/fixed_parameters['gamma'],
                                     M_mu_C_combinations[0, :],
                                     M_mu_C_combinations[1, :]/M_mu_C_combinations[0, :],
                                     np.repeat(sigma_C, M_mu_C_combinations.shape[1])/np.sqrt(M_mu_C_combinations[0, :])])

    parameters = sce.variable_fixed_parameters(variable_parameters,
                                               fixed_parameters,
                                               ['S', 'M', 'mu_c', 'sigma_c'])

    for parms in parameters:

        parms['S'] = np.int32(parms['S'])
        parms['M'] = np.int32(parms['M'])

    return parameters

# %%

M_range = np.array([75, 150, 250])
mu_C_range = (100, 250)
sigma_C = 1.6
n = 3

# same as mu_c_vs_M_gated_false.py, but p=0.999 instead of p=1
base_fixed_parameters = dict(mu_y = 1, sigma_y = 0.13,
                             d = 1, o = 0, b = 1, A = 1, p = 0.999,
                             network_method = 'step', p_s = 1,
                             gated = False,
                             gamma = 1)

for shared_network in [False, True]:

    fixed_parameters = dict(base_fixed_parameters, shared_network = shared_network)

    save_dir = f"resource_diversity_stability_crossfeeding/mu_c_vs_M_gated_false_p0999_shared_{shared_network}"

    M_effect_fixed_C(M_range, mu_C_range, sigma_C, n, fixed_parameters, save_dir)

# %%

for shared_network in [False, True]:

    full_location = "C:/Users/jamil/Documents/PhD/Data/" + \
        f"resource_diversity_stability_crossfeeding/mu_c_vs_M_gated_false_p0999_shared_{shared_network}"

    df = pd.concat([pd.read_csv(full_location + "/" + file, index_col=False)
                   for file in os.listdir(full_location)],
                   axis = 0, ignore_index = True)

    print(f"--- shared_network={shared_network}, p=0.999 ---")
    print(df[['M', 'mu_c', 'Max. lyapunov exponent', 'Divergence measure']].sort_values(['M', 'mu_c']))
