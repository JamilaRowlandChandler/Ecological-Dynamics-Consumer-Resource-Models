# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 2026

@author: jamil

Rough mu_c vs M sweep for MP_CRM with growth_saturation=True (log-corrected
growth: R_alpha*[w_alpha - w_beta - log(R_alpha/R_beta)] instead of the
original R_alpha*(w_alpha-w_beta); consumption/production unchanged), 5
communities per M/mu_c combination. Structured like mu_c_vs_M.py (same
fixed parameters, gated=True default network), but does not edit that file.
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

def M_effect_fixed_C(M_range, mu_C_range, sigma_C, n, fixed_parameters, save_dir, no_communities):

    parameters = generate_parameters(M_range, mu_C_range, sigma_C, n, fixed_parameters)

    CRM_across_parameter_space(parameters,
                               save_dir,
                               ['M', 'mu_c'],
                               model = "Metabolic pathways",
                               save_method = 'v3',
                               no_communities = no_communities,
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

# same fixed parameters as mu_c_vs_M.py's o=0 sweep, but with
# growth_saturation=True (gated=True default network, as in that script)
fixed_parameters = dict(mu_y = 1, sigma_y = 0.13,
                        d = 1, o = 0, b = 1, A = 1, p = 1,
                        network_method = 'step', p_s = 1,
                        growth_saturation = True,
                        gamma = 1)

save_dir = "resource_diversity_stability_crossfeeding/mu_c_vs_M_growth_saturation"

M_effect_fixed_C(M_range, mu_C_range, sigma_C, n, fixed_parameters, save_dir, no_communities=5)

# %%

full_location = "C:/Users/jamil/Documents/PhD/Data/" + \
    "resource_diversity_stability_crossfeeding/mu_c_vs_M_growth_saturation"

df = pd.concat([pd.read_csv(full_location + "/" + file, index_col=False)
               for file in os.listdir(full_location)],
               axis = 0, ignore_index = True)

print(df[['M', 'mu_c', 'Max. lyapunov exponent', 'Divergence measure']].sort_values(['M', 'mu_c']))
