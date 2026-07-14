# -*- coding: utf-8 -*-
"""
Created on Wed May  7 19:27:31 2025

@author: jamil

Investigates how resource pool size (M = no_resources) affects stability in
the metabolic pathway model (MP_CRM), for a range of mean consumption rates
(mu_c). Structured like
"C:/Users/jamil/Documents/PhD/Code Repositories/CRM-Resource-diversity-vs-Stability/Simulation codes/mu_c_vs_M.py",
adapted to call MP_CRM via simulation_functions_unified.py instead.
"""

import numpy as np
import sys
import os

# %%

abspath = os.path.abspath(__file__)
file_directory_name = os.path.dirname(abspath)
os.chdir(file_directory_name)

# %%

# simulation_functions_unified.py and self_consistency_equation_functions.py
# both live one level up, in resource_diversity_stability_crossfeeding/
sys.path.insert(0, 'C:/Users/jamil/Documents/PhD/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models/resource_diversity_stability_crossfeeding')
from simulation_functions_unified import CRM_across_parameter_space

import self_consistency_equation_functions as sce

# %%

def M_effect_fixed_C(M_range, mu_C_range, sigma_C, n, fixed_parameters):

    parameters = generate_parameters(M_range, mu_C_range, sigma_C, n,
                                     fixed_parameters)

    CRM_across_parameter_space(parameters,
                               "resource_diversity_stability_crossfeeding/mu_c_vs_M",
                               ['M', 'mu_c'],
                               model = "Metabolic pathways",
                               save_method = 'v3')

# %%

def generate_parameters(M_range, mu_C_range, sigma_C, n,
                        fixed_parameters):

    # M-scaling: mu_c = mu_C/M, sigma_c = sigma_C/sqrt(M) - keep this scaling,
    # it's what lets mu_C/sigma_C be compared meaningfully across resource
    # pool sizes
    M_mu_C_combinations = np.unique(sce.parameter_combinations([M_range,
                                                                mu_C_range],
                                                               n),
                                    axis = 1)

    variable_parameters = np.vstack([M_mu_C_combinations[0, :]/fixed_parameters['gamma'],
                                     M_mu_C_combinations[0, :],
                                     M_mu_C_combinations[1, :]/M_mu_C_combinations[0, :],
                                     np.repeat(sigma_C, M_mu_C_combinations.shape[1])/np.sqrt(M_mu_C_combinations[0, :])])

    # array of all parameter combinations
    parameters = sce.variable_fixed_parameters(variable_parameters,
                                               fixed_parameters,
                                               ['S', 'M', 'mu_c', 'sigma_c'])

    for parms in parameters:

        parms['S'] = np.int32(parms['S'])
        parms['M'] = np.int32(parms['M'])

    return parameters

# %%

resource_pool_sizes = np.arange(50, 275, 25)

# fixed (not M-scaled) parameters for MP_CRM. mu_y/sigma_y are the
# growth-yield coupling (growth_consumption_rates(method='growth function of
# consumption', ...)); d/o/b/A are MP_CRM's death, resource-outflux, resource
# self-decay and resource self-inhibition rates (all 'constant' since given
# as bare values, matching MP_CRM's own defaults where the spec didn't call
# out a different value); p is the resource production rate; network_method/
# p_s configure the structured metabolic network (a link from resource alpha
# to beta exists with probability p_s whenever w_alpha > w_beta, and never
# otherwise); gamma = M/S sets the species:resource ratio (gamma=1 -> S=M).
fixed_parameters = dict(mu_y = 1, sigma_y = 0.13,
                        d = 1, o = 1, b = 1, A = 1, p = 1,
                        network_method = 'step', p_s = 1,
                        gamma = 1)

# %%

# NOTE: (mu_C_range, sigma_C, n) below are a starting point, not yet the
# fully-explored transition range - MP_CRM's cross-feeding dynamics behave
# differently from the self-limiting models the template was written for,
# so this still needs a proper sweep to find where the stability transition
# actually sits before treating these numbers as final.
M_effect_fixed_C(resource_pool_sizes, (100, 250), 1.6,
                 11, fixed_parameters)
