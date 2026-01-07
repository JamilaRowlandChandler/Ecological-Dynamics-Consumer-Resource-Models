# -*- coding: utf-8 -*-
"""
Created on Fri May  9 15:32:30 2025

@author: jamil
"""

import numpy as np
import pandas as pd
import sys
import os
from matplotlib import pyplot as plt

os.chdir("C:/Users/jamil/Documents/PhD/GitHub projects/Ecological-Dynamics-and-Community-Selection/" + \
         "Ecological-Dynamics/Consumer-Resource Models/resource_diversity_stability(sl)/" + \
             "stability_transitions")
    
sys.path.insert(0, "C:/Users/jamil/Documents/PhD/GitHub projects/Ecological-Dynamics-and-Community-Selection/" + \
                    "Ecological-Dynamics/Consumer-Resource Models/resource_diversity_stability(sl)")
from simulation_functions import CRM_across_parameter_space, generate_simulation_df, \
    le_pivot, generic_heatmaps

sys.path.insert(0, 'C:/Users/jamil/Documents/PhD/Github Projects/Ecological-Dynamics-and-Community-Selection/Ecological-Dynamics/Consumer-Resource Models/cavity_method_functions')
import self_consistency_equation_functions as sce

# %%

def M_effect_mu_y(M_range, mu_y_range, mu_C, sigma_C, n, fixed_parameters):
    
    parameters = generate_parameters_M_mu_y(M_range, mu_y_range, mu_C, sigma_C,
                                            n, fixed_parameters)
    
    #CRM_across_parameter_space(parameters, 'finite_effects_fixed_C_mu_y_final',
    CRM_across_parameter_space(parameters, 'finite_effects_fixed_C_mu_y_final_2',
                               ['M', 'mu_y'])
                
# %%

def generate_parameters_M_mu_y(M_range, mu_y_range, mu_C, sigma_C, n,
                               fixed_parameters):
    
    M_mu_C_combinations = np.unique(sce.parameter_combinations([M_range,
                                                                mu_y_range],
                                                               n),
                                    axis = 1)
    
    variable_parameters = np.vstack([M_mu_C_combinations[0, :]/fixed_parameters['gamma'],
                                     M_mu_C_combinations,
                                     np.repeat(mu_C, M_mu_C_combinations.shape[1])/M_mu_C_combinations[0, :],
                                     np.repeat(sigma_C, M_mu_C_combinations.shape[1])/np.sqrt(M_mu_C_combinations[0, :])])

    # array of all parameter combinations
    parameters = sce.variable_fixed_parameters(variable_parameters,
                                               fixed_parameters,
                                               ['S', 'M', 'mu_y', 'mu_c', 'sigma_c'])
    
    for parms in parameters:
        
        parms['S'] = np.int32(parms['S'])
        parms['M'] = np.int32(parms['M'])
    
    return parameters
    
# %%

resource_pool_sizes = np.arange(50, 275, 25)

# %%

M_effect_mu_y(resource_pool_sizes, np.array([0.25, 1]), 130, 1.6, 7,
              {'sigma_y' : 1.6/np.sqrt(150), 'b' : 1, 'd' : 1, 'gamma' : 1})

# %%

df_mu_y_M = generate_simulation_df("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/" \
                                   + 'finite_effects_fixed_C_mu_y_final_2')
    
# %%

fig, axs = generic_heatmaps(df_mu_y_M,
                            'no_resources', 'mu_y', 
                           'resource pool size, ' + r'$M$',
                           'average resource use efficiency, ' + r'$\mu_y$',
                            ['Max. lyapunov exponent'], 'Purples',
                            '',
                            (1, 1), (6.5, 4),
                            pivot_functions = {'Max. lyapunov exponent' : le_pivot},
                            specify_min_max={'Max. lyapunov exponent' : [0,1]})

axs.set_xticks(np.arange(0.5, len(resource_pool_sizes) + 0.5, 1),
                          labels = resource_pool_sizes, fontsize = 14)

cbar = axs.collections[0].colorbar
cbar.set_label(label = 'Proportion of simulations with max. LE ' + r'$> 0.00$',
               size = '14')
cbar.ax.tick_params(labelsize = 12)

plt.show()
