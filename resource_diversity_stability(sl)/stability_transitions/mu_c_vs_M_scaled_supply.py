# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 10:31:30 2025

@author: jamil
"""

# -*- coding: utf-8 -*-
"""
Created on Wed May  7 19:27:31 2025

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
    le_pivot_r, generic_heatmaps

sys.path.insert(0, 'C:/Users/jamil/Documents/PhD/Github Projects/Ecological-Dynamics-and-Community-Selection/Ecological-Dynamics/Consumer-Resource Models/cavity_method_functions')
import self_consistency_equation_functions as sce

# %%

def M_effect_fixed_C_fixed_supply(M_range, mu_C_range, sigma_C, b, n,
                                  fixed_parameters):
    
    parameters = generate_parameters_M_C(M_range, mu_C_range, sigma_C, b, n,
                                         fixed_parameters)
    
    CRM_across_parameter_space(parameters, 'finite_effects_fixed_C_fixed_supply_final',
                               ['M', 'mu_c'])
                    
# %%

def generate_parameters_M_C(M_range, mu_C_range, sigma_C, b, n,
                            fixed_parameters):
    
    M_mu_C_combinations = np.unique(sce.parameter_combinations([M_range,
                                                                mu_C_range],
                                                               n),
                                    axis = 1)
    
    variable_parameters = np.vstack([M_mu_C_combinations[0, :]/fixed_parameters['gamma'],
                                     M_mu_C_combinations[0, :],
                                     M_mu_C_combinations[1, :]/M_mu_C_combinations[0, :],
                                     np.repeat(sigma_C, M_mu_C_combinations.shape[1])/np.sqrt(M_mu_C_combinations[0, :]),
                                     np.repeat(b, M_mu_C_combinations.shape[1])/M_mu_C_combinations[0, :]])

    # array of all parameter combinations
    parameters = sce.variable_fixed_parameters(variable_parameters,
                                               fixed_parameters,
                                               ['S', 'M', 'mu_c', 'sigma_c', 'b'])
    
    for parms in parameters:
        
        parms['S'] = np.int32(parms['S'])
        parms['M'] = np.int32(parms['M'])
    
    return parameters
    
# %%

resource_pool_sizes = np.arange(50, 275, 25)

# %%

M_effect_fixed_C_fixed_supply(resource_pool_sizes, np.array([100, 250]), 1.6, 150,
                              11, {'mu_y': 1, 'sigma_y' : 1.6/np.sqrt(150),
                                   'd' : 1, 'gamma' : 1})

# %%

df_mu_c_b_M = generate_simulation_df("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/" \
                                     + 'finite_effects_fixed_C_fixed_supply_final')

# %%

fig, axs = generic_heatmaps(df_mu_c_b_M,
                            'M', 'mu_c', 
                           'resource pool size, ' + r'$M$',
                           'average total consumption  rate, ' + r'$\mu_c$',
                            ['Max. lyapunov exponent'], 'Purples_r',
                            'Large resource pools still stabilise communites\n' + \
                            'under constant (total) resource flux',
                            (1, 1), (5.5, 5.5),
                            pivot_functions = {'Max. lyapunov exponent' : le_pivot_r},
                            specify_min_max={'Max. lyapunov exponent' : [0,1]})

axs.set_xticks(np.arange(0.5, len(resource_pool_sizes) + 0.5, 2),
                          labels = resource_pool_sizes[::2], fontsize = 14)

cbar = axs.collections[0].colorbar
cbar.set_label(label = 'Proportion of simulations with stable dynamics',
               size = '14')
cbar.ax.tick_params(labelsize = 12)

plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/self_limit_fixed_C_fixed_supply_M.png",
            bbox_inches='tight', dpi = 400)
plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/self_limit_fixed_C_fixed_supply_M.svg",
            bbox_inches='tight')

plt.show()

