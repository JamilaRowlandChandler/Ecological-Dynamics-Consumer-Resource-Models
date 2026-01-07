# -*- coding: utf-8 -*-
"""
Created on Sat Oct  4 00:51:27 2025

@author: jamil
"""

import numpy as np
import pandas as pd
import sys
import os
from tqdm import tqdm
from matplotlib import pyplot as plt

os.chdir("C:/Users/jamil/Documents/PhD/GitHub projects/Ecological-Dynamics-and-Community-Selection/" + \
         "Ecological-Dynamics/Consumer-Resource Models/resource_diversity_stability(sl)/" + \
             "stability_transitions")
    
sys.path.insert(0, "C:/Users/jamil/Documents/PhD/GitHub projects/Ecological-Dynamics-and-Community-Selection/" + \
                    "Ecological-Dynamics/Consumer-Resource Models/resource_diversity_stability(sl)")
from simulation_functions import pickle_dump, generic_heatmaps, \
    generate_simulation_df, le_pivot_r

sys.path.insert(0, 'C:/Users/jamil/Documents/PhD/Github Projects/Ecological-Dynamics-and-Community-Selection/Ecological-Dynamics/Consumer-Resource Models/consumer_resource_modules')
from community_level_properties import max_le

# %%

def recalculate_max_le(old_directory, new_directory,
                       le_kwargs = dict(T = 1000, perturbation = 1e-6)):
    
    full_old_directory = "C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/" \
                           + old_directory
    
    full_new_directory = "C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/" \
                            + new_directory

    if not os.path.exists(full_new_directory):
        
        os.makedirs(full_new_directory)
        
        
    for file in tqdm(os.listdir(full_old_directory), position = 0, leave = True):
        
        communities = pd.read_pickle(full_old_directory + "/" + file)
        
        for i, community in enumerate(communities):
            
            community.lyapunov_exponent = max_le(community,
                                                 community.ODE_sols[0].y[:, -1],
                                                 **le_kwargs)
       
        pickle_dump(full_new_directory + "/" + file, communities)
        
# %%

def le_test(path,
            le_kwargs = dict(T = 1000, perturbation = 1e-6)):
    
    communities = pd.read_pickle("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/" \
                                 + path)
    
    for i, community in enumerate(communities):
        
        print(i)
        
        community.lyapunov_exponent = max_le(community,
                                             community.ODE_sols[0].y[:, -1],
                                             **le_kwargs)
        
        
    return np.mean([community.lyapunov_exponent for community in communities])


# %%

recalculate_max_le("finite_effects_fixed_C_final", "M_vs_mu_c_new_le_3")

recalculate_max_le("finite_effects_sigma_c_final", "M_vs_sigma_c_new_le_3")

recalculate_max_le("finite_effect_sigma_y_final", "M_vs_sigma_y_new_le_3")

# %%

le_test("finite_effect_sigma_y_final/simulations_250_0.05.pkl")
        #le_kwargs=dict(T = 1500, perturbation = 1e-6))

# %%

df_mu_c_M = generate_simulation_df("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/" \
                                       + 'M_vs_mu_c_new_le_3')
    
# %%
resource_pool_sizes = np.unique(df_mu_c_M['M'])

fig, axs = generic_heatmaps(df_mu_c_M,
                            'M', 'mu_c', 
                           'resource pool size, ' + r'$M$',
                           'average total consumption  rate, ' + r'$\mu_c$',
                            ['Max. lyapunov exponent'], 'Purples_r',
                            '',
                            (1, 1), (4.5, 4),
                            pivot_functions = {'Max. lyapunov exponent' : le_pivot_r},
                            specify_min_max={'Max. lyapunov exponent' : [0,1]})

axs.set_xticks(np.arange(0.5, len(resource_pool_sizes) + 0.5, 1),
                          labels = resource_pool_sizes, fontsize = 14)

cbar = axs.collections[0].colorbar
cbar.set_label(label = 'Proportion of simulations with max. LE ' + r'$< 0$',
               size = '14')
cbar.ax.tick_params(labelsize = 12)

plt.show()

                                 
                                   