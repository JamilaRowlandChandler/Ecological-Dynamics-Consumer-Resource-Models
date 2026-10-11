# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 21:30:16 2026

@author: jamil
"""


# %%

import numpy as np
import pandas as pd
import os
import sys

from matplotlib import pyplot as plt

abspath = os.path.abspath(__file__)
file_directory_name = os.path.dirname(abspath)
os.chdir(file_directory_name)

os.chdir("C:/Users/jamil/Documents/PhD/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models/" + \
         "resource_diversity_stability(sl)/stability_transitions")
    
sys.path.insert(0, "C:/Users/jamil/Documents/PhD/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models/" + \
                    "resource_diversity_stability(sl)")
from complete_simulation_functions import le_pivot_r, generic_heatmaps

# %%

directory = "C:/Users/jamil/Documents/PhD/Data/resource_diversity_stability/simulations/M_vs_mu_c_all_pos"

df_simulation = pd.concat([pd.read_csv(directory + "/" + file,
                                       index_col=0)
                           for file in os.listdir(directory)],
                          ignore_index=True)

df_simulation = df_simulation[df_simulation['mu_c'] <= 205]
    
# %%

resource_pool_sizes = np.unique(df_simulation['M'])
mu_cs = np.unique(df_simulation['mu_c'])
    
fig, axs = generic_heatmaps(df_simulation,
                            'M', 'mu_c', 
                            '',
                            '',
                            ['Max. lyapunov exponent'], 
                            'Purples_r',
                            '',
                            (1, 1),
                            (3, 2.5),
                            pivot_functions = {'Max. lyapunov exponent' : le_pivot_r},
                            specify_min_max={'Max. lyapunov exponent' : [0,1]})

axs.set_xlabel('resource pool size, ' + r'$M$',
               weight="bold",
               fontsize=10)
axs.set_xticks(np.arange(0.5, len(resource_pool_sizes) + 0.5, 2),
               labels = resource_pool_sizes[::2],
               fontsize = 8)

axs.set_ylabel('avg. total consumption rate, ' + r'$\mu_c$',
               weight="bold",
               fontsize=10)
axs.set_yticks(np.arange(0.5, len(mu_cs) + 0.5, 2),
               labels = mu_cs[::2].astype(int),
               fontsize = 8)

cbar = axs.collections[0].colorbar
cbar.set_label(label = 'Prob. stability',
               size = '9',
               weight = 'bold')
cbar.ax.tick_params(labelsize = 8)

fig.suptitle("Large resource pools stabilise communities with\n" + \
             "fixed metabolic budgets",
             fontsize = 11,
             weight = "bold")
    
plt.savefig("C:/Users/jamil/Documents/PhD/Figures/resource_diversity_stability/self_limit_M_vs_mu_c_all_pos.png",
            bbox_inches='tight')
plt.savefig("C:/Users/jamil/Documents/PhD/Figures/resource_diversity_stability/self_limit_M_vs_mu_c_all_pos.svg",
            bbox_inches='tight')

plt.show()