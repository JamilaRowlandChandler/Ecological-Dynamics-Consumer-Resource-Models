# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 22:08:02 2026

@author: jamil
"""

# %%

import numpy as np
import pandas as pd
import seaborn as sns
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
from complete_simulation_functions import le_pivot_r

# %%

shared_directory = "C:/Users/jamil/Documents/PhD/Data/resource_diversity_stability/"

directories = [shared_directory + "simulations/M_vs_mu_c_all_pos",
               shared_directory + "simulations/M_vs_mu_c_uniform",
               shared_directory + "simulations/M_vs_mu_c_fixed_budget"]

dfs = [pd.concat([pd.read_csv(directory + "/" + file,
                              index_col=0)
                  for file in os.listdir(directory)],
                 ignore_index=True)
       for directory in directories]

dfs = dfs + [pd.read_csv(shared_directory + "/Repository Data/simulation_dataframes/M_vs_mu_c_consumer_inhibition.csv",
                         index_col=0),
             pd.read_csv(shared_directory + "/Repository Data/simulation_dataframes/M_vs_mu_c_fixed_supply.csv",
                                      index_col=0)]
   
# %%

def Stability_Plots(dfs):
    
    def stability_plot(df,
                       ax):
    
        df = df.loc[df['mu_c'] < 220, :]
        
        resource_pool_sizes = np.unique(df['M'])
        mu_cs = np.unique(df['mu_c'])
        
        ######################## Phase diagram ######################################
        
        # Simulation data
        
        stability_pivot = le_pivot_r(df,
                                     columns = 'M',
                                     index = 'mu_c')[0]
        
        sns.set_style('ticks')
    
        subfig = sns.heatmap(stability_pivot,
                             ax = ax,
                             vmin = 0, vmax = 1,
                             cbar = True,
                             cmap = 'Purples_r')
        
        subfig.axhline(0, 0, 1,
                       color = 'black',
                       linewidth = 2)
        subfig.axhline(stability_pivot.shape[0],
                       0, 1,
                       color = 'black',
                       linewidth = 2)
        subfig.axvline(0, 0, 1,
                       color = 'black',
                       linewidth = 2)
        subfig.axvline(stability_pivot.shape[1],
                       0, 1,
                       color = 'black',
                       linewidth = 2)
        
        ax.set_xticks(np.arange(0.5, len(resource_pool_sizes) + 0.5, 2),
                      labels = resource_pool_sizes[::2],
                      fontsize = 8,
                      rotation = 0)
        
        ax.set_yticks(np.arange(0.5, len(mu_cs) + 0.5, 2),
                      labels = np.int32(mu_cs[::2]),
                      fontsize = 8,
                      rotation = 0)
        
        ax.set_xlabel('resource pool size, ' + r'$M$',
                      fontsize = 10,
                      weight = 'bold')
        ax.set_ylabel('avg. total consumption coeff., ' + r'$\mu_c$',
                      fontsize = 10,
                      weight = 'bold')

        ax.invert_yaxis()
             
        cbar = ax.collections[0].colorbar
        cbar.set_label(label = 'Probability(stability)',
                       size = '8',
                       horizontalalignment = 'center', 
                       verticalalignment = 'top')
        cbar.ax.tick_params(labelsize = 10)
    
    fig, axs = plt.subplots(2, 3,
                            figsize=(9.5, 4.5),
                            layout="constrained")
    
    for df, ax in zip(dfs, axs.flatten()):
        
        stability_plot(df, ax)
        
    plt.savefig("C:/Users/jamil/Documents/PhD/Figures/resource_diversity_stability/M_vs_mu_c_robustness.png",
                bbox_inches='tight')
    plt.savefig("C:/Users/jamil/Documents/PhD/Figures/resource_diversity_stability/M_vs_mu_c_robustness.svg",
                bbox_inches='tight')
        
    plt.show()

Stability_Plots(dfs)
