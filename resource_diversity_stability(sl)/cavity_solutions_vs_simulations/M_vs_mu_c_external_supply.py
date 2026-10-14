# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 12:51:27 2026

@author: jamil
"""

# %%

import numpy as np
import pandas as pd
import os
import sys
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patheffects as patheffects
import seaborn as sns

abspath = os.path.abspath(__file__)
file_directory_name = os.path.dirname(abspath)
os.chdir(file_directory_name)

os.chdir("C:/Users/jamil/Documents/PhD/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models/" + \
         "resource_diversity_stability(sl)/stability_transitions")
    
sys.path.insert(0, "C:/Users/jamil/Documents/PhD/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models/" + \
                    "resource_diversity_stability(sl)")
from complete_simulation_functions import le_pivot_r, generic_heatmaps

# %%

shared_directory = "C:/Users/jamil/Documents/PhD/Data/resource_diversity_stability/simulations/"

directories = [shared_directory + "M_vs_mu_c_external_supply",
               shared_directory + "M_vs_mu_c_external_supply_more_spec",
               shared_directory + "M_vs_mu_c_external_supply_mu650"]

df_simulation = pd.concat(sum([[pd.read_csv(directory + "/" + file,
                                            index_col=0)
                                for file in os.listdir(directory)]
                               for directory in directories], []),
                          ignore_index=True)

# %%

def example_stability_and_condition(example_mu_c :float = 450):
    
    def example_stability_plot(example_mu_c,
                               ax):
        
        example_stability = le_pivot_r(df_simulation,
                                       columns="M",
                                       index="mu_c")[0].loc[example_mu_c, :].to_frame()
        example_stability.reset_index(inplace = True)
        example_stability.rename(columns = {example_mu_c : 'P(stability)'}, inplace = True)
        
        sns.lineplot(data = example_stability,
                     x = 'M',
                     y = 'P(stability)',
                     ax = ax,
                     linewidth = 1.5,
                     color = 'black',
                     err_style = "bars",
                     errorbar = ("pi", 100),
                     marker = "o")
        
        se_95 = 1.96*np.sqrt((example_stability['P(stability)'] * \
                              (1 - example_stability['P(stability)']))/5)
        
        ax.errorbar(x = example_stability['M'], 
                    y =  example_stability['P(stability)'],
                    yerr = se_95,
                    fmt = 'none',
                    ecolor = 'black',
                    linewidth = 1.5) 
        
        ax.set_xticks(resource_pool_sizes[::4],
                                      labels = resource_pool_sizes[::4],
                                      fontsize = 10, rotation = 0)
        
        ax.yaxis.set_tick_params(labelsize = 10)
        
        ax.set_xlabel('resource pool size, ' + r'$M$',
                      fontsize = 10,
                      weight = 'bold')
        ax.set_ylabel('prob. (stability)',
                      fontsize = 10,
                      weight = 'bold',)
        
        sns.despine(ax = ax)
        
    def example_stability_condition_plot(example_mu_c,
                                         ax):
        
        example_id = df_simulation[df_simulation['mu_c'] == example_mu_c].groupby('M')[['M',
                                                                                      'rho',
                                                                                      'Species packing']].apply('mean')
        dfl = pd.melt(example_id[['M', 'rho', 'Species packing']], ['M'])
        dfl.loc[dfl['variable'] == 'rho', 'value'] = \
            dfl.loc[dfl['variable'] == 'rho', 'value']**2
            
        ax.add_patch(Rectangle((np.min(resource_pool_sizes) - 20,
                                np.min(dfl['value']) - 0.04),
                               np.max(resource_pool_sizes) - np.min(resource_pool_sizes) + 40,
                               1.0 - np.min(dfl['value']) + 0.04,
                               fill = True,
                               color = '#8f8cc0ff',
                               zorder = 0))
        
        sns.lineplot(dfl,
                     x = 'M',
                     y = 'value',
                     hue = 'variable',
                     ax = ax,
                     linewidth = 3,
                     palette = sns.color_palette(['black', 'black'], 2),
                     zorder = 10)
        
        sns.lineplot(dfl,
                     x = 'M',
                     y = 'value',
                     hue = 'variable',
                     ax = ax,
                     linewidth = 2.5,
                     marker = 'o',
                     markersize = 8,
                     palette = sns.color_palette(['#00557aff', '#3dc27aff'], 2),
                     zorder = 10,
                     markeredgewidth = 0.4,
                     markeredgecolor = 'black')

        ax.set_xlim([np.min(dfl['M']) - 20,
                     np.max(dfl['M']) + 20])
        ax.set_ylim([np.min(dfl['value']) - 0.02,
                     np.max(dfl['value']) + 0.02])
        
        ax.set_xlabel('resource pool size, ' + r'$M$',
                      fontsize = 10,
                      weight = 'bold')
        ax.set_ylabel('')
        ax.tick_params(axis='both',
                       which='major',
                       labelsize=10)
        ax.set_xticks(resource_pool_sizes[::4],
                      labels = resource_pool_sizes[::4])

        ax.legend_.remove()
        
    resource_pool_sizes = np.unique(df_simulation['M'])

    fig, (ax1, ax2) = plt.subplots(1, 2,
                                   layout="constrained",
                                   figsize=(4.5, 2))
    
    example_stability_plot(example_mu_c,
                           ax1)
    
    example_stability_condition_plot(example_mu_c,
                                     ax2)
    
    plt.savefig("C:/Users/jamil/Documents/PhD/Figures/resource_diversity_stability/self_limit_M_vs_mu_c_external_supply.png",
                bbox_inches='tight')
    plt.savefig("C:/Users/jamil/Documents/PhD/Figures/resource_diversity_stability/self_limit_M_vs_mu_c_external_supply.svg",
                bbox_inches='tight')
    
    plt.show()

example_stability_and_condition()
example_stability_and_condition(example_mu_c=650)

# %%

rho_pivot = pd.pivot_table(df_simulation,
                           columns="M",
                           index="mu_c",
                           values="rho",
                           aggfunc="mean")

sp_pivot = pd.pivot_table(df_simulation,
                          columns="M",
                          index="mu_c",
                          values="Species packing",
                          aggfunc="mean")

