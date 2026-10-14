# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 10:38:51 2026

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

directory = "C:/Users/jamil/Documents/PhD/Data/resource_diversity_stability/simulations/M_vs_mu_c_generic_resource"

df_simulation = pd.concat([pd.read_csv(directory + "/" + file,
                                       index_col=0)
                           for file in os.listdir(directory)],
                          ignore_index=True)

# %%

def full_stability_diagram():

    resource_pool_sizes = np.unique(df_simulation['M'])
    exps = np.unique(df_simulation['exp'])
        
    fig, axs = generic_heatmaps(df_simulation,
                                'M',
                                'exp', 
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
    
    axs.set_ylabel('resource growth exponent, ' + r'$e$',
                   weight="bold",
                   fontsize=10)
    axs.set_yticks(np.arange(0.5, len(exps) + 0.5, 1),
                   labels = exps,
                   fontsize = 8)
    
    cbar = axs.collections[0].colorbar
    cbar.set_label(label = 'Prob. stability',
                   size = '9',
                   weight = 'bold')
    cbar.ax.tick_params(labelsize = 8)
    
    fig.suptitle("Large resource pools stabilise communities with\n" + \
                 "semi-substitutable resources",
                 fontsize = 11,
                 weight = "bold")
        
    plt.savefig("C:/Users/jamil/Documents/PhD/Figures/resource_diversity_stability/self_limit_M_vs_mu_c_generic_res_full.png",
                bbox_inches='tight')
    plt.savefig("C:/Users/jamil/Documents/PhD/Figures/resource_diversity_stability/self_limit_M_vs_mu_c_generic_res_full.svg",
                bbox_inches='tight')
    
    plt.show()
    
full_stability_diagram()

# %%

def example_stability_and_condition(example_exp :float = 0.7):
    
    def example_stability_plot(example_exp,
                               ax):
        
        example_stability = le_pivot_r(df_simulation,
                                       columns="M",
                                       index="exp")[0].loc[example_exp, :].to_frame()
        example_stability.reset_index(inplace = True)
        example_stability.rename(columns = {example_exp : 'P(stability)'}, inplace = True)
        
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
        
        ax.set_xticks(resource_pool_sizes[::2],
                                      labels = resource_pool_sizes[::2],
                                      fontsize = 10, rotation = 0)
        
        ax.yaxis.set_tick_params(labelsize = 10)
        
        ax.set_xlabel('resource pool size, ' + r'$M$',
                      fontsize = 10,
                      weight = 'bold')
        ax.set_ylabel('prob. (stability)',
                      fontsize = 10,
                      weight = 'bold',)
        
        sns.despine(ax = ax)
        
    def example_stability_condition_plot(example_exp,
                                         ax):
        
        example_id = df_simulation[df_simulation['exp'] == example_exp].groupby('M')[['M',
                                                                                      'rho',
                                                                                      'Species packing']].apply('mean')
        dfl = pd.melt(example_id[['M', 'rho', 'Species packing']], ['M'])
        dfl.loc[dfl['variable'] == 'Species packing', 'value'] = \
            np.sqrt(dfl.loc[dfl['variable'] == 'Species packing', 'value'])
            
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
        ax.set_ylim([np.min(dfl['value']) - 0.04,
                     1.0])
        
        ax.set_xlabel('resource pool size, ' + r'$M$',
                      fontsize = 10,
                      weight = 'bold')
        ax.set_ylabel('')
        ax.tick_params(axis='both',
                       which='major',
                       labelsize=10)
        ax.set_xticks(resource_pool_sizes[::2],
                      labels = resource_pool_sizes[::2])

        ax.legend_.remove()
        
    resource_pool_sizes = np.unique(df_simulation['M'])
    exps = np.unique(df_simulation['exp'])

    fig, (ax1, ax2) = plt.subplots(1, 2,
                                   layout="constrained",
                                   figsize=(4, 2))
    
    example_stability_plot(example_exp,
                           ax1)
    
    example_stability_condition_plot(example_exp,
                                     ax2)
    
    plt.savefig("C:/Users/jamil/Documents/PhD/Figures/resource_diversity_stability/self_limit_M_vs_mu_c_generic_res.png",
                bbox_inches='tight')
    plt.savefig("C:/Users/jamil/Documents/PhD/Figures/resource_diversity_stability/self_limit_M_vs_mu_c_generic_res.svg",
                bbox_inches='tight')
    
    plt.show()

example_stability_and_condition()

# %%

rho_pivot = pd.pivot_table(df_simulation,
                           columns="M",
                           index="exp",
                           values="rho",
                           aggfunc="mean")

sp_pivot = pd.pivot_table(df_simulation,
                          columns="M",
                          index="exp",
                          values="Species packing",
                          aggfunc="mean")

id_pivot = pd.pivot_table(df_simulation,
                          columns="M",
                          index="exp",
                          values="Instability distance",
                          aggfunc="mean")

