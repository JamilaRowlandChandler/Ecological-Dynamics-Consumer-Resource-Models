# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 21:29:26 2026

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

directory = "C:/Users/jamil/Documents/PhD/Data/resource_diversity_stability/simulations/M_vs_mu_c_external_opt_short"

df_simulation = pd.concat([pd.read_csv(directory + "/" + file,
                                       index_col=0)
                           for file in os.listdir(directory)],
                          ignore_index=True)

df_simulation = df_simulation[df_simulation['M'] > 25]

# %%

def stability_and_condition():
    
    def example_stability_plot(mu_c,
                               ax):
        
        example_stability = le_pivot_r(df_simulation,
                                       columns="mu_c",
                                       index="M")[0]
        example_stability.reset_index(inplace = True)
        example_stability.rename(columns = {mu_c : 'P(stability)'}, inplace = True)
        
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
        
    def example_stability_condition_plot(mu_c,
                                         ax):
        
        example_id = df_simulation[df_simulation['mu_c'] == mu_c].groupby('M')[['M',
                                                                                      'rho',
                                                                                      'Species packing']].apply('mean')
        dfl = pd.melt(example_id[['M', 'rho', 'Species packing']], ['M'])
        dfl.loc[dfl['variable'] == 'Species packing', 'value'] = \
            np.sqrt(dfl.loc[dfl['variable'] == 'Species packing', 'value'])
        #dfl.loc[dfl['variable'] == 'rho', 'value'] = \
        #    dfl.loc[dfl['variable'] == 'rho', 'value']**2
            
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
                     np.max(dfl['value']) + 0.04])
        
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
    mu_c = np.unique(df_simulation['mu_c'])[0]

    fig, (ax1, ax2) = plt.subplots(1, 2,
                                   layout="constrained",
                                   figsize=(4, 2))
    
    example_stability_plot(mu_c,
                           ax1)
    
    example_stability_condition_plot(mu_c,
                                     ax2)
    
    plt.savefig("C:/Users/jamil/Documents/PhD/Figures/resource_diversity_stability/external_supply_M_vs_mu_c_opt.png",
                bbox_inches='tight')
    plt.savefig("C:/Users/jamil/Documents/PhD/Figures/resource_diversity_stability/external_supply_M_vs_mu_c_opt.svg",
                bbox_inches='tight')
    
    plt.show()

stability_and_condition()

# %%

def pos_neg_growth_consumption(M,
                               mu_c, 
                               sigma_c,
                               mu_y,
                               sigma_y):

    consumption = (mu_c/M) + (sigma_c/np.sqrt(M))*np.random.randn(M**2)
    yield_conversion = mu_y + sigma_y*np.random.randn(M**2)
    
    growth = yield_conversion * consumption 
    
    tot_sign_mismatch = np.sum((growth < 0) & (consumption > 0)) + \
                        np.sum((growth > 0) & (consumption < 0))
                        
    return (tot_sign_mismatch/M**2)*100

print(pos_neg_growth_consumption(50,
                                 np.unique(df_simulation['mu_c'])[0], 
                                 np.unique(df_simulation['sigma_c'])[0],
                                 np.unique(df_simulation['mu_y'])[0],
                                 np.unique(df_simulation['sigma_y'])[0]))

print(pos_neg_growth_consumption(250,
                                 np.unique(df_simulation['mu_c'])[0], 
                                 np.unique(df_simulation['sigma_c'])[0],
                                 np.unique(df_simulation['mu_y'])[0],
                                 np.unique(df_simulation['sigma_y'])[0]))

