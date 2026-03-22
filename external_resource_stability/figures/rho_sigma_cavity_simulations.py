# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 16:03:52 2026

@author: jamil
"""

# %%

import numpy as np
import pandas as pd
import seaborn as sns
import os
import sys
from scipy.optimize import curve_fit

from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import matplotlib.patheffects as patheffects

os.chdir('C:/Users/jamil/Documents/PhD/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models/external_resource_stability/figures')

sys.path.insert(0, "C:/Users/jamil/Documents/PhD/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models/" + \
                    "external_resource_stability")
from simulation_functions import le_pivot_r

# %%

def plot_sigma_vs_stability(simulations, sces,
                            example_rho = 0.9, smooth_method = "quadratic"):
    
    ### stability threshold ###
    
    def quadratic_fit(sigmas, stability_distance):
        
        smoother = np.poly1d(np.polyfit(sigmas, stability_distance, 2))
        
        return smoother
    
    def ndimension_fit(sigmas, stability_distance):
        
        smoother = np.poly1d(np.polyfit(sigmas, stability_distance, 10))
        
        return smoother
    
    sces_subset = sces.iloc[np.where((sces['rho'] == example_rho) & \
                                     (np.isnan(sces['loss']) == False))]
    
    #stability_distance = 2*np.sqrt(sces_subset['Packing ratio'].to_numpy()) - \
    #                        1/example_rho
    stability_distance = sces_subset['Packing ratio'].to_numpy() - \
                           sces_subset['Stability Term'].to_numpy()
                            
    smoothed_x = np.arange(np.min(sces_subset['sigma_c']),
                           np.max(sces_subset['sigma_c']),
                           0.01)
    
    match smooth_method:
        
        case "quadratic":
            
            smoother = quadratic_fit(sces_subset['sigma_c'].to_numpy(),
                                     stability_distance)
            stability_threshold = smoothed_x[np.abs(smoother(smoothed_x)).argmin()]
            
        case "n-dimension":
             
             smoother = ndimension_fit(sces_subset['sigma_c'].to_numpy(),
                                       stability_distance)
             stability_threshold = smoothed_x[np.abs(smoother(smoothed_x)).argmin()]
            
    ####
    
    example_stability = simulations.loc[example_rho, :].to_frame()
    example_stability.reset_index(inplace = True)
    example_stability.rename(columns = {example_rho : 'P(stability)'}, inplace = True)
    
    fig, axs = plt.subplots(1, 2, figsize = (5.1, 3.2), sharex = True,
                            layout = "constrained")
    
    axs[0].vlines(stability_threshold, 0, 1,
                  color = 'grey', linewidth = 2, zorder = 1, linestyle = "--")
    
    sns.lineplot(data = example_stability, x = 'sigma_c', y = 'P(stability)',
                 ax = axs[0], linewidth = 1.5, color = 'black',
                 err_style = "bars", errorbar = ("pi", 100),
                 marker = "o")
    
    axs[0].xaxis.set_tick_params(labelsize = 10)
    axs[0].yaxis.set_tick_params(labelsize = 10)
    
    axs[0].set_xlabel('')
    axs[0].set_ylabel('prob. (stability)',
                      fontsize = 10, weight = 'bold',
                      verticalalignment = 'center', horizontalalignment = 'center')
    
    sns.despine(ax = axs[0])
    
    ##########
    
    
    dfl = pd.melt(sces_subset[['sigma_c', 'Stability Term', 'Packing ratio']], ['sigma_c'])
    
    
    axs[1].vlines(stability_threshold, np.min(dfl['value']), np.max(dfl['value']),
                  color = 'grey', linewidth = 2, zorder = 1, linestyle = "--")
    
    sns.lineplot(dfl, x = 'sigma_c', y = 'value', hue = 'variable',
                 ax = axs[1], linewidth = 3,
                 palette = sns.color_palette(['black', 'black'], 2),
                 zorder = 10)
    
    sns.lineplot(dfl, x = 'sigma_c', y = 'value', hue = 'variable',
                 ax = axs[1], linewidth = 2.5, marker = 'o', markersize = 8,
                 palette = sns.color_palette(['#00557aff', '#3dc27aff'], 2),
                 zorder = 10, markeredgewidth = 0.4, markeredgecolor = 'black')
    
    #dfl_sqrt = dfl[dfl['variable'] == "Packing ratio"]
    #dfl_sqrt['value'] = np.sqrt(dfl_sqrt['value'])
    
    #sns.lineplot(dfl_sqrt, x = 'sigma_c', y = 'value',
    #             ax = axs[1], linewidth = 2.5, marker = 'o', markersize = 8,
    #             color = "cyan",
    #             zorder = 10, markeredgewidth = 0.4, markeredgecolor = 'black')
    
    #axs[1].hlines(example_rho, np.min(dfl['sigma_c']), np.max(dfl['sigma_c']),
    #              color = 'grey', linewidth = 2, zorder = 1, linestyle = "--")
    
    axs[1].set_xlabel('')
    axs[1].set_ylabel('')
    axs[1].tick_params(axis='both', which='major', labelsize=10)

    axs[1].legend_.remove()
        
    axs[1].text(-0.22, 0,
                    'stability term', fontsize = 10, weight = 'bold', color = '#00557aff',
                    path_effects = [patheffects.withStroke(linewidth=0.5, foreground='black')],
                    verticalalignment = 'bottom', horizontalalignment = 'left',
                    rotation = 90, transform=axs[1].transAxes)
    
    axs[1].text(-0.22, 0.5,
                'packing ratio', fontsize = 10, weight = 'bold',
                color = '#3dc27aff', 
                path_effects = [patheffects.withStroke(linewidth=0.5, foreground='black')],
                verticalalignment = 'bottom', horizontalalignment = 'left',
                rotation = 90, transform=axs[1].transAxes)
    
    fig.supxlabel('std. deviation in consumption, ' + r'$\sigma_c$',
                  fontsize = 10, weight = 'bold')
    
    fig.suptitle(r'$\rho = $' + str(example_rho))
    
    plt.show()

# %%

rho_sigma_df = pd.read_csv("C:/Users/jamil/Documents/PhD/Data/external_resource_stability/simulations/rho_vs_sigma_2(mathematica).csv")
rho_sigma_df.rename(columns = {"maxLe" : "Max. lyapunov exponent"}, inplace = True)
rho_sigma_df = np.round(rho_sigma_df, 7)
rho_sigma_df = rho_sigma_df.iloc[np.where(rho_sigma_df['sigma_c'] >= 0.3)]

stability_pivot = le_pivot_r(rho_sigma_df, index = "rho", columns = "sigma_c")[0]

rho_sigma_sces = pd.read_csv("C:/Users/jamil/Documents/PhD/Data/external_resource_stability/self_consistency_equations/rho_vs_sigma_updupdupdupd07(mathematica).csv",
                             na_values=["Missing[Failed]", "Missing[]"])
rho_sigma_sces = rho_sigma_sces.apply(pd.to_numeric, errors="coerce")
rho_sigma_sces = rho_sigma_sces.iloc[np.where(rho_sigma_sces['sigma_c'] >= 0.3)]

# %%

plot_sigma_vs_stability(stability_pivot, rho_sigma_sces,
                        example_rho=0.8, smooth_method="n-dimension")
plot_sigma_vs_stability(stability_pivot, rho_sigma_sces,
                        example_rho=0.9, smooth_method="n-dimension")
plot_sigma_vs_stability(stability_pivot, rho_sigma_sces,
                        example_rho=1, smooth_method="n-dimension")