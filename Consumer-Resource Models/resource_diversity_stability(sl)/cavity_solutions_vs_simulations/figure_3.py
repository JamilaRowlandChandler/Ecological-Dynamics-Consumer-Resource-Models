# -*- coding: utf-8 -*-
"""
Created on Fri Oct  3 10:49:24 2025

@author: jamil
"""

# %%

import numpy as np
import pandas as pd
import seaborn as sns
import os
import sys

from matplotlib import pyplot as plt

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import matplotlib.patheffects as patheffects

os.chdir('C:/Users/jamil/Documents/PhD/GitHub projects/Ecological-Dynamics-and-Community-Selection/Ecological-Dynamics/Consumer-Resource Models/resource_diversity_stability(sl)/cavity_solutions_vs_simulations')

sys.path.insert(0, "C:/Users/jamil/Documents/PhD/GitHub projects/Ecological-Dynamics-and-Community-Selection/" + \
                    "Ecological-Dynamics/Consumer-Resource Models/alternative_growth_consumption_coupling")
from simulation_functions import generate_simulation_df, le_pivot_r

# %%

df_simulation = generate_simulation_df("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/" \
                                       + 'resource_diversity_stability/simulations/M_vs_mu_c')
    
#pd.read_pickle("simulations/M_vs_mu_c.pkl")
globally_solved_sces = pd.read_pickle("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/" \
                                      + "resource_diversity_stability/self_consistency_equations/M_vs_mu_c.pkl") 
    #pd.read_pickle("self_consistency_equations/M_vs_mu_c.pkl")
solved_boundary = pd.read_pickle("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/" \
                                 + "resource_diversity_stability/self_consistency_equations/stability_bound/M_vs_mu_c.pkl")

#pd.read_pickle("self_consistency_equations/M_vs_mu_c_stable_bound.pkl")

# %%

def Stability_Plot(df_simulation, globally_solved_sces, solved_boundary):
    
    df_simulation = df_simulation.loc[df_simulation['mu_c'] < 220, :]
    
    resource_pool_sizes = np.unique(df_simulation['M'])
    mu_cs = np.unique(df_simulation['mu_c'])
    
    ######################## Phase diagram ######################################
    
    # Simulation data
    
    stability_sim_pivot = le_pivot_r(df_simulation, columns = 'M',
                                     index = 'mu_c')[0]
    
    sns.set_style('ticks')
    
    mosaic = [["M_S_star", ".", ".", ".", "D1", ".","."],
              ["M_S_star", ".", "P", ".", "D1", ".","I_C"],
              ["M_S_star", ".", "P", ".", ".", ".","I_C"],
              ["M_S_star", ".", "P", ".", "D2",  ".", "I_C" ],
              [".", ".", "P", ".", ".",  ".", "I_C" ],
              ["M_stability", ".", "P", ".", "D3",  ".", "I_C" ],
              ["M_stability", ".", "P", ".", ".",  ".", "I_C" ],
              ["M_stability", ".", "P", ".", "D4",  ".", "I_C" ]]
    
    #fig, axs = plt.subplot_mosaic(mosaic, figsize = (8.5, 2.2),
    #                              width_ratios = [2.5, 0.7, 5, 0.1, 1.6, 0.5, 4.2],
    #                              height_ratios = [0.2, 0.3, 0.05, 0.5, 0.1, 0.5, 0.05, 0.5],
    #                              gridspec_kw = {'hspace' : 0, 'wspace' : 0.1})
    
    fig, axs = plt.subplot_mosaic(mosaic, figsize = (9, 2.5),
                                  width_ratios = [2.5, 1.3, 5, 0.5, 1.6, 0.9, 4.2],
                                  height_ratios = [0.2, 0.3, 0.05, 0.5, 0.1, 0.5, 0.05, 0.5],
                                  gridspec_kw = {'hspace' : 0, 'wspace' : 0.1})
    
    subfig = sns.heatmap(stability_sim_pivot, ax = axs["P"],
                         vmin = 0, vmax = 1, cbar = True, cmap = 'Purples_r')
    
    subfig.axhline(0, 0, 1, color = 'black', linewidth = 2)
    subfig.axhline(stability_sim_pivot.shape[0], 0, 1,
                   color = 'black', linewidth = 2)
    subfig.axvline(0, 0, 1, color = 'black', linewidth = 2)
    subfig.axvline(stability_sim_pivot.shape[1], 0, 1,
                   color = 'black', linewidth = 2)
    
    axs["P"].set_xticks(np.arange(0.5, len(resource_pool_sizes) + 0.5, 2),
                        labels = resource_pool_sizes[::2], fontsize = 10,
                        rotation = 0)
    
    axs["P"].set_yticks(np.arange(0.5, len(mu_cs) + 0.5, 2), labels = np.int32(mu_cs[::2]),
                        fontsize = 10, rotation = 0)
    
    axs["P"].set_xlabel('resource pool size, ' + r'$M$', fontsize = 10,
                        weight = 'bold')
    axs["P"].set_ylabel('avg. total consumption coeff., ' + r'$\mu_c$',
                        fontsize = 10, weight = 'bold')
    #axs["P"].set_ylabel('')
    axs["P"].invert_yaxis()
         
    cbar = axs["P"].collections[0].colorbar
    cbar.set_label(label = 'Probability(stability)',
                   size = '8', horizontalalignment = 'center', 
                   verticalalignment = 'top')
    cbar.ax.tick_params(labelsize = 10)
    
    # Analytically-derived stability boundary
    
    good_solves = solved_boundary.loc[(solved_boundary['loss'] <= -28) & 
                                      (np.abs(solved_boundary['Instability distance']) <= 1e-5),
                                      :]
    
    smoother = np.poly1d(np.polyfit(good_solves['M'], good_solves['mu_c'], 2))
    
    smoothed_x = np.arange(np.min(resource_pool_sizes) - 25,
                           np.max(resource_pool_sizes) + 25,
                           1)
    
    y_phase = smoother(smoothed_x) - np.min(mu_cs)
    divider = np.unique(np.abs(np.diff(mu_cs)))
    y_vals = (1/divider)*y_phase + 0.5
    
    x_vals = (0.5 - (np.min(resource_pool_sizes) - np.min(smoothed_x))/25) + \
        np.arange(0, len(smoothed_x), 1)/25
    
    sns.lineplot(x = x_vals, y = y_vals, ax = axs["P"], color = 'black',
                 linewidth = 3)
    
    #################### Instability condition vs M #####################
    
    # Example relathips with mu_c = 145
    example_mu_c = 145

    df_plot = globally_solved_sces.loc[globally_solved_sces['mu_c'] == example_mu_c, :]
    dfl = pd.melt(df_plot[['M', 'rho', 'Species packing']], ['M'])
    dfl.loc[dfl['variable'] == 'Species packing', 'value'] = \
        np.sqrt(dfl.loc[dfl['variable'] == 'Species packing', 'value'])
    
    M_stability_threshold = smoothed_x[np.abs(smoother(smoothed_x) - example_mu_c).argmin()]
    
    axs['I_C'].add_patch(Rectangle((np.min(resource_pool_sizes) - 20, np.min(dfl['value']) - 0.1),
                                   M_stability_threshold - np.min(resource_pool_sizes) + 20,
                                   np.max(dfl['value']) + 0.5 - np.min(dfl['value']),
                                   fill = True, color = '#8f8cc0ff', zorder = 0))
    
    axs["I_C"].vlines(M_stability_threshold, np.min(dfl['value']) - 0.1,
                      np.max(dfl['value']) + 0.1,
                      color = 'black', linewidth = 2.5, zorder = 1)
    
    subfig1 = sns.lineplot(dfl, x = 'M', y = 'value', hue = 'variable',
                           ax = axs["I_C"], linewidth = 3,
                           palette = sns.color_palette(['black', 'black'], 2),
                           zorder = 10)
    
    subfig1 = sns.lineplot(dfl, x = 'M', y = 'value', hue = 'variable',
                           ax = axs["I_C"], linewidth = 2.5, marker = 'o', markersize = 8,
                           palette = sns.color_palette(['#00557aff', '#3dc27aff'], 2),
                           zorder = 10, markeredgewidth = 0.4, markeredgecolor = 'black')
    
    axs["I_C"].set_xlim([np.min(resource_pool_sizes) - 10, np.max(resource_pool_sizes) + 10])
    axs["I_C"].set_ylim([np.min(dfl['value']) - 0.015, np.max(dfl['value']) + 0.035])

    axs["I_C"].set_xlabel('resource pool size, ' + r'$M$', fontsize = 10,
                          weight = 'bold')
    axs["I_C"].set_ylabel('')
    axs["I_C"].tick_params(axis='both', which='major', labelsize=10)
    axs["I_C"].set_xticks(resource_pool_sizes[::2], labels = resource_pool_sizes[::2])

    axs["I_C"].legend_.remove()
        
    axs['I_C'].text((0.5*(np.max(resource_pool_sizes) + M_stability_threshold) - np.min(resource_pool_sizes))/(np.max(resource_pool_sizes) - np.min(resource_pool_sizes)),
                    0.98,
                    'Stable', fontsize = 10, weight = 'bold', color = 'black',
                    verticalalignment = 'top', horizontalalignment = 'center',
                    transform=axs["I_C"].transAxes)
    
    axs['I_C'].text((0.5*(M_stability_threshold - np.min(resource_pool_sizes)))/(np.max(resource_pool_sizes) - np.min(resource_pool_sizes)),
                    0.98,
                    'Unstable', fontsize = 10, weight = 'bold', color = 'white',
                    path_effects = [patheffects.withStroke(linewidth=1, foreground='black')],
                    verticalalignment = 'top', horizontalalignment = 'center',
                    transform=axs["I_C"].transAxes)
   
    axs['I_C'].text(-0.3, 0,
                    'reciprocity', fontsize = 10, weight = 'bold', color = '#00557aff',
                    path_effects = [patheffects.withStroke(linewidth=0.5, foreground='black')],
                    verticalalignment = 'bottom', horizontalalignment = 'left',
                    rotation = 90, transform=axs['I_C'].transAxes)
    
    axs['I_C'].text(-0.3, 0.5,
                    r'$\sqrt{\text{packing ratio}}$', fontsize = 10, weight = 'bold',
                    color = '#3dc27aff', 
                    path_effects = [patheffects.withStroke(linewidth=0.5, foreground='black')],
                    verticalalignment = 'bottom', horizontalalignment = 'left',
                    rotation = 90, transform=axs['I_C'].transAxes)
        
    ####################### Example population dynamics ######################
    
    # M = 75 and 225, mu_c = 145
    
    example_M = [75, 225]
    
    chaotic_populations = pd.read_pickle("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/" +
                                         "resource_diversity_stability/simulations/M_vs_mu_c/" + 
                                         "simulations_75_1.9333.pkl")
        
    stable_populations = pd.read_pickle("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/" +\
                                        "resource_diversity_stability/simulations/M_vs_mu_c/" +  
                                        "simulations_225_0.6444.pkl")
                 
    def indices_and_cmaps(M):
        
        species, resources = np.arange(M), np.arange(M, M*2)
        
        s_colour_index, r_colour_index = np.arange(M), np.arange(M)
        np.random.shuffle(s_colour_index)
        np.random.shuffle(r_colour_index)
        
        cmap_s = LinearSegmentedColormap.from_list('custom YlGBl',
                                                   ['#e9a100ff','#1fb200ff',
                                                    '#1f5a00ff','#00e9e9ff','#001256fd'],
                                                   N = M)
        
        cmap_r = LinearSegmentedColormap.from_list('custom YlGBl',
                                                   ['#e9a100ff','#1fb200ff',
                                                    '#1f5a00ff','#00e9e9ff','#001256fd'],
                                                   N = M)
        
        return [species, s_colour_index, cmap_s], [resources, r_colour_index,
                                                   cmap_r]
    
    def plot_dynamics(ax, simulation, i_c_rp_M, title):
        
        #breakpoint()
        
        var_pos, colour_index, cmap = i_c_rp_M
        data = simulation.ODE_sols[0]
        
        for i, v in zip(colour_index, var_pos):
        
            ax.plot(data.t, data.y[v,:].T, color = 'black', linewidth = 0.5)
            ax.plot(data.t, data.y[v,:].T, color = cmap(i), linewidth = 0.45)
        
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            
            ax.set_title(title, fontsize = 9, y = 0.85)
        
        return ax
    
    i_c_rp = [indices_and_cmaps(M) for M in example_M]
    i_c_rp = [i_c for i_c_rp_M in i_c_rp for i_c in i_c_rp_M]

    for ax, simulation, i_c_rp_M, title in \
        zip([axs['D1'], axs['D2'], axs['D3'], axs['D4']],
            [chaotic_populations[0], chaotic_populations[0],
             stable_populations[2], stable_populations[2]],
            i_c_rp,
            ['', '', '', '']):
        
        plot_dynamics(ax, simulation, i_c_rp_M, title)
        sns.despine(ax = ax)
   
    ####################### M vs S* ####################################
    
    df_simulation['S*'] = df_simulation['phi_N'] * df_simulation['M']

    sns.lineplot(data = df_simulation[df_simulation['mu_c'] == example_mu_c],
                 x = 'M', y = 'S*', ax = axs['M_S_star'], linewidth = 1.5, color = 'black',
                 err_style = "bars", errorbar = ("pi", 100))

    axs['M_S_star'].set_xticks(resource_pool_sizes[::4],
                               #labels = resource_pool_sizes[::4],
                               labels = np.repeat("", len(resource_pool_sizes[::4])),
                               fontsize = 10, rotation = 0)
    
    axs['M_S_star'].yaxis.set_tick_params(labelsize = 10)
    
    axs['M_S_star'].set_xlabel('', fontsize = 10,
                               weight = 'bold')
    axs['M_S_star'].set_ylabel('')
    
    axs['M_S_star'].text(-0.37, 0.5, 'no. coexisting\nspecies, ' + r'$S^*$',
                         fontsize = 10, weight = 'bold',
                         verticalalignment = 'center', horizontalalignment = 'center',
                         transform=axs["M_S_star"].transAxes, rotation = 90,
                         linespacing = 0.9)
    
    sns.despine(ax = axs["M_S_star"])
    
    ################ M vs P(Stability) ####################
    
    example_stability = stability_sim_pivot.loc[example_mu_c, :].to_frame()
    example_stability.reset_index(inplace = True)
    example_stability.rename(columns = {example_mu_c : 'P(stability)'}, inplace = True)
    
    sns.lineplot(data = example_stability, x = 'M', y = 'P(stability)',
                 ax = axs['M_stability'], linewidth = 1.5, color = 'black',
                 err_style = "bars", errorbar = ("pi", 100),
                 marker = "o")

    axs['M_stability'].set_xticks(resource_pool_sizes[::4],
                                  labels = resource_pool_sizes[::4],
                                  fontsize = 10, rotation = 0)
    
    axs['M_stability'].yaxis.set_tick_params(labelsize = 10)
    
    axs['M_stability'].set_xlabel('resource pool size, ' + r'$M$', fontsize = 10,
                               weight = 'bold')
    axs['M_stability'].set_ylabel('')
    
    axs['M_stability'].text(-0.37, 0.5, 'prob. (stability)',
                         fontsize = 10, weight = 'bold',
                         verticalalignment = 'center', horizontalalignment = 'center',
                         transform=axs["M_stability"].transAxes, rotation = 90,
                         linespacing = 1.4)
    
    sns.despine(ax = axs["M_stability"])
    
    #plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/self_limit_M_vs_mu_c_digram_condition.png",
    #            bbox_inches='tight')
    #plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/self_limit_M_vs_mu_c_digram_condition.svg",
    #            bbox_inches='tight')
        
    plt.show()

Stability_Plot(df_simulation, globally_solved_sces, solved_boundary)
