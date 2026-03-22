# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 12:46:03 2026

@author: jamil
"""

import numpy as np
import pandas as pd
import seaborn as sns
import os
import sys

from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

os.chdir("C:/Users/jamil/Documents/PhD/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models/" + \
         "resource_diversity_stability(sl)/cavity_solutions_vs_simulations")
    
sys.path.insert(0, "C:/Users/jamil/Documents/PhD/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models/" + \
                    "resource_diversity_stability(sl)")
from simulation_functions import le_pivot_r, load_in_communities

# %%

directory = "C:/Users/jamil/Documents/PhD/Data/resource_diversity_stability/simulations/M_vs_mu_c_complex_ecosystem_2"
df_simulation = pd.concat([pd.read_csv(directory + "/" + file) 
                           for file in os.listdir(directory)
                           if not file.endswith('bz2')],
                          axis = 0, ignore_index = True)

chaotic_populations = load_in_communities(directory + "/simulations_75_140.bz2")
    
stable_populations = load_in_communities(directory + "/simulations_250_140.bz2")

# %%

def Stability_Plot(df_simulation,
                   chaotic_populations,
                   stable_populations,
                   example_mu_c=140):
    
    resource_pool_sizes = np.unique(df_simulation['PS_1'])
    mu_cs = np.unique(df_simulation['mu_c_2'])
    
    ######################## Phase diagram ######################################
    
    # Simulation data
    
    stability_sim_pivot = le_pivot_r(df_simulation, columns = 'PS_1',
                                     index = 'mu_c_2')[0]
    
    sns.set_style('ticks')
    
    mosaic = [[".", ".", ".", ".", "D1", ".","."],
              ["M_S_star", ".", "P", ".", "D1", ".","I_C"],
              ["M_S_star", ".", "P", ".", ".", ".","I_C"],
              ["M_S_star", ".", "P", ".", "D2",  ".", "I_C" ],
              ["M_S_star", ".", "P", ".", ".", ".","I_C"],
              ["M_S_star", ".", "P", ".", "D3",  ".", "I_C" ],
              [".", ".", "P", ".", ".",  ".", "I_C" ],
              ["M_stability", ".", "P", ".", "D4",  ".", "I_C" ],
              ["M_stability", ".", "P", ".", ".",  ".", "I_C" ],
              ["M_stability", ".", "P", ".", "D5",  ".", "I_C" ],
              ["M_stability", ".", "P", ".", ".",  ".", "I_C" ],
              ["M_stability", ".", "P", ".", "D6",  ".", "I_C" ],
              [".", ".", ".", ".", "D6",  ".", "I_C" ]]
    
    fig, axs = plt.subplot_mosaic(mosaic, figsize = (9, 3.5),
                                  width_ratios = [2.5, 1.3, 7, 0.35, 1.75, 0.9, 2],
                                  height_ratios = [0.4, 0.1, 0.05, 0.5, 0.05, 0.5,
                                                   0.13, 0.5, 0.05, 0.5, 0.05, 0.1, 0.4],
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
    
        
    ####################### Example population dynamics ######################
    
    # M = 75 and 225, mu_c = 140
    
    example_M = [75, 250]
                 
    def indices_and_cmaps(M):
        
        PS1, PS2, PS3 = np.arange(M), np.arange(M, M*2), np.arange(M*2, M*3)
        
        ps1_colidx, ps2_colidx, ps3_colidx = np.arange(M), np.arange(M), np.arange(M)
        np.random.shuffle(ps1_colidx)
        np.random.shuffle(ps2_colidx)
        np.random.shuffle(ps3_colidx)
        
        cmap_ps1 = LinearSegmentedColormap.from_list('custom YlGBl',
                                                   ['#e9a100ff','#1fb200ff',
                                                    '#1f5a00ff','#00e9e9ff','#001256fd'],
                                                   N = M)
        
        cmap_ps2 = LinearSegmentedColormap.from_list('custom YlGBl',
                                                   ['#e9a100ff','#1fb200ff',
                                                    '#1f5a00ff','#00e9e9ff','#001256fd'],
                                                   N = M)
        
        cmap_ps3 = LinearSegmentedColormap.from_list('custom YlGBl',
                                                   ['#e9a100ff','#1fb200ff',
                                                    '#1f5a00ff','#00e9e9ff','#001256fd'],
                                                   N = M)
        
        return [PS1, ps1_colidx, cmap_ps1],\
                [PS2, ps2_colidx, cmap_ps2], \
                [PS3, ps3_colidx, cmap_ps3]
    
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
        zip([axs['D1'], axs['D2'], axs['D3'],
             axs['D4'], axs['D5'], axs['D6']],
            [chaotic_populations[0], chaotic_populations[0], chaotic_populations[0],
             stable_populations[2], stable_populations[2], stable_populations[2]],
            i_c_rp,
            ['', '', '', '', '', '']):
        
        plot_dynamics(ax, simulation, i_c_rp_M, title)
        sns.despine(ax = ax)
   
    ####################### M vs S* ####################################
    
    df_simulation['S*_2'] = df_simulation['phi_TL2'] * df_simulation['PS_2']
    df_simulation['S*_3'] = df_simulation['phi_TL3'] * df_simulation['PS_3']
    
    df_meld = \
        pd.melt(df_simulation[df_simulation['mu_c_2'] == example_mu_c][['PS_1',
                                                                        'S*_2',
                                                                        'S*_3']],
                ['PS_1'])
        
    sns.lineplot(data = df_meld,
                 x = 'PS_1', y = 'value', hue = 'variable',
                 ax = axs['M_S_star'],
                 linewidth = 1.5,
                 palette = sns.color_palette(['grey', 'black'], 2),
                 err_style = "bars", errorbar = ("pi", 100))

    axs['M_S_star'].set_xticks(resource_pool_sizes[::4],
                               labels = np.repeat("", len(resource_pool_sizes[::4])),
                               fontsize = 10, rotation = 0)
    
    axs['M_S_star'].yaxis.set_tick_params(labelsize = 10)
    
    axs['M_S_star'].set_xlabel('', fontsize = 10,
                               weight = 'bold')
    axs['M_S_star'].set_ylabel('')
    
    axs['M_S_star'].text(-0.37, 0.5, 'no. coexisting\nspecies per\ntrophic level',
                         fontsize = 10, weight = 'bold',
                         verticalalignment = 'center', horizontalalignment = 'center',
                         transform=axs["M_S_star"].transAxes, rotation = 90,
                         linespacing = 0.9)
    
    axs['M_S_star'].get_legend().remove()
    
    sns.despine(ax = axs["M_S_star"])
    
    ################ M vs P(Stability) ####################
    
    example_stability = stability_sim_pivot.loc[example_mu_c, :].to_frame()
    example_stability.reset_index(inplace = True)
    example_stability.rename(columns = {example_mu_c : 'P(stability)'}, inplace = True)
    
    sns.lineplot(data = example_stability, x = 'PS_1', y = 'P(stability)',
                 ax = axs['M_stability'], linewidth = 1.5, color = 'black',
                 err_style = "bars", errorbar = ("pi", 100),
                 marker = "o")
    
    se_95 = 1.96*np.sqrt((example_stability['P(stability)'] * \
                          (1 - example_stability['P(stability)']))/20)
    
    error_bars = axs['M_stability'].errorbar(x = example_stability['PS_1'], 
                                             y =  example_stability['P(stability)'],
                                             yerr = se_95,
                                             fmt = 'none',
                                             ecolor = 'black',
                                             linewidth = 1.5) 
    
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
    
    plt.savefig("C:/Users/jamil/Documents/PhD/Figures/resource_diversity_stability/M_vs_mu_c_complex_ecosystem.png",
                bbox_inches='tight')
    plt.savefig("C:/Users/jamil/Documents/PhD/Figures/resource_diversity_stability/M_vs_mu_c_complex_ecosystem.svg",
                bbox_inches='tight')
        
    plt.show()

Stability_Plot(df_simulation,
               chaotic_populations,
               stable_populations)