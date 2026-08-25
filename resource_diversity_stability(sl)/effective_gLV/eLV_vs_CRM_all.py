# -*- coding: utf-8 -*-
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

os.chdir('C:/Users/jamil/Documents/PhD/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models/resource_diversity_stability(sl)/cavity_solutions_vs_simulations')

sys.path.insert(0, "C:/Users/jamil/Documents/PhD/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models/" + \
                    "resource_diversity_stability(sl)")
from simulation_functions import generate_simulation_df, le_pivot_r

# %%

def read_eLV_data(subdirectory):
       
    directory = "C:/Users/jamil/Documents/PhD/Data/resource_diversity_stability/simulations/" + \
                    subdirectory
    
    egLV_Ms = [pd.read_pickle(os.path.join(directory, file)) for file in os.listdir(directory)]
    
    egLV_df = pd.concat([pd.DataFrame({'M' : np.repeat(egLV_communities[0].no_resources,
                                                         len(egLV_communities)),
                                       'mu_c' : np.repeat(np.round(egLV_communities[0].no_resources * egLV_communities[0].mu_c, 4),
                                                            len(egLV_communities)),
                 'Max. lyapunov exponent' : [gLV_community.max_lyapunov_exponent 
                          for gLV_community in egLV_communities],
                 'mu_Aij' : [gLV_community.mu_Aij 
                          for gLV_community in egLV_communities],
                 'sigma_Aij' : [gLV_community.sigma_Aij 
                          for gLV_community in egLV_communities],
                 'rho_D' : [gLV_community.rho_D
                          for gLV_community in egLV_communities],
                 'rho_R' : [gLV_community.rho_R
                          for gLV_community in egLV_communities],
                 'rho_C' : [gLV_community.rho_C
                          for gLV_community in egLV_communities],
                 'rho_1idx' : [gLV_community.rho_1idx
                          for gLV_community in egLV_communities],
                 'corr_violate' : [gLV_community.corr_violate
                                   if hasattr(gLV_community, "corr_violate")
                                   else False
                              for gLV_community in egLV_communities],
                 'mu_Aii' : [gLV_community.mu_Aii 
                          for gLV_community in egLV_communities],
                 'sigma_Aii' : [gLV_community.sigma_Aii
                          for gLV_community in egLV_communities],
                 'mu_r' : [gLV_community.mu_r 
                           if hasattr(gLV_community, "mu_r")
                           else gLV_community.r
                           for gLV_community in egLV_communities],
                 'sigma_r' : [gLV_community.sigma_r 
                              if hasattr(gLV_community, "sigma_r")
                              else 0
                              for gLV_community in egLV_communities],
                 'Divergence' : [gLV_community.ODE_sols[0].t[-1] 
                              for gLV_community in egLV_communities]
                 })
                         for egLV_communities in egLV_Ms]) 
    
    return egLV_df

# %%

def Stability_Plot(df_eLV_ar,
                   df_eLV_phi_R,
                   df_CRM):
    
    resource_pool_sizes = np.unique(df_eLV_ar['M'])
    mu_cs = np.unique(df_eLV_ar.loc[df_eLV_ar['mu_c'] < 220, 'mu_c'])
    
    ######################## Phase diagram ######################################
    
    # Simulation data
    
    stability_pivots = [le_pivot_r(df.loc[df['mu_c'] < 220, :],
                                  columns = 'M',
                                  index = 'mu_c')[0]
                        for df in [df_CRM, df_eLV_ar, df_eLV_phi_R]]
    
    titles = ["Consumer-resource model",
              "Consumer-only model",
              "Consumer-only model\n(inc. extinct resources)"]
    
    sns.set_style('ticks')

    fig, axs = plt.subplots(1, 3,
                            sharex=True, sharey=True,
                            layout='constrained',
                            figsize=(7, 2.6))
    
    for i, (stability_pivot, title, ax) in enumerate(zip(reversed(stability_pivots),
                                                         reversed(titles),
                                                         reversed(axs))):
        
        if i == 0:
            
            cbar = True
            
        else:
            
            cbar = False
    
        subfig = sns.heatmap(stability_pivot,
                             ax = ax,
                             vmin = 0,
                             vmax = 1,
                             cbar = cbar,
                             cmap = 'viridis_r')#'Purples_r')
        
        subfig.axhline(0, 0, 1, color = 'black', linewidth = 2)
        subfig.axhline(stability_pivot.shape[0], 0, 1,
                       color = 'black', linewidth = 2)
        subfig.axvline(0, 0, 1, color = 'black', linewidth = 2)
        subfig.axvline(stability_pivot.shape[1], 0, 1,
                       color = 'black', linewidth = 2)
        
        ax.set_xticks(np.arange(0.5, len(resource_pool_sizes) + 0.5, 2),
                            labels = resource_pool_sizes[::2], fontsize = 10,
                            rotation = 0)
        
        ax.set_yticks(np.arange(0.5, len(mu_cs) + 0.5, 2), labels = np.int32(mu_cs[::2]),
                            fontsize = 10, rotation = 0)
        
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.invert_yaxis()
        
        ax.set_title(title,
                     fontsize=10,
                     weight="bold",
                     horizontalalignment = 'center',
                     verticalalignment = 'top')
        
        #cbar = ax.collections[0].colorbar
        #cbar.set_label(label = 'Probability(stability)',
        #               size = '8', horizontalalignment = 'center', 
        #               verticalalignment = 'top')
        #cbar.ax.tick_params(labelsize = 10)
             
    cbar = axs[-1].collections[0].colorbar
    cbar.set_label(label = 'Probability(stability)',
                   size = '8', horizontalalignment = 'center', 
                   verticalalignment = 'top')
    cbar.ax.tick_params(labelsize = 10)
    
    fig.supxlabel('resource pool size, ' + r'$M$', fontsize = 10,
                  weight = 'bold')
    fig.supylabel('avg. total consumption coeff., ' + r'$\mu_c$',
                  fontsize = 10, weight = 'bold')
        
    plt.show()
    
# %%

def population_dynamics():
    
    ####################### Example population dynamics ######################
    
    # M = 75 and 225, mu_c = 145
    
    example_M = np.array([75, 225])
    
    chaotic_CRM = pd.read_pickle("C:/Users/jamil/Documents/PhD/Data/" +
                                         "resource_diversity_stability/simulations/M_vs_mu_c/" + 
                                         "simulations_75_1.9333.pkl")
        
    stable_CRM = pd.read_pickle("C:/Users/jamil/Documents/PhD/Data/" +\
                                        "resource_diversity_stability/simulations/M_vs_mu_c/" +  
                                        "simulations_225_0.6444.pkl")
        
    chaotic_eLV = pd.read_pickle("C:/Users/jamil/Documents/PhD/Data/" +
                                         "resource_diversity_stability/simulations/gLV/M_vs_mu_c_drc/" + 
                                         "simulations_75_1.9333.pkl")
        
    stable_eLV = pd.read_pickle("C:/Users/jamil/Documents/PhD/Data/" +\
                                        "resource_diversity_stability/simulations/gLV/M_vs_mu_c_drc/" +  
                                        "simulations_225_0.6444.pkl")
                 
    def indices_and_cmaps(M):
        
        colour_index = np.arange(M)
        np.random.shuffle(colour_index)
        
        cmap = LinearSegmentedColormap.from_list('custom YlGBl',
                                                 ['#e9a100ff','#1fb200ff',
                                                  '#1f5a00ff','#00e9e9ff','#001256fd'],
                                                   N = M)
        
        return colour_index, cmap
    
    def plot_dynamics(ax,
                      simulation,
                      i_c_rp_M,
                      title):
        
        data = simulation.ODE_sols[0]
        
        colour_index, cmap = i_c_rp_M
        var_pos = np.arange(len(colour_index))
        
        if title == "resource":
            
            var_pos += len(colour_index)
        
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
    i_c_rp = [i_c_sublist for _ in range(3) 
              for i_c_sublist in i_c_rp]
    
    fig, axs = plt.subplot_mosaic([['CRM_chaoticC',
                                    'CRM_stableC',
                                    'eLV_chaoticC',
                                    'eLV_stableC'],
                                   ['CRM_chaoticR',
                                    'CRM_stableR',
                                    '.',
                                    '.']],
                                    layout='constrained',
                                    sharex=True,
                                    sharey=True,
                                    figsize=(5, 2.6))

    for ax, simulation, i_c_rp_M, title in \
        zip(axs.values(),
            [chaotic_CRM[2], stable_CRM[4],
             chaotic_eLV[2], stable_eLV[4],
             chaotic_CRM[2], stable_CRM[4]],
            i_c_rp,
            ['consumer', 'consumer', 'consumer', 'consumer',
             'resource', 'resource']):
        
        plot_dynamics(ax, simulation, i_c_rp_M, title)
        sns.despine(ax = ax)
   

# %%

df_eLV_ar = read_eLV_data("eLV/M_vs_mu_c")

# %%

df_eLV_phiR = read_eLV_data("gLV/M_vs_mu_c(averaged)")

# %%

df_CRM = generate_simulation_df("C:/Users/jamil/Documents/PhD/Data/" \
                                + 'resource_diversity_stability/simulations/M_vs_mu_c')

    
# %%

Stability_Plot(df_eLV_ar,
               df_eLV_phiR,
               df_CRM)

# %%

population_dynamics()