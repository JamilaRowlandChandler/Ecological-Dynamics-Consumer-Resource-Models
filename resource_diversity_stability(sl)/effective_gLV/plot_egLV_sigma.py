# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 11:17:47 2025

@author: jamil
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import seaborn as sns
import os
import sys
from typing import Union

from matplotlib import pyplot as plt
import matplotlib.patheffects as patheffects
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

os.chdir('C:/Users/jamil/Documents/PhD/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models/resource_diversity_stability(sl)/effective_gLV')

sys.path.insert(0, "C:/Users/jamil/Documents/PhD/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models/" + \
                    "consumer_resource_modules")
    
sys.path.insert(0, "C:/Users/jamil/Documents/PhD/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models/" + \
                    "resource_diversity_stability(sl)")
from simulation_functions import prop_chaotic, generate_simulation_df

# %%

def sigma_vs_competition(sigma_name, egLVs):
    
    '''
    Generate dataframe of resource pool size vs inter-species interaction
    statistics from the eLVs
    '''
    
    competition_stats = dict(mu_Aii = [community.mu_Aii 
                                       for egLV_communities in egLVs
                                       for community in egLV_communities],
                             mu_Aij = [community.mu_Aij 
                                       for egLV_communities in egLVs
                                       for community in egLV_communities],
                             sigma_Aii = [community.sigma_Aii 
                                          for egLV_communities in egLVs
                                          for community in egLV_communities],
                             sigma_Aij = [community.sigma_Aij 
                                          for egLV_communities in egLVs
                                          for community in egLV_communities],
                             mu_Aij_tot = [community.interaction_statistics['mu_Aij_tot'] 
                                           for egLV_communities in egLVs
                                           for community in egLV_communities],
                             sigma_Aij_tot = [community.interaction_statistics['sigma_Aij_tot'] 
                                              for egLV_communities in egLVs
                                              for community in egLV_communities],
                             rho_A = [community.rho_A 
                                      for egLV_communities in egLVs
                                      for community in egLV_communities])
    
    match sigma_name:
        
        case 'sigma_c':
            
            competition_stats[sigma_name] = np.round([community.sigma_c * \
                                                      np.sqrt(community.no_resources)
                                                      for egLV_communities in egLVs
                                                      for community in egLV_communities],
                                                     4)
        
        case 'sigma_y':
            
            competition_stats[sigma_name] = np.round([community.sigma_y
                                                      for egLV_communities in egLVs
                                                      for community in egLV_communities],
                                                     4)
        
    competition_df = pd.DataFrame(competition_stats)
    
    return competition_df

# %%

def CRM_vs_egLV_plot(sigma_name,
                     CRM_simulation, egLV_simulation,
                     filename_save,
                     M = 225):
    
    CRM_simulation = CRM_simulation.loc[CRM_simulation['M'] == M,
                                        [sigma_name, "Max. lyapunov exponent"]]
    
    sigmas = np.unique(CRM_simulation[sigma_name])
    
    #### Stability pivots ####
    
    stability_CRM = 1 - CRM_simulation.groupby(sigma_name).apply(prop_chaotic,
                                                                 include_groups = False).to_frame()
    stability_egLV = 1 - egLV_simulation.groupby(sigma_name).apply(prop_chaotic,
                                                                   include_groups = False).to_frame()
    
    stability_CRM.rename(columns = {0 : "P(stability) (CRM)"},
                         inplace = True)
    stability_egLV.rename(columns = {0 : "P(stability) (egLV)"},
                          inplace = True)
    
    stability_df = pd.melt(pd.concat([stability_CRM, stability_egLV],
                                     axis = 1).reset_index(),
                           sigma_name)
    
    se_95 = 1.96*np.sqrt((stability_df['value'] * (1 - stability_df['value']))/20)
    
    se_colours = np.select([stability_df["variable"] == "P(stability) (CRM)",
                            stability_df["variable"] == "P(stability) (egLV)"],
                           ["black", "gray"],
                           default = "No color")
    
    #############################
    
    fig, axs = plt.subplots(1, 1, figsize = (2.5, 2.3))
    
    sns.lineplot(data = stability_df,
                 x = sigma_name, y = 'value', hue = "variable",
                 ax = axs, linewidth = 3,
                 palette = sns.color_palette(['black', 'black'], 2),
                 zorder = 10)
    
    sns.lineplot(data = stability_df,
                 x = sigma_name, y = 'value', hue = "variable",
                 ax = axs, linewidth = 2.5,
                 palette = sns.color_palette(['black', 'gray'], 2),
                 zorder = 10, marker = 'o', markersize = 7,
                 markeredgewidth = 0.4, markeredgecolor = 'black')
    
    error_bars = axs.errorbar(x = stability_df[sigma_name],
                              y = stability_df['value'],
                                 yerr = se_95, fmt = 'none', 
                                 ecolor = se_colours,
                                 linewidth = 1.8) 
    
    error_bars[2][0].set_path_effects([patheffects.Stroke(linewidth = 2.4,
                                                          foreground = 'black'),
                                       patheffects.Normal()])

    axs.set_xticks(sigmas[::2],
                   labels = sigmas[::2],
                   fontsize = 10, rotation = 0)
    
    axs.yaxis.set_tick_params(labelsize = 10)
    
    #axs.set_xlabel('resource pool size, ' + r'$M$', fontsize = 10, weight = 'bold')
    axs.set_ylabel('prob. (stability)', fontsize = 10, weight = 'bold')
    
    axs.legend_.remove()
    
    sns.despine()
    
    plt.savefig("C:/Users/jamil/Documents/PhD/Figures/resource_diversity_stability/" + \
                filename_save + ".png", 
                bbox_inches='tight', dpi = 400)
    plt.savefig("C:/Users/jamil/Documents/PhD/Figures/resource_diversity_stability/" + \
                filename_save + ".svg",
                bbox_inches='tight')
    
    plt.show()
    
# %%

def eLV_interact_stats_plot(sigma_name, 
                            competition_df, 
                            filename_save, 
                            zoom_sigma):
    
    df_mu = pd.melt(competition_df[[sigma_name, "mu_Aii", "mu_Aij", "mu_Aij_tot"]],
                    sigma_name)

    df_sigma = pd.melt(competition_df[[sigma_name, "sigma_Aii", "sigma_Aij",
                                       "sigma_Aij_tot"]],
                       sigma_name)
    
    df_mu = df_mu.groupby([sigma_name, 'variable'])['value'].apply('mean').to_frame()
    df_mu.reset_index(inplace = True)
    df_mu['value'] = np.log10(df_mu['value'])
    
    df_sigma = df_sigma.groupby([sigma_name, 'variable'])['value'].apply('mean').to_frame()
    df_sigma.reset_index(inplace = True)
    df_sigma['value'] = np.log10(df_sigma['value'])
    
    ############################
    
    fig, axs = plt.subplots(1, 2, figsize = (4.8, 2.4), sharex = True,
                            layout = 'tight')
    
    colour_palette = list(np.array(sns.husl_palette())[[0, 3, 4]])
    
    sns.lineplot(data = df_mu,
                 x = sigma_name, y = 'value', hue = "variable",
                 ax = axs[0], linewidth = 3,
                 palette = ['black', 'black', 'black'],
                 zorder = 0, err_style="bars")
    
    fig_mu = sns.lineplot(data = df_mu,
                          x = sigma_name, y = 'value', hue = "variable",
                          ax = axs[0], linewidth = 2.5,
                          palette = colour_palette, zorder = 1, err_style="bars")
    
    axins = axs[0].inset_axes([0.3, 0.4, 0.7, 0.4], zorder = 10)
    
    sns.lineplot(data = df_mu,
                 x = sigma_name, y = 'value', hue = "variable",
                 ax = axins, linewidth = 3,
                 palette = ['black', 'black', 'black'],
                 zorder = 9, err_style="bars")
    
    sns.lineplot(data = df_mu,
                 x = sigma_name, y = 'value', hue = "variable",
                 ax = axins, linewidth = 2.5,
                 palette = colour_palette, zorder = 10, err_style="bars")
    
    xmin, xmax = zoom_sigma
    axins.set_xlim([xmin, xmax])
    
    axins.set_ylim([np.min(df_mu.loc[(df_mu['variable'] != "mu_Aij_tot") & \
                                     (df_mu[sigma_name] >= xmin) & \
                                     (df_mu[sigma_name] <= xmax), 
                                     'value']) - 0.005,
                    np.max(df_mu.loc[(df_mu['variable'] != "mu_Aij_tot") & \
                                     (df_mu[sigma_name] >= xmin) & \
                                     (df_mu[sigma_name] <= xmax), 
                                     'value']) + 0.005])
    axins.legend_.remove()
    axins.xaxis.set_tick_params(labelsize = 8)
    axins.yaxis.set_tick_params(labelsize = 8)
    axins.set_xlabel('')
    axins.set_ylabel('')
    
    mark_inset(axs[0], axins, loc1=4, loc2=3, ec="0.5", 
               zorder = 5)                
    
    ##########
    
    sns.lineplot(data = df_sigma,
                 x = sigma_name, y = 'value',
                 ax = axs[1], linewidth = 3, hue = "variable",
                 palette = ['black', 'black', 'black'],
                 zorder = 10, err_style="bars")
    
    fig_sigma = sns.lineplot(data = df_sigma,
                             x = sigma_name, y = 'value', hue = "variable",
                             ax = axs[1], linewidth = 2.5,
                             palette = colour_palette, zorder = 10, err_style="bars")

    #fig.supxlabel('resource pool size, ' + r'$M$', fontsize = 10, weight = 'bold')
    
    for ax in axs.flatten():
        
        ax.set_xticks(np.unique(df_mu[sigma_name].to_numpy())[::2],
                      labels = np.unique(df_mu[sigma_name].to_numpy())[::2],
                      fontsize = 10, rotation = 0)
    
        ax.yaxis.set_tick_params(labelsize = 10)
    
        ax.set_xlabel('', fontsize = 10, weight = 'bold')
        ax.set_ylabel('', fontsize = 10, weight = 'bold')
    
        sns.despine(ax = ax)
        ax.legend_.remove()
    
    plt.savefig("C:/Users/jamil/Documents/PhD/Figures/resource_diversity_stability/" + \
                filename_save + ".png", 
                bbox_inches='tight', dpi = 400)
    plt.savefig("C:/Users/jamil/Documents/PhD/Figures/resource_diversity_stability/" + \
                filename_save + ".svg",
                bbox_inches='tight')
    
    plt.show()
    
# %%

def read_eLV_data(sigma_name, subdirectory):
       
    directory = "C:/Users/jamil/Documents/PhD/Data/resource_diversity_stability/simulations/egLV/" + \
                    subdirectory
    
    egLVs = [pd.read_pickle(os.path.join(directory, file)) for file in os.listdir(directory)]
    
    match sigma_name:
        
        case 'sigma_c':
            
            egLV_df = pd.concat([pd.DataFrame({sigma_name : np.round(np.repeat(egLV_communities[0].sigma_c * \
                                                                               np.sqrt(egLV_communities[0].no_resources),
                                                                               len(egLV_communities)), 4),
                                               'max_le' : [gLV_community.max_lyapunov_exponent 
                                                           for gLV_community in egLV_communities]})
                                 for egLV_communities in egLVs])
            
        case 'sigma_y':
            
            egLV_df = pd.concat([pd.DataFrame({sigma_name : np.round(np.repeat(getattr(egLV_communities[0],
                                                                                       sigma_name),
                                                                               len(egLV_communities)), 4),
                                               'max_le' : [gLV_community.max_lyapunov_exponent 
                                                           for gLV_community in egLV_communities]})
                                 for egLV_communities in egLVs])
    
    competition_df = sigma_vs_competition(sigma_name, egLVs) 
    
    return egLV_df, competition_df

# %%

sigma_c_CRM_df = generate_simulation_df("C:/Users/jamil/Documents/PhD/Data/resource_diversity_stability/simulations/M_vs_sigma_c") 
sigma_y_CRM_df = generate_simulation_df("C:/Users/jamil/Documents/PhD/Data/resource_diversity_stability/simulations/M_vs_sigma_y") 

sigma_c_egLV_df, sigma_c_competition_df = read_eLV_data("sigma_c", 
                                                        "M_vs_sigma_c_M_225")  

sigma_y_egLV_df, sigma_y_competition_df = read_eLV_data("sigma_y",
                                                        "M_vs_sigma_y_M_225")      

# %%

CRM_vs_egLV_plot("sigma_c", 
                 sigma_c_CRM_df,
                 sigma_c_egLV_df,
                 "self_limit_stability_egLV_sigma_c")

CRM_vs_egLV_plot("sigma_y", 
                 sigma_y_CRM_df,
                 sigma_y_egLV_df,
                 "self_limit_stability_egLV_sigma_y")

# %%

eLV_interact_stats_plot("sigma_c", 
                        sigma_c_competition_df,
                        "self_limit_interact_stats_egLV_sigma_c",
                        [1.3, 2.1])

eLV_interact_stats_plot("sigma_y", 
                        sigma_y_competition_df,
                        "self_limit_interact_stats_egLV_sigma_y",
                        [0.15, 0.25])

# %%

sigma_c_phi_R_egLV_df, sigma_c_phi_R_competition_df = read_eLV_data("sigma_c", 
                                                                    "M_vs_sigma_c_M_225(phi_R)")  

sigma_y_phi_R_egLV_df, sigma_y_phi_R_competition_df = read_eLV_data("sigma_y",
                                                                    "M_vs_sigma_y_M_225(phi_R)")      

# %%

CRM_vs_egLV_plot("sigma_c", 
                 sigma_c_CRM_df,
                 sigma_c_phi_R_egLV_df,
                 "self_limit_stability_egLV_sigma_c(phi_R)")

CRM_vs_egLV_plot("sigma_y", 
                 sigma_y_CRM_df,
                 sigma_y_phi_R_egLV_df,
                 "self_limit_stability_egLV_sigma_y(phi_R)")

# %%

eLV_interact_stats_plot("sigma_c", 
                        sigma_c_phi_R_competition_df,
                        "self_limit_interact_stats_egLV_sigma_c(phi_R)")

eLV_interact_stats_plot("sigma_y", 
                        sigma_y_phi_R_competition_df,
                        "self_limit_interact_stats_egLV_sigma_y(phi_R)")