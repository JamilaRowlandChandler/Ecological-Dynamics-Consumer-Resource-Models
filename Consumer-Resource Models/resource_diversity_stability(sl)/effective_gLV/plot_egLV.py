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

os.chdir('C:/Users/jamil/Documents/PhD/GitHub projects/Ecological-Dynamics-and-Community-Selection/Ecological-Dynamics/Consumer-Resource Models/resource_diversity_stability(sl)/effective_gLV')

sys.path.insert(0, "C:/Users/jamil/Documents/PhD/GitHub projects/Ecological-Dynamics-and-Community-Selection/" + \
                    "Ecological-Dynamics/Consumer-Resource Models/consumer_resource_modules")
    
sys.path.insert(0, "C:/Users/jamil/Documents/PhD/GitHub projects/Ecological-Dynamics-and-Community-Selection/" + \
                    "Ecological-Dynamics/Consumer-Resource Models/resource_diversity_stability(sl)")
from simulation_functions import prop_chaotic, generate_simulation_df

# %%

def M_vs_competition(egLV_Ms):
    
    '''
    Generate dataframe of resource pool size vs inter-species interaction
    statistics from the eLVs
    '''
        
    competition_df = pd.DataFrame(dict(M = [community.no_resources
                                            for egLV_communities in egLV_Ms
                                            for community in egLV_communities],
                                       mu_Aii = [community.mu_Aii 
                                                 for egLV_communities in egLV_Ms
                                                 for community in egLV_communities],
                                       mu_Aij = [community.mu_Aij 
                                                 for egLV_communities in egLV_Ms
                                                 for community in egLV_communities],
                                       sigma_Aii = [community.sigma_Aii 
                                                    for egLV_communities in egLV_Ms
                                                    for community in egLV_communities],
                                       sigma_Aij = [community.sigma_Aij 
                                                    for egLV_communities in egLV_Ms
                                                    for community in egLV_communities],
                                       mu_Aij_tot = [community.interaction_statistics['mu_Aij_tot'] 
                                                    for egLV_communities in egLV_Ms
                                                    for community in egLV_communities],
                                       sigma_Aij_tot = [community.interaction_statistics['sigma_Aij_tot'] 
                                                        for egLV_communities in egLV_Ms
                                                        for community in egLV_communities]))
    
    return competition_df

# %%

def CRM_vs_egLV_plot(CRM_simulation, egLV_simulation,
                     filename_save,
                     mu_c = 145):
    
    CRM_simulation = CRM_simulation.loc[CRM_simulation['mu_c'] == mu_c,
                                        ["M", "Max. lyapunov exponent"]]
    
    resource_pool_sizes = np.unique(CRM_simulation['M'])
    
    #### Stability pivots ####
    
    stability_CRM = 1 - CRM_simulation.groupby("M").apply(prop_chaotic,
                                                          include_groups = False).to_frame()
    stability_egLV = 1 - egLV_simulation.groupby("M").apply(prop_chaotic,
                                                          include_groups = False).to_frame()
    
    stability_CRM.rename(columns = {0 : "P(stability) (CRM)"},
                           inplace = True)
    stability_egLV.rename(columns = {0 : "P(stability) (egLV)"},
                           inplace = True)
    
    stability_df = pd.melt(pd.concat([stability_CRM, stability_egLV],
                                     axis = 1).reset_index(),
                           "M")
    
    se_95 = 1.96*np.sqrt((stability_df['value'] * (1 - stability_df['value']))/20)
    
    se_colours = np.select([stability_df["variable"] == "P(stability) (CRM)",
                            stability_df["variable"] == "P(stability) (egLV)"],
                           ["black", "gray"])
    
    #############################
    
    fig, axs = plt.subplots(1, 1, figsize = (2.5, 2.3))
    
    sns.lineplot(data = stability_df,
                 x = 'M', y = 'value', hue = "variable",
                 ax = axs, linewidth = 3,
                 palette = sns.color_palette(['black', 'black'], 2),
                 zorder = 10)
    
    sns.lineplot(data = stability_df,
                 x = 'M', y = 'value', hue = "variable",
                 ax = axs, linewidth = 2.5,
                 palette = sns.color_palette(['black', 'gray'], 2),
                 zorder = 10, marker = 'o', markersize = 7,
                 markeredgewidth = 0.4, markeredgecolor = 'black')
    
    error_bars = axs.errorbar(x = stability_df['M'], y = stability_df['value'],
                                 yerr = se_95, fmt = 'none', ecolor = se_colours,
                                 linewidth = 1.8) 
    
    error_bars[2][0].set_path_effects([patheffects.Stroke(linewidth = 2.4,
                                                          foreground = 'black'),
                                       patheffects.Normal()])

    axs.set_xticks(resource_pool_sizes[::2],
                  labels = resource_pool_sizes[::2],
                  fontsize = 10, rotation = 0)
    
    axs.yaxis.set_tick_params(labelsize = 10)
    
    axs.set_xlabel('resource pool size, ' + r'$M$', fontsize = 10, weight = 'bold')
    axs.set_ylabel('prob. (stability)', fontsize = 10, weight = 'bold')
    
    axs.legend_.remove()
    
    sns.despine()
    
    plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/" + \
                filename_save + ".png", #self_limit_stability_egLV_all_resources.png
                bbox_inches='tight', dpi = 400)
    plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/" + \
                filename_save + ".svg",
                bbox_inches='tight')
    
    plt.show()
    
# %%

def eLV_interact_stats_plot(competition_df, filename_save):
    
    #breakpoint()
    
    df_mu = pd.melt(competition_df[["M", "mu_Aii", "mu_Aij", "mu_Aij_tot"]],
                    "M")

    df_sigma = pd.melt(competition_df[["M", "sigma_Aii", "sigma_Aij", "sigma_Aij_tot"]],
                       "M")
    
    df_mu = df_mu.groupby(['M', 'variable'])['value'].apply('mean').to_frame()
    df_mu.reset_index(inplace = True)
    df_mu['value'] = np.log10(df_mu['value'])
    
    df_sigma = df_sigma.groupby(['M', 'variable'])['value'].apply('mean').to_frame()
    df_sigma.reset_index(inplace = True)
    df_sigma['value'] = np.log10(df_sigma['value'])
    
    ############################
    
    fig, axs = plt.subplots(1, 2, figsize = (4.8, 2.6), sharex = True,
                            layout = 'tight')
    
    colour_palette = list(np.array(sns.husl_palette())[[0, 3, 4]])
    
    sns.lineplot(data = df_mu,
                 x = 'M', y = 'value', hue = "variable",
                 ax = axs[0], linewidth = 3,
                 palette = ['black', 'black', 'black'],
                 zorder = 10, err_style="bars")
    
    fig_mu = sns.lineplot(data = df_mu,
                          x = 'M', y = 'value', hue = "variable",
                          ax = axs[0], linewidth = 2.5,
                          palette = colour_palette, zorder = 10, err_style="bars")
    
    sns.lineplot(data = df_sigma,
                 x = 'M', y = 'value',
                 ax = axs[1], linewidth = 3, hue = "variable",
                 palette = ['black', 'black', 'black'],
                 zorder = 10, err_style="bars")
    
    fig_sigma = sns.lineplot(data = df_sigma,
                             x = 'M', y = 'value', hue = "variable",
                             ax = axs[1], linewidth = 2.5,
                             palette = colour_palette, zorder = 10, err_style="bars")
    
    fig.supxlabel('resource pool size, ' + r'$M$', fontsize = 10, weight = 'bold')
    
    for ax in axs:
        
        ax.set_xticks(np.unique(df_mu['M'].to_numpy())[::2],
                      labels = np.unique(df_mu['M'].to_numpy())[::2],
                      fontsize = 10, rotation = 0)
    
        ax.yaxis.set_tick_params(labelsize = 10)
    
        ax.set_xlabel('', fontsize = 10, weight = 'bold')
        ax.set_ylabel('', fontsize = 10, weight = 'bold')
    
        sns.despine(ax = ax)
        ax.legend_.remove()
    
    plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/" + \
                filename_save + ".png", #self_limit_stability_egLV_all_resources.png
                bbox_inches='tight', dpi = 400)
    plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/" + \
                filename_save + ".svg",
                bbox_inches='tight')
    
    plt.show()
    
# %%

def read_eLV_data(subdirectory):
       
    directory = "C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/resource_diversity_stability/simulations/egLV/" + \
                    subdirectory
    
    egLV_Ms = [pd.read_pickle(os.path.join(directory, file)) for file in os.listdir(directory)]
    
    egLV_df = pd.concat([pd.DataFrame(dict(M = np.repeat(egLV_communities[0].no_resources,
                                                         len(egLV_communities)),
                max_le = [gLV_community.max_lyapunov_exponent 
                          for gLV_community in egLV_communities]))
                         for egLV_communities in egLV_Ms])
    
    competition_df = M_vs_competition(egLV_Ms) 
    
    return egLV_df, competition_df

# %%

CRM_df = generate_simulation_df("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/resource_diversity_stability/simulations/M_vs_mu_c") 

egLV_145_df, competition_145_df = read_eLV_data("M_vs_mu_c_145")    
egLV_190_df, competition_190_df = read_eLV_data("M_vs_mu_c_190")    

egLV_145_all_df, competition_145_all_df = read_eLV_data("M_vs_mu_c_145(all_resource)")    
egLV_190_all_df, competition_190_all_df = read_eLV_data("M_vs_mu_c_190(all_resource)")    

# %%

CRM_vs_egLV_plot(CRM_df, egLV_145_df,
                 "self_limit_stability_egLV_145",
                 mu_c=145)
CRM_vs_egLV_plot(CRM_df, egLV_190_df,
                 "self_limit_stability_egLV_190",
                 mu_c=190)
CRM_vs_egLV_plot(CRM_df, egLV_145_all_df,
                 "self_limit_stability_egLV_all_resources_145",
                 mu_c=145)
CRM_vs_egLV_plot(CRM_df, egLV_190_all_df,
                 "self_limit_stability_egLV_all_resources_190",
                 mu_c=190)

# %%

eLV_interact_stats_plot(competition_145_df,
                        "self_limit_interact_stats_egLV_145.png")
eLV_interact_stats_plot(competition_190_df,
                        "self_limit_interact_stats_egLV_190.png")
eLV_interact_stats_plot(competition_145_all_df,
                        "self_limit_interact_stats_egLV_all_resources_145.png")
eLV_interact_stats_plot(competition_190_all_df,
                        "self_limit_interact_stats_egLV_all_resources_190.png")
