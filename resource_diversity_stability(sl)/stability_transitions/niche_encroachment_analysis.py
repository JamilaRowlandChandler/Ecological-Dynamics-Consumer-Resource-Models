# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 15:51:06 2026

@author: jamil
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Dec 15 10:58:57 2025

@author: jamil
"""

import numpy as np
import pandas as pd
import os
import sys
from tqdm import tqdm
from typing import Literal, Union
import numpy.typing as npt
from scipy.spatial.distance import cdist
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

os.chdir('C:/Users/jamil/Documents/PhD/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models/resource_diversity_stability(sl)/effective_gLV')
    
sys.path.insert(0, "C:/Users/jamil/Documents/PhD/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models" + \
                    "/resource_diversity_stability(sl)")
from simulation_functions import pickle_dump

# %%

def encroachment(CRM_community : Literal["SL_CRM"]):
    
    def encroachment1():
        
        GG_D = cdist(Gn, Gn, metric="euclidean")

        mask = ~np.eye(GG_D.shape[0], dtype=bool)

        GG_D_min = np.min(GG_D[mask].reshape(GG_D.shape[0],
                                             GG_D.shape[0] - 1),
                          axis=1)
        
        return np.mean(np.diag(CG_D) / (0.5*GG_D_min))
    
    def encroachment2():
        
        mask = ~np.eye(CG_D.shape[0], dtype=bool)

        CG_D_min = np.min(CG_D[mask].reshape(CG_D.shape[0],
                                             CG_D.shape[0] - 1),
                          axis=1)
        
        return np.mean(np.diag(CG_D) / CG_D_min)
    
    def encroachment3():
        
        M = 1 - cdist(G,
                      C,
                      metric="cosine")   # M[i, j] = cos(G_i, C_j)

        E_ii = np.diag(M).copy()
        off = M.copy()
        np.fill_diagonal(off,
                         0)
        
        E_ij = np.sqrt((off**2).sum(axis=1))
        E = np.mean(E_ij / E_ii) # E_ij.mean() / own.mean()  
        
        E_ii_nonorm = np.diag(1 - cdist(CRM_community.growth - mean_g,
                                        CRM_community.consumption.T - mean_c,
                                        metric="cosine"))
        
        return np.mean(E_ij), np.mean(E_ii), E, np.mean(E_ii_nonorm)
    

    M = CRM_community.no_resources
    mu_c = CRM_community.mu_c * M
    mu_y = CRM_community.mu_g
    mean_c = mu_c / M
    mean_g = mu_y * mean_c
    
    surviving_resources = np.where(CRM_community.ODE_sols[0].y[M:, -1] > 1e-4)[0]    
    surviving_consumers = np.where(CRM_community.ODE_sols[0].y[:M, -1] > 1e-4)[0]    
    
    G = (CRM_community.growth - mean_g)[np.ix_(surviving_consumers,
                                                 surviving_resources)]
    C = (CRM_community.consumption - mean_c).T[np.ix_(surviving_consumers,
                                                      surviving_resources)]

    # Paper's normalisation: project onto the simplex
    Gn = G / G.sum(axis=1, keepdims=True)
    Cn = C / C.sum(axis=1, keepdims=True)
    
    CG_D = cdist(Cn, Gn, metric="euclidean")
    
    encroachment1_val = encroachment1(),
    encroachment2_val = encroachment2(),
    E_ij_tot, E_ii, encroachment3_val, E_ii_nonorm = encroachment3()

    return dict(M = CRM_community.no_resources,
                mu_c = np.round(CRM_community.mu_c * CRM_community.no_resources, 4),
                encroachment1 = encroachment1_val,
                encroachment2 = encroachment2_val,
                encroachment3 = encroachment3_val,
                E_ij_tot = E_ij_tot,
                E_ii = E_ii,
                E_ii_nonorn = E_ii_nonorm)
    
    
# %%

def encroachment_M(CRM_directory : str):
    

    def read_call_CRM(full_CRM_directory : str,
                      filename : str):
        
        # read in consumer-resource model (CRM) communities
        CRM_communities = pd.read_pickle(full_CRM_directory + "/" + filename)
        
        encroachments = [encroachment(CRM_community) 
                         for CRM_community in 
                         tqdm(CRM_communities, leave = False, position = 0,
                              total = len(CRM_communities))]
        
        return pd.DataFrame(encroachments)
    
    ###################################################################################
    
    full_CRM_directory = "C:/Users/jamil/Documents/PhD/Data/resource_diversity_stability/simulations/" + \
                           CRM_directory
                           
    # generate filenames based on mu_c
    filenames = os.listdir(full_CRM_directory)
    
    encroachment_df = pd.concat([read_call_CRM(full_CRM_directory,
                                               filename)
                                for filename in tqdm(filenames,
                                                 leave = True,
                                                 position = 1,
                                                 total = len(filenames))],
                                  ignore_index=True)
    
    return encroachment_df
    
# %%

df = encroachment_M(CRM_directory = "M_vs_mu_c")

# %%

globally_solved_sces = pd.read_pickle("C:/Users/jamil/Documents/PhD/Data/" \
                                      + "resource_diversity_stability/self_consistency_equations/M_vs_mu_c.pkl") 
    #pd.read_pickle("self_consistency_equations/M_vs_mu_c.pkl")
solved_boundary = pd.read_pickle("C:/Users/jamil/Documents/PhD/Data/" \
                                 + "resource_diversity_stability/self_consistency_equations/stability_bound/M_vs_mu_c.pkl")

# %%

def Stability_Plot(encroach_df,
                   globally_solved_sces,
                   solved_boundary):
    
    def stability_threshold(ax,
                            data,
                            variables : list):
        
        example_mu_c = 145

        df_plot = data.loc[data['mu_c'] == example_mu_c, :]
        
        dfl = pd.melt(df_plot[['M'] + variables],
                      ['M'])
        
        M_stability_threshold = smoothed_x[np.abs(smoother(smoothed_x) - example_mu_c).argmin()]
        
        ax.add_patch(Rectangle((np.min(resource_pool_sizes) - 20, np.min(dfl['value']) - 0.1),
                               M_stability_threshold - np.min(resource_pool_sizes) + 20,
                               np.max(dfl['value']) + 0.5 - np.min(dfl['value']),
                               fill = True, color = '#8f8cc0ff', zorder = 0))
        
        ax.vlines(M_stability_threshold, np.min(dfl['value']) - 0.1,
                   np.max(dfl['value']) + 0.1,
                   color = 'black', linewidth = 2.5, zorder = 1)
        
        sns.lineplot(dfl,
                     x = 'M', y = 'value', hue = 'variable',
                     ax = ax,
                     linewidth = 3,
                     palette = sns.color_palette(['black', 'black'], 2),
                     zorder = 10,
                     err_style = "bars")
        
        sns.lineplot(dfl,
                     x = 'M', y = 'value', hue = 'variable',
                     ax = ax,
                     linewidth = 2.5,
                     marker = 'o',
                     markersize = 8,
                     palette = sns.color_palette(['#00557aff', '#3dc27aff'], 2),
                     zorder = 10,
                     markeredgewidth = 0.4,
                     markeredgecolor = 'black',
                     err_style = "bars")
        
        ax.set_xlim([np.min(resource_pool_sizes) - 10, np.max(resource_pool_sizes) + 10])
        ax.set_ylim([0.49, 0.81])

        ax.set_xlabel('resource pool size, ' + r'$M$', fontsize = 10,
                              weight = 'bold')
        ax.set_ylabel('')
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.set_xticks(resource_pool_sizes[::2], labels = resource_pool_sizes[::2])

        ax.legend_.remove()
    
    df_simulation = encroach_df.loc[encroach_df['mu_c'] < 220, :]
    
    resource_pool_sizes = np.unique(df_simulation['M'])

    sns.set_style('ticks')
    
    
    fig, (ax1, ax2) = plt.subplots(1, 2,
                                   layout="constrained",
                                   figsize=(4.5, 2.5))
    
    
    # Analytically-derived stability boundary
    
    good_solves = solved_boundary.loc[(solved_boundary['loss'] <= -28) & 
                                      (np.abs(solved_boundary['Instability distance']) <= 1e-5),
                                      :]
    
    smoother = np.poly1d(np.polyfit(good_solves['M'], good_solves['mu_c'], 2))
    
    smoothed_x = np.arange(np.min(resource_pool_sizes) - 25,
                           np.max(resource_pool_sizes) + 25,
                           1)
    
    #################### Instability condition vs M #####################
    
    globally_solved_sces['Packing ratio sqrt'] = \
        np.sqrt(globally_solved_sces['Species packing'])
        
    stability_threshold(ax1,
                        globally_solved_sces,
                        ['rho', 'Packing ratio sqrt'])
    
    stability_threshold(ax2,
                        df_simulation,
                        ['E_ii', 'E_ij_tot'])

Stability_Plot(df,
               globally_solved_sces,
               solved_boundary)







# %%

pivot_table = pd.pivot_table(df[df['mu_c'] <= 210.0],
                             index="mu_c",
                             columns="M",
                             values="encroachment3",
                             aggfunc=lambda x: np.sum(x > 1)/len(x))

fig, ax = plt.subplots(1, 1)

sns.heatmap(pivot_table,
            vmin=0,
            vmax=1,
            ax = ax,
            cbar = True,
            cmap = 'Purples')

ax.invert_yaxis()

# %%

pivot_table1 = pd.pivot_table(df[df['mu_c'] <= 210.0],
                             index="mu_c",
                             columns="M",
                             values="E_ij_tot",
                             aggfunc="mean")

pivot_table2 = pd.pivot_table(df[df['mu_c'] <= 210.0],
                             index="mu_c",
                             columns="M",
                             values="E_ii_nonorn",
                             aggfunc="mean")


fig, (ax1, ax2) = plt.subplots(1, 2, layout="constrained")

vmin = np.min(np.concat([pivot_table1.to_numpy(),
                          pivot_table2.to_numpy()]))
vmax = np.max(np.concat([pivot_table1.to_numpy(),
                          pivot_table2.to_numpy()]))

sns.heatmap(pivot_table1,
            ax = ax1,
            vmin=vmin,
            vmax=vmax,
            square=True,
            cbar = False,
            cmap = 'Greens')

sns.heatmap(pivot_table2,
            ax = ax2,
            vmin=vmin,
            vmax=vmax,
            square=True,
            cbar = True,
            cmap = 'Greens')

ax.invert_yaxis()

# %%

fig, ax = plt.subplots(1, 1)

dfl = pd.melt(df.loc[df['mu_c'] == 145.0,
                     ['M', 'E_ii', 'E_ij_tot']],
              ['M'])

#axs['I_C'].add_patch(Rectangle((np.min(resource_pool_sizes) - 20, np.min(dfl['value']) - 0.1),
#                               M_stability_threshold - np.min(resource_pool_sizes) + 20,
##                               np.max(dfl['value']) + 0.5 - np.min(dfl['value']),
#                               fill = True, color = '#8f8cc0ff', zorder = 0))

#axs["I_C"].vlines(M_stability_threshold,
#                  np.min(dfl['value']) - 0.1,
#                  np.max(dfl['value']) + 0.1,
#                  color = 'black', linewidth = 2.5, zorder = 1)

sns.lineplot(dfl,
             x = 'M',
             y = 'value',
             hue = 'variable',
             ax = ax,
             linewidth = 3,
             palette = sns.color_palette(['black', 'black'], 2),
             zorder = 10,
             err_style="bars")

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
             markeredgecolor = 'black',
             err_style="bars")