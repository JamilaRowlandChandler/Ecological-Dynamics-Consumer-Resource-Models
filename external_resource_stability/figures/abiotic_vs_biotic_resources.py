# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 12:48:57 2026

@author: jamil
"""

import numpy as np
import pandas as pd
import seaborn as sns
import os
import sys

from matplotlib import pyplot as plt
from matplotlib.colors import colorConverter, LinearSegmentedColormap

os.chdir('C:/Users/jamil/Documents/PhD/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models/external_resource_stability/figures')

sys.path.insert(0, "C:/Users/jamil/Documents/PhD/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models/" + \
                    "external_resource_stability")
from simulation_functions import le_pivot_r

# %%

def load_clean_simulations(data_location):
    
    full_location = "C:/Users/jamil/Documents/PhD/Data/external_resource_stability/simulations/" + \
                        data_location
    
    if full_location.endswith(".csv"):
    
        df = pd.read_csv(full_location, index_col=False)
            
    else: 
       
        df = pd.concat([pd.read_csv(full_location + "/" + file, index_col=False) 
                       for file in os.listdir(full_location)],
                       axis = 0, ignore_index = True) 
        
    df = df.apply(pd.to_numeric, errors="coerce")
    
    #df.rename(columns = {"maxLe" : "Max. lyapunov exponent"}, inplace = True)
    df = np.round(df, 7)
    
    #df.loc[df["EndTime"] < np.round(np.max(df["EndTime"]), 5),
    #       "Max. lyapunov exponent"] = np.nan
    df.loc[df["Divergence measure"] < np.round(np.max(df["Divergence measure"]), 5),
           "Max. lyapunov exponent"] = np.nan

    stability_pivot = le_pivot_r(df, index = "rho", columns = "sigma_c")[0]
    
    return df, stability_pivot
    
# %%

def load_clean_sces(filename):
    
    sces = pd.read_csv("C:/Users/jamil/Documents/PhD/Data/external_resource_stability/self_consistency_equations/" + \
                       filename + ".csv",
                       na_values=["Missing[Failed]", "Missing[]"])
    sces = sces.apply(pd.to_numeric, errors="coerce")
    
    return sces

# %%

def feasible_region(df, index = 'rho', columns = 'sigma_c'):
    
    def prop_feasible(x,
                      feasibility_threshold = 1000.0):
        
        return np.count_nonzero(x == feasibility_threshold)/len(x)

        
    return pd.pivot_table(df,
                          index = index,
                          columns = columns,
                          values = "Divergence measure", #'EndTime',
                          aggfunc = prop_feasible)

# %%

def compare_abiotic_biotic(stability_ab,
                           feasibility_ab,
                           stability_b,
                           feasibility_b,
                           example_simulations,
                           sces):
    
    def stability_plot(stability_pivot,
                       feasibility_pivot,
                       title,
                       ax):
        
        cmap_stable = LinearSegmentedColormap.from_list("cmap_feasible",
                                                        [(0.0, colorConverter.to_rgba('#30007dff', alpha = 1)),
                                                         #(0.999, colorConverter.to_rgba('#e4e4e4ff', alpha=1)),
                                                         (1.0, colorConverter.to_rgba('white', alpha=0))]) 
      
        cmap_feasible = LinearSegmentedColormap.from_list("cmap_feasible",
                                                          [(0.0, colorConverter.to_rgba('#595959ff', alpha = 1)),
                                                           #(0.999, colorConverter.to_rgba('#e4e4e4ff', alpha=1)),
                                                           (1.0, colorConverter.to_rgba('white', alpha=0))]) 
        
        subfig = sns.heatmap(stability_pivot,
                             ax = ax,
                             vmin = 0, vmax = 1,
                             cbar = False,
                             cmap = cmap_stable)
    
        sns.heatmap(feasibility_pivot,
                    ax = ax,
                    vmin = 0, vmax = 1,
                    cbar = False,
                    cmap = cmap_feasible)
        
        subfig.axhline(0, 0, 1, color = 'black', linewidth = 2)
        subfig.axhline(stability_pivot.shape[0], 0, 1,
                       color = 'black', linewidth = 2)
        subfig.axvline(0, 0, 1, color = 'black', linewidth = 2)
        subfig.axvline(stability_pivot.shape[1], 0, 1,
                       color = 'black', linewidth = 2)
        
        ax.set_yticks(np.arange(0.5, len(rhos) + 0.5, 2),
                            labels = rhos[::2], fontsize = 10,
                            rotation = 0)
        ax.set_ylabel('growth-consumption\ncorrelation, ' + r'$\rho$',
                      fontsize = 10, weight = 'bold')
        ax.invert_yaxis()
        
        ax.set_xticks(np.arange(0.5, len(sigmas) + 0.5, 4), labels = sigmas[::4],
                            fontsize = 10, rotation = 0)
        ax.set_xlabel('std. dev. in growth and consumption, ' + r'$\sigma$',
                      fontsize = 10, weight = 'bold')
        
        ax.set_title(title, fontsize = 10, weight = 'bold')
             
        #cbar = ax.collections[0].colorbar
        #cbar.set_label(label = 'Prob. (stability)',
        #               size = '10', horizontalalignment = 'center', 
        #               verticalalignment = 'top', weight = 'bold')
        #cbar.ax.tick_params(labelsize = 10)
        
    def example_dynamics(df, sub_axs):
    
        def indices_and_cmaps(M):
            
            v_index = np.arange(M)
            
            colour_index = np.arange(M)
            np.random.shuffle(colour_index)
            
            cmap = LinearSegmentedColormap.from_list('custom YlGBl',
                                                     ['#e9a100ff','#1fb200ff',
                                                      '#1f5a00ff','#00e9e9ff','#001256fd'],
                                                     N = M)
            
            return [v_index, colour_index, cmap]
        
        def plot_dynamics(t, y,
                          ax,
                          i_c_rp_M):
            
            var_pos, colour_index, cmap = i_c_rp_M
            
            for i, v in zip(colour_index, var_pos):
            
                ax.plot(t, y[v,:].T, color = 'black', linewidth = 1.2)
                ax.plot(t, y[v,:].T, color = cmap(i), linewidth = 1.1)
            
                #ax.set_xticks([])
                #ax.set_yticks([])
                #ax.set_xticklabels([])
                #ax.set_yticklabels([])
                
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            ax.tick_params(axis='both', which='major', labelsize=8)
            ax.tick_params(axis='both', which='minor', labelsize=8)
            
            return ax
        
        times = df.ODE_sols[0].t
        
        resource_dynamics = df.ODE_sols[0].y[df.no_resources:]
        consumer_dynamics = df.ODE_sols[0].y[:df.no_resources]
        
        #times = df['t'].to_numpy()
        
        #resource_dynamics = df.filter(regex='resource').to_numpy().T
        #consumer_dynamics = df.filter(regex='resource').to_numpy().T
        
        i_c_rp = [indices_and_cmaps(M) for M in [resource_dynamics.shape[0], 
                                                 consumer_dynamics.shape[0]]]

        for t, y, ax, i_c_rp_M in \
            zip([times, times],
                [resource_dynamics, consumer_dynamics],
                sub_axs,
                i_c_rp):
            
            plot_dynamics(t, y, ax, i_c_rp_M)
            sns.despine(ax = ax)
            
        axs["DA_UR"].set_xlim([-0.5, 12.5])
        axs["DA_UC"].set_xlim([-0.5, 12.5])
    
    ##################################
     
    rhos = stability_ab.index.to_numpy()
    sigmas = stability_ab.columns.to_numpy()
    
    sns.set_style('ticks')
    
    #mosaic = [["PA", ".", "DA_SR", ".", "DA_UR"],
    #          ["PA", ".", "DA_SC", ".", "DA_UC"],
    #          [".", ".", ".", ".", "."],
    #          ["PB", ".", "DB_SR", ".", "DB_UR"],
    #          ["PB", ".", "DB_SC", ".", "DB_UC"]]
    
    #mosaic = [["PA", "PA", "PA", ".", ".", "PB", "PB", "PB", "."],
    #          [".", ".", ".", ".", ".", ".", ".", ".", "."],
    #          ["DA_SR", ".", "DA_UR",  ".", "DB_SR", "DB_SR", "DB_CR", "DB_UR", "DB_UR"],
    #          ["DA_SC", ".", "DA_UC", ".",  "DB_SC", "DB_SC", "DB_CC", "DB_UC", "DB_UC"]]
    
    mosaic = [["PA", "PA", ".", "PB", "PB"],
              [".", ".", ".", ".", "."],
              ["DA_SR", "DA_SC",  ".", "DB_SR", "DB_SC"],
              [".", ".",  ".", "DB_CR", "DB_CC"],
              ["DA_UR", "DA_UC",  ".", "DB_UR", "DB_UC"]]
       
    fig, axs = plt.subplot_mosaic(mosaic, figsize = (6, 5), #(8, 5),
                                  width_ratios = [1, 1, 0.5, 1, 1], #[1, 0.4, 1, 0.5, 0.3, 0.7, 1, 0.7, 0.3], #,
                                  height_ratios = [2, 1, 1, 1, 1], #[2, 0.7, 0.7, 0.7],  #,
                                  gridspec_kw = {'hspace' : 0.0, 'wspace' : 0.2})
    
    stability_plot(stability_ab,
                   feasibility_ab,
                   "Externally-supplied resources",
                   axs["PA"])
    
    stability_plot(stability_b,
                   feasibility_b,
                   "Self-limiting resources",
                   axs["PB"])
    
    axs["PA"].set_facecolor('white')
    axs["PB"].set_facecolor('white')
         
    ####################### Example population dynamics ######################
    
    for dynamics, axR, axC in zip(example_simulations, 
                                  ["DA_SR", "DA_UR", "DB_SR", "DB_CR", "DB_UR"],
                                  ["DA_SC", "DA_UC", "DB_SC", "DB_CC", "DB_UC"]):
        
        example_dynamics(dynamics[0], [axs[axR], axs[axC]])

    plt.savefig("C:/Users/jamil/Documents/PhD/Figures/externally_supplied_resources/simulations_rho_sigma_large_mu.png",
                bbox_inches='tight')
    plt.savefig("C:/Users/jamil/Documents/PhD/Figures/externally_supplied_resources/simulations_rho_sigma_large_mu.svg",
                bbox_inches='tight')
                 
    plt.show()

# %%

simulations_abiotic, stability_abiotic  = load_clean_simulations("rho_sigma_mu50_es")
simulations_biotic, stability_biotic = load_clean_simulations("rho_sigma_mu50_sl")

sces_abiotic = load_clean_sces("rho_sigma_newprotocol_upd4")
sces_abiotic = np.round(sces_abiotic, 7)

stable_es = pd.read_pickle("C:/Users/jamil/Documents/PhD/Data/external_resource_stability/simulations/rho_sigma_mu50_es_examplesim/simulations_1.0_6.0.pkl")
infeasible_es = pd.read_pickle("C:/Users/jamil/Documents/PhD/Data/external_resource_stability/simulations/rho_sigma_mu50_es_examplesim/simulations_0.2_6.0.pkl")

stable_sl = pd.read_pickle("C:/Users/jamil/Documents/PhD/Data/external_resource_stability/simulations/rho_sigma_mu50_sl_examplesim/simulations_1.0_6.0.pkl")
chaotic_sl = pd.read_pickle("C:/Users/jamil/Documents/PhD/Data/external_resource_stability/simulations/rho_sigma_mu50_sl_examplesim/simulations_0.8_6.0.pkl")
infeasible_sl = pd.read_pickle("C:/Users/jamil/Documents/PhD/Data/external_resource_stability/simulations/rho_sigma_mu50_sl_examplesim/simulations_0.2_6.0.pkl")

feasibility_abiotic = feasible_region(simulations_abiotic)
feasibility_biotic = feasible_region(simulations_biotic)

# %%

compare_abiotic_biotic(stability_abiotic,
                       feasibility_abiotic,
                       stability_biotic,
                       feasibility_biotic,
                       [stable_es, infeasible_es,
                        stable_sl, chaotic_sl, infeasible_sl],
                       sces_abiotic)

# %%

example_rho = 0.7

sces_subset = sces_abiotic.iloc[np.where((np.round(sces_abiotic['rho'], 4) == example_rho) & \
                                 (np.isnan(sces_abiotic['loss']) == False))]

dfl = pd.melt(sces_subset[['sigma_c', 'Stability Term', 'Packing ratio']], ['sigma_c'])

sns.lineplot(dfl, x = 'sigma_c', y = 'value', hue = 'variable',
             linewidth = 2.5, marker = 'o', markersize = 8,
             palette = sns.color_palette(['#00557aff', '#3dc27aff'], 2),
             zorder = 10, markeredgewidth = 0.4, markeredgecolor = 'black')
plt.show()

