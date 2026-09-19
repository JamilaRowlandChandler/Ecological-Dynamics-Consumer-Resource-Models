# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 09:43:40 2026

@author: jamil
"""

import numpy as np
import pandas as pd
import seaborn as sns
import os
import sys
from scipy.optimize import curve_fit
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

    stability_pivot = le_pivot_r(df, index = "rho", columns = "timescalar")[0]
    
    return df, stability_pivot
    
# %%

def feasible_region(df, index = 'rho', columns = 'timescalar'):
    
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
                           feasibility_ab):
    
    def stability_plot(stability_pivot,
                       feasibility_pivot,
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
        ax.set_ylabel('reciprocity, ' + r'$\rho$',
                      fontsize = 10, weight = 'bold')
        ax.invert_yaxis()
        
        ax.set_xticks(np.arange(0.5,
                                len(epsilons) + 0.5,
                                1),
                            labels = [f"$10^{{{e}}}$" 
                                      for e in 
                                      np.int64(np.round(np.log10(1/epsilons),
                                                        1))],
                            fontsize = 10,
                            rotation = 0)
        ax.set_xlabel('timescale separation, ' + r'$1/\epsilon$',
                      fontsize = 10, weight = 'bold')
    
    ##################################
     
    rhos = stability_ab.index.to_numpy()
    epsilons = stability_ab.columns.to_numpy()
    
    sns.set_style('ticks')
    
    fig, axs = plt.subplots(1, 1,
                            figsize=(3, 2.3),
                            layout="constrained")
    
    stability_plot(stability_ab,
                   feasibility_ab,
                   axs)
    
    
    axs.set_facecolor('white')
    
    #plt.savefig("C:/Users/jamil/Documents/PhD/Figures/externally_supplied_resources/simulations_rho_sigma_large_M.png",
    #            bbox_inches='tight')
    #plt.savefig("C:/Users/jamil/Documents/PhD/Figures/externally_supplied_resources/simulations_rho_sigma_large_M.svg",
    #            bbox_inches='tight')
                 
    plt.show()
    
# %%

simulations_abiotic, stability_abiotic = load_clean_simulations("rho_timescale_es")
feasibility_abiotic = feasible_region(simulations_abiotic)

compare_abiotic_biotic(stability_abiotic,
                       feasibility_abiotic)

# %%

simulations_abiotic, stability_abiotic = load_clean_simulations("rho_timescale_es_smalldilution")
feasibility_abiotic = feasible_region(simulations_abiotic)

compare_abiotic_biotic(stability_abiotic,
                       feasibility_abiotic)

# %%

simulations_abiotic, stability_abiotic = load_clean_simulations("rho_timescale_es_largedilution")
feasibility_abiotic = feasible_region(simulations_abiotic)

compare_abiotic_biotic(stability_abiotic,
                       feasibility_abiotic)
