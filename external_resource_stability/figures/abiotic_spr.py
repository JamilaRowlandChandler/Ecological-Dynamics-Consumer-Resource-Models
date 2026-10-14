# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 11:36:26 2026

@author: jamil
"""

import numpy as np
import pandas as pd
import seaborn as sns
import os
import sys
from scipy.optimize import curve_fit

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patheffects as patheffects
from matplotlib import colors


os.chdir('C:/Users/jamil/Documents/PhD/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models/external_resource_stability/figures')

sys.path.insert(0, "C:/Users/jamil/Documents/PhD/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models/" + \
                    "external_resource_stability")
from simulation_functions import le_pivot_r

    
# %%

def load_clean_sces(filename):
    
    sces = pd.read_csv("C:/Users/jamil/Documents/PhD/Data/external_resource_stability/self_consistency_equations/" + \
                       filename + ".csv",
                       na_values=["Missing[Failed]", "Missing[]"])
    sces = np.round(sces.apply(pd.to_numeric, errors="coerce"), 7)
    
    sces['Stability Term Simple'] = 1/(4*sces['rho']**2)
    
    return sces

# %%

def species_packing(sces):
    
    ################
    
    def stability_plot(pivot,
                       title):
        
        subfig = sns.heatmap(pivot,
                             ax = ax,
                             vmin = 0,
                             vmax = 0.5,
                             cbar = True,
                             cmap = "Greens_r") #pal)
        
        subfig.axhline(0, 0, 1, color = 'black', linewidth = 2)
        subfig.axhline(pivot.shape[0], 0, 1,
                       color = 'black', linewidth = 2)
        subfig.axvline(0, 0, 1, color = 'black', linewidth = 2)
        subfig.axvline(pivot.shape[1], 0, 1,
                       color = 'black', linewidth = 2)
        
        ax.set_yticks(np.arange(0.5, len(rhos) + 0.5, 2),
                            labels = rhos[::2], fontsize = 10,
                            rotation = 0)
        ax.set_ylabel('growth-consumption\ncorrelation, ' + r'$\rho$',
                      fontsize = 10, weight = 'bold')
        ax.invert_yaxis()
        
        ax.set_xticks(np.arange(0.5, len(sigmas) + 0.5, 6), labels = sigmas[::6],
                            fontsize = 10, rotation = 0)
        ax.set_xlabel('std. dev. in growth and consumption, ' + r'$\sigma$',
                      fontsize = 10, weight = 'bold')
        
        ax.set_title(title, fontsize = 10, weight = 'bold')
             
        cbar = ax.collections[0].colorbar
        cbar.set_label(label = 'packing ratio',
                       size = '10', horizontalalignment = 'center', 
                       verticalalignment = 'top', weight = 'bold')
        cbar.ax.tick_params(labelsize = 10)
                                                                            
    
    ##################################
    
    spr_pivot = pd.pivot(sces,
                         columns="sigma_c",
                         index="rho",
                         values="Packing ratio")
     
    rhos = spr_pivot.index.to_numpy()
    sigmas = spr_pivot.columns.to_numpy()
    
    sns.set_style('ticks')
    
    fig, ax = plt.subplots(1, 1,
                            layout="constrained",
                            figsize=(4, 2.5))
    
    stability_plot(spr_pivot,
                   "Externally-supplied resources")

    plt.savefig("C:/Users/jamil/Documents/PhD/Figures/externally_supplied_resources/simulations_analytics_rho_sigma_spr.png",
                bbox_inches='tight')
    plt.savefig("C:/Users/jamil/Documents/PhD/Figures/externally_supplied_resources/simulations_analytics_rho_sigma_spr.svg",
                bbox_inches='tight')
                 
    plt.show()

# %%

sces_abiotic = load_clean_sces("rho_sigma_smaller_mu")

# %%

species_packing(sces_abiotic)
