# -*- coding: utf-8 -*-
"""
Created on Fri Oct  3 11:51:23 2025

@author: jamil
"""

import numpy as np
import pandas as pd
import seaborn as sns
import os
import sys

from matplotlib import pyplot as plt

os.chdir('C:/Users/jamil/Documents/PhD/GitHub projects/Ecological-Dynamics-and-Community-Selection/Ecological-Dynamics/Consumer-Resource Models/resource_diversity_stability(sl)/cavity_solutions_vs_simulations')

sys.path.insert(0, "C:/Users/jamil/Documents/PhD/GitHub projects/Ecological-Dynamics-and-Community-Selection/" + \
                    "Ecological-Dynamics/Consumer-Resource Models/alternative_growth_consumption_coupling")
from simulation_functions import generic_heatmaps_multi

# %%

directory = "C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/" \
                              + "resource_diversity_stability/self_consistency_equations/" 
            # "self_consistency_equations"
            
solved_sces_mu_y = pd.read_pickle(directory + "M_vs_mu_y.pkl")
solved_sces_mu_d = pd.read_pickle(directory + "M_vs_mu_d.pkl")
solved_sces_sigma_d = pd.read_pickle(directory + "M_vs_sigma_d.pkl")
solved_sces_mu_b = pd.read_pickle(directory + "M_vs_mu_b.pkl")
solved_sces_sigma_b  = pd.read_pickle(directory + "M_vs_sigma_b.pkl")

# %%

def plot_instability_distances():
    
    sces_list = [sces.loc[sces['loss'] < -30, :] 
                 for sces in [solved_sces_mu_y,
                              solved_sces_mu_d, solved_sces_sigma_d,
                              solved_sces_mu_b, solved_sces_sigma_b]]
    
    quantities = ['mu_y', 'mu_d', 'sigma_d', 'mu_b', 'sigma_b']
    
    y_labels = ['average yield conversio\nefficiency, $\mu_y$',
                'average death rate, $\mu_d$', 
                'std deviation in death rate,\n$\sigma_d$',
                'average intrinsic resource\ngrowth rate, $\mu_b$',
                'std deviation in intrinsic\nresource growth rate, $\sigma_b$']
    
    
    for sces, quantity in zip(sces_list, quantities): 
        
        sces['Instability distance'] = sces['rho'] - np.sqrt(sces['Species packing'])
        sces[quantity] = np.round(sces[quantity], 7)
    
    id_max = np.max(np.concatenate([sces['Instability distance'].to_numpy() 
                                    for sces in sces_list]))
    id_min = np.min(np.concatenate([sces['Instability distance'].to_numpy() 
                                    for sces in sces_list]))
    
    if id_max > np.abs(id_min): id_min = -id_max 
    else: id_max = -id_min
    
    fig, axs = generic_heatmaps_multi(sces_list, 'M',
                                      quantities,
                                      "resource pool size, $M$",
                                      y_labels, "Instability distance", "RdBu",
                                      (2, 3), (8.5, 4.4),
                                      specify_min_max = np.tile([id_min, id_max],
                                                                len(sces)).reshape((len(sces), 2)),
                                      cbar_pos = 2)
    
    cbar = axs.flatten()[2].collections[0].colorbar
    cbar.set_label(label = r'$\text{reciprocity} - \sqrt{\text{packing ratio}}$',
                   size = '10')
    cbar.ax.tick_params(labelsize = 8)
    
    fig.delaxes(axs.flatten()[-1])
    
    plt.show()
    
plot_instability_distances()
