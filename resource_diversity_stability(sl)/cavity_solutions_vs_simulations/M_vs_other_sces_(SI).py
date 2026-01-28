# -*- coding: utf-8 -*-
"""
Created on Mon Jan 12 16:17:16 2026

@author: jamil
"""

import numpy as np
import pandas as pd
import seaborn as sns
import os
import sys

from matplotlib import pyplot as plt

os.chdir('C:/Users/jamil/Documents/PhD/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models/resource_diversity_stability(sl)/cavity_solutions_vs_simulations')

sys.path.insert(0, "C:/Users/jamil/Documents/PhD/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models" + \
                    "/resource_diversity_stability(sl)")
from simulation_functions import generic_heatmaps

# %%

def plot_instability_distances(sces):
    
    sces_good = np.round(sces.loc[sces['loss'] < -30, :], 7)
    
    quantities = ['phi_N', 'N_mean', 'q_N',
                  'phi_R', 'R_mean', 'q_R',
                  'v_N', 'chi_R']
    
   # quantities = ['phi_N', 'phi_R',
   #               'N_mean', 'R_mean',
   #               'q_N', 'q_R',
   #               'v_N', 'chi_R']
    
    titles = ['species survival fraction, ' + r'$\phi_N$',
              'average species abundance, ' + r'$\langle N \rangle$', 
              'fluctuations in species abundances, ' + r'$\langle N^2 \rangle$',
              'resource survival fraction, ' + r'$\phi_R$',
              'average resource abundance, ' + r'$\langle R \rangle$', 
              'fluctuations in resource abundances, ' + r'$\langle R^2 \rangle$',
              'Avg. susceptibility of a species to\nperturbations by the cavity ' + \
                  'variables, ' + r'$v^{(N)}$',
              'Avg. susceptibility of a resource to\nperturbations by the' + \
                  ' cavity variables, ' + r'$\chi^{(R)}$']
        
    #titles = ['species survival fraction, ' + r'$\phi_N$',
    #          'resource survival fraction, ' + r'$\phi_R$',
    #          'average species abundance, ' + r'$\langle N \rangle$',
    #          'average resource abundance, ' + r'$\langle R \rangle$', 
    #          'fluctuations in species abundances, ' + r'$\langle N^2 \rangle$',
    #          'fluctuations in resource abundances, ' + r'$\langle R^2 \rangle$',
    #          'Avg. susceptibility of a species to\nperturbations by the cavity ' + \
    #              'variables, ' + r'$v^{(N)}$',
    #          'Avg. susceptibility of a resource to\nperturbations by the' + \
    #              ' cavity variables, ' + r'$\chi^{(R)}$']
        
    def min_max_quantity(sces, quantity1, quantity2):
        
        vals = np.concatenate([sces[quantity1].to_numpy(),
                               sces[quantity2].to_numpy()])
        
        q_min, q_max = np.min(vals), np.max(vals)
        
        return {quantity1 : [q_min, q_max], quantity2 : [q_min, q_max]}
        
    min_maxs = min_max_quantity(sces, 'phi_N', 'phi_R') | \
               min_max_quantity(sces, 'N_mean', 'R_mean') | \
               min_max_quantity(sces, 'q_N', 'q_R')
    
    fig, axs = generic_heatmaps(sces,
                                'M', 'mu_c',
                                'resource pool size, $M$',
                                'avaerage total consumption coefficient, $\mu_c$',
                                quantities,
                                ['Blues', 'Greens', 'Oranges',
                                 'Blues', 'Greens', 'Oranges',
                                 'magma', 'inferno'],
                                #['Blues','Blues',
                                # 'Greens','Greens',
                                # 'Oranges', 'Oranges',
                                # 'magma', 'inferno'],
                                titles,
                                (3, 3),
                                #(2, 4),
                                (8.5, 7),
                                #(15, 8),
                                #specify_min_max = min_maxs,
                                mosaic = [['phi_N', 'N_mean', 'N_squared'],
                                          ['phi_R', 'R_mean', 'R_squared'],
                                          ['v_N', 'chi_R', '.']],
                                gridspec_kw = {'hspace' : 0.25,
                                               'wspace' : 0.1})
                                #)
    
    plt.savefig("C:/Users/jamil/Documents/PhD/Figures/resource_diversity_stability/self_limit_M_vs_mu_c_sces.png",
                bbox_inches='tight')
    plt.savefig("C:/Users/jamil/Documents/PhD/Figures/resource_diversity_stability/self_limit_M_vs_mu_c_sces.svg",
                bbox_inches='tight')
    
    plt.show()

# %%

directory = "C:/Users/jamil/Documents/PhD/Data/" \
                              + "resource_diversity_stability/self_consistency_equations/" 
            # "self_consistency_equations"
            
sces = pd.read_pickle(directory + "M_vs_mu_c.pkl")
    
# %%
    
plot_instability_distances(sces)