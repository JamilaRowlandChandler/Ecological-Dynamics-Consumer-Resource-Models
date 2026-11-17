# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 21:34:35 2026

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

def plot_sces(sces,
              variable):
    
    match variable:
        
        case 'sigma_c':
            
            xlabel = "std. dev. in consumption"
            mu = 'mu_c'
        
        case 'sigma_y':
            
            xlabel = "std. dev. in yield"
            mu = 'mu_y'
    
     
    quantities = ['phi_N', 'N_mean', 'q_N',
                  'phi_R', 'R_mean', 'q_R',
                  'v_N', 'chi_R']
    
    titles = ['consumer survival fraction, ' + r'$\phi_N$',
              'average consumer abundance, ' + r'$\langle N \rangle$', 
              'fluctuations in consumer abundances, ' + r'$\langle N^2 \rangle$',
              'resource survival fraction, ' + r'$\phi_R$',
              'average resource abundance, ' + r'$\langle R \rangle$', 
              'fluctuations in resource abundances, ' + r'$\langle R^2 \rangle$',
              'Avg. susceptibility of a species to\nperturbations by the cavity ' + \
                  'variables, ' + r'$v^{(N)}$',
              'Avg. susceptibility of a resource to\nperturbations by the' + \
                  ' cavity variables, ' + r'$\chi^{(R)}$']
        
    colours = ['#08519cff', '#1f8542ff', '#cb4302ff',
               '#08519cff', '#1f8542ff', '#cb4302ff',
               '#a02f7fff', '#87216bff']
        
    sces_subset = sces[sces['M'] == 250]
    sigmas = np.round(np.unique(sces_subset[variable]), 4)
    
    sns.set_style('ticks')
    
    fig, axs = plt.subplots(3, 3,
                            figsize=(7, 5.5),
                            layout="constrained",
                            sharex=True)
    
    for ax, quantity, title, colour in zip(axs.flatten(),
                                           quantities,
                                           titles,
                                           colours):
        
        sns.lineplot(sces_subset,
                     x = variable,
                     y = quantity,
                     ax = ax,
                     linewidth = 3,
                     color = 'black',
                     zorder = 10)
        
        sns.lineplot(sces_subset,
                     x = variable,
                     y = quantity,
                     ax = ax,
                     linewidth = 2,
                     marker = 'o',
                     markersize = 8,
                     color = colour,
                     zorder = 10,
                     markeredgewidth = 0.4,
                     markeredgecolor = 'black')
       
        #ax.set_ylim([np.min(sces_subset[quantity]) - 0.02,
        #             np.max(sces_subset[quantity]) + 0.02])
        
        ax.set_xlabel(xlabel,
                      fontsize = 10,
                      weight = 'bold')
        
        ax.set_ylabel('')
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.set_xticks(sigmas[::2],
                      labels = [r'$\frac{sigma^2}{mu^2}$'.replace('sigma', str(sigma)).replace('mu', str(np.unique(sces_subset[mu])[0])) #+ '^2$' 
                                for sigma in sigmas[::2]])
        ax.set_title(title,
                     weight='bold',
                     fontsize=11)
        
    sns.despine()
    
    plt.savefig("C:/Users/jamil/Documents/PhD/Figures/resource_diversity_stability/self_limit_" + variable + "_sces.png",
                bbox_inches='tight')
    plt.savefig("C:/Users/jamil/Documents/PhD/Figures/resource_diversity_stability/self_limit_" + variable + "_sces.svg",
                bbox_inches='tight')
    
    plt.show()

# %%

directory = "C:/Users/jamil/Documents/PhD/Data/" \
                              + "resource_diversity_stability/self_consistency_equations/" 
            # "self_consistency_equations"
            
sces_c = pd.read_pickle(directory + "/M_vs_sigma_c.pkl") 
sces_y = pd.read_pickle(directory + "M_vs_sigma_y.pkl") 

    
# %%
    
plot_sces(sces_c,
          "sigma_c")

# %%

plot_sces(sces_y,
          "sigma_y")
