# -*- coding: utf-8 -*-
"""
Created on Sat Nov  1 20:53:32 2025

@author: jamil
"""

import numpy as np
import pandas as pd
import seaborn as sns
import os
from copy import deepcopy

from matplotlib import pyplot as plt

os.chdir('C:/Users/jamil/Documents/PhD/GitHub projects/Ecological-Dynamics-and-Community-Selection/Ecological-Dynamics/Consumer-Resource Models/resource_diversity_stability(sl)/effective_gLV')

# %%

def interaction_statistics(data, variable, fixed_var, fixed_var_val):
    
    potential_terms = np.array(['M', 'mu_c', 'sigma_c', 'mu_y', 'sigma_y',
                                'phi_N', 'phi_R', 'N_mean'])
    
    actual_terms = potential_terms[potential_terms != fixed_var]
    
    gLV_rev_terms = data.loc[data[fixed_var] == fixed_var_val,
                             actual_terms].groupby(variable).apply('mean')
    gLV_rev_terms.reset_index(inplace=True)
    gLV_rev_terms[fixed_var] = fixed_var_val
    
    #####
    
    M = gLV_rev_terms['M']
    mu_c = gLV_rev_terms['mu_c']
    sigma_c = gLV_rev_terms['sigma_c']
    mu_y = gLV_rev_terms['mu_y']
    sigma_y = gLV_rev_terms['sigma_y']
    
    phi_R = gLV_rev_terms['phi_R']
    phi_N = gLV_rev_terms['phi_N'] 
    N_mean =  gLV_rev_terms['N_mean']
    
    #####
    
    A_ii = mu_y * (mu_c**2/M + sigma_c**2)
    A_ij = mu_y * (mu_c**2/M)
    diff_Aii_Aij = A_ii - A_ij

    Aii_Ni = A_ii * N_mean
    sum_Aij_Nj = phi_N * phi_R * (mu_y * mu_c**2) * N_mean
    
    sigma_A =  np.sqrt(((sigma_c**4 * (mu_y**2 + sigma_y**2))/M + \
                        (2 * mu_c**2 * sigma_c**2)/M**2 + \
                        (mu_c**4 * sigma_y**2)/M**3))

    sum_sigma_A = np.sqrt(phi_N * M) * np.sqrt(((sigma_c**4 * (mu_y**2 + sigma_y**2))/M + \
                                                (2 * mu_c**2 * sigma_c**2)/M**2 + \
                                                    (mu_c**4 * sigma_y**2)/M**3))
        
    #####
        
    interaction_df = pd.DataFrame({variable : gLV_rev_terms[variable],
                                   'A_ii' : A_ii,
                                   'A_ij' :np.repeat(A_ij, len(A_ii)),
                                   'diff_Aii_Aij' : diff_Aii_Aij,
                                   'Aii_Ni' : Aii_Ni,
                                   'sum_Aij_Nj' : sum_Aij_Nj,
                                   'sum_sigma_A' : sum_sigma_A,
                                   'sigma_A' : sigma_A})

    return interaction_df

# %%

# sigma_c and sigma_y

sigma_c_M = pd.read_pickle("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/" \
                                      + "resource_diversity_stability/self_consistency_equations/M_vs_sigma_c.pkl") 

sigma_y_M = pd.read_pickle("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/" \
                                      + "resource_diversity_stability/self_consistency_equations/M_vs_sigma_y.pkl") 

# %%

sigma_c_interaction = interaction_statistics(sigma_c_M, "sigma_c", "M", 200)

sigma_y_interaction = interaction_statistics(sigma_y_M, "sigma_y", "M", 200)

sigma_c_Aii_sigmaA = pd.melt(sigma_c_interaction[['sigma_c',
                                                  'A_ii',
                                                  'sigma_A']],
                             ['sigma_c']) 
sigma_y_Aii_sigmaA = pd.melt(sigma_y_interaction[['sigma_y',
                                                  'A_ii',
                                                  'sigma_A']],
                             ['sigma_y']) 

sigma_c_rel = pd.melt(sigma_c_interaction[['sigma_c',
                                           'diff_Aii_Aij',
                                           'sum_sigma_A']],
                      ['sigma_c']) 
sigma_y_rel = pd.melt(sigma_y_interaction[['sigma_y',
                                           'diff_Aii_Aij',
                                           'sum_sigma_A']],
                      ['sigma_y']) 

# %%

fig, axs = plt.subplots(2, 2, figsize = (8.3, 5), layout = "constrained")

for ax, df, var, xlabel in zip(axs.flatten(), 
                               [sigma_c_Aii_sigmaA, sigma_y_Aii_sigmaA,
                                sigma_c_rel, sigma_y_rel],
                               ['sigma_c', 'sigma_y', 'sigma_c', 'sigma_y'],
                               ['std. dev. in consumption, ' + r'$\sigma_c$',
                                'std. dev. in yield conversion, ' + r'$\sigma_y$',
                                'std. dev. in consumption, ' + r'$\sigma_c$',
                                'std. dev. in yield conversion, ' + r'$\sigma_y$']):

    sns.lineplot(df, x = var, y = 'value', hue = 'variable', ax = ax,
                 linewidth = 2.5, marker = 'o', markersize = 8,
                 palette = sns.color_palette(['#00557aff', '#3dc27aff'], 2),
                 zorder = 10, markeredgewidth = 0.4, markeredgecolor = 'black')
    
    ax.set_xlabel(xlabel, fontsize = 11, weight = 'bold')
    ax.set_ylabel('')
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    ax.set_title("", fontsize = 11, weight = 'bold', horizontalalignment = 'left', x = 0)

axs.flatten()[0].get_legend().remove()
axs.flatten()[2].get_legend().remove()

h_handles1, _ = axs.flatten()[1].get_legend_handles_labels()
h_handles2, _ = axs.flatten()[3].get_legend_handles_labels()

axs.flatten()[1].legend(handles = h_handles1,
                      title = '',
                      labels=['(avg.) self-inhibition, ' + r'$\langle A_{ii} \rangle$', 
                              'std. dev. in inter-species\ninteractions, ' + r'$\sigma_A$'],
                          fontsize = 11,
                          bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0)
    
axs.flatten()[3].legend(handles = h_handles1,
                      title = '',
                      labels=['diff. between self-inhibition &\ninter-species interactions,\n' + \
                              r'$\langle A_{ii} \rangle - \langle A_{ij} \rangle = \mu_y \sigma_c^2$', 
                              'total std. dev. in inter-species\ninteractions, ' + \
                                  r'$\sigma_A \sqrt{S}$'],
                          fontsize = 11,
                          bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0)

sns.despine()

plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/sigma_cy_egLV_stats.png",
            bbox_inches='tight')
    
plt.show()
    


