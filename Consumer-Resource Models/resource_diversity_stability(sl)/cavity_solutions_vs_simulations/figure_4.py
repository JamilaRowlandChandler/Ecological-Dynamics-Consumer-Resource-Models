# -*- coding: utf-8 -*-
"""
Created on Fri Oct  3 11:50:49 2025

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

os.chdir('C:/Users/jamil/Documents/PhD/GitHub projects/Ecological-Dynamics-and-Community-Selection/Ecological-Dynamics/Consumer-Resource Models/resource_diversity_stability(sl)/cavity_solutions_vs_simulations')

sys.path.insert(0, "C:/Users/jamil/Documents/PhD/GitHub projects/Ecological-Dynamics-and-Community-Selection/" + \
                    "Ecological-Dynamics/Consumer-Resource Models/resource_diversity_stability(sl)")
from simulation_functions import generate_simulation_df, le_pivot_r

# %%

################################ sigma_c ################################

df_simulation_c = generate_simulation_df("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/" \
                                         + 'resource_diversity_stability/simulations/M_vs_sigma_c')
#pd.read_pickle("simulations/M_vs_sigma_c.pkl")
globally_solved_sces_c = pd.read_pickle("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/" \
                                      + "resource_diversity_stability/self_consistency_equations/M_vs_sigma_c.pkl") 
#pd.read_pickle("self_consistency_equations/M_vs_sigma_c.pkl")
solved_boundary_c = pd.read_pickle("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/" \
                                 + "resource_diversity_stability/self_consistency_equations/stability_bound/M_vs_sigma_c.pkl")
#pd.read_pickle("self_consistency_equations/M_vs_sigma_c_stable_bound.pkl")

# %%

################################ sigma_y ################################

df_simulation_y = generate_simulation_df("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/" \
                                         + 'resource_diversity_stability/simulations/M_vs_sigma_y') 
#pd.read_pickle("self_consistency_equations/M_vs_sigma_y.pkl")
globally_solved_sces_y = pd.read_pickle("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/" \
                                        + "resource_diversity_stability/self_consistency_equations/M_sigma_y.pkl") 
#pd.read_pickle("self_consistency_equations/M_vs_sigma_y.pkl")
solved_boundary_y = pd.read_pickle("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/" \
                                 + "resource_diversity_stability/self_consistency_equations/stability_bound/M_vs_sigma_y.pkl")
#pd.read_pickle("self_consistency_equations/M_vs_sigma_y_stable_bound.pkl")

# %%

############# Phase diagram + analytically-derived boundary #####################

def Stability_Plot(df_simulation_c, globally_solved_sces_c, solved_boundary_c,
                   df_simulation_y, globally_solved_sces_y, solved_boundary_y):
    
    def hyperbolic_fit(variable, boundary):
        
        fit_p, _ = curve_fit(hyperbolic, boundary['M'], boundary[variable],
                             bounds = [0, [1e6, 1e6]])
        
        return fit_p
    
    def hyperbolic(x, a, b):
        
        return a + b/x
    
    ###################
    
    def quadratic_fit(variable, boundary):
        
        smoother = np.poly1d(np.polyfit(boundary['M'], boundary[variable], 2))
        
        return smoother
    
    ######################
    
    def stability_diagram(ax, variable, pivot, sigmas, ylabel,
                          boundary, boundary_method, cbar = True):
        
        subfig = sns.heatmap(pivot, ax = ax, vmin = 0, vmax = 1,
                             cbar = cbar, cbar_ax = axs["cbar"], cmap = 'Purples_r')
            
        subfig.axhline(0, 0, 1, color = 'black', linewidth = 2)
        subfig.axhline(pivot.shape[0], 0, 1,
                       color = 'black', linewidth = 2)
        subfig.axvline(0, 0, 1, color = 'black', linewidth = 2)
        subfig.axvline(pivot.shape[1], 0, 1,
                       color = 'black', linewidth = 2)
        
        ax.set_xticks(np.arange(0.5, len(resource_pool_sizes) + 0.5, 2),
                      labels = resource_pool_sizes[::2], fontsize = 6,
                      rotation = 0)

        ax.set_yticks(np.arange(0.5, len(sigmas) + 0.5, 2), 
                      labels = [r'$' + str(sigma) + '^2$' 
                                for sigma in sigmas[::2]], fontsize = 6,
                      rotation = 0)
        
        ax.set_xlabel('resource pool size, ' + r'$M$', fontsize = 10,
                            weight = 'bold')
        ax.set_ylabel(ylabel, fontsize = 10, weight = 'bold')
        ax.invert_yaxis()
        
        if cbar == True:
            
            cbar = ax.collections[0].colorbar
            cbar.set_label(label = 'Proportion of simulations with stable dynamics',
                           size = '8', horizontalalignment = 'center', 
                           verticalalignment = 'top')
            cbar.ax.tick_params(labelsize = 6)
        
        # Analytically-derived stability boundary
        
        good_solves = boundary.loc[boundary['loss'] <= -28, :]
        
        smoothed_x = np.arange(np.min(resource_pool_sizes) - 25,
                               np.max(resource_pool_sizes) + 25,
                               1)
        
        boundary_x = (0.5 - (np.min(resource_pool_sizes) - np.min(smoothed_x))/25) + \
                np.arange(0, len(smoothed_x), 1)/25
        
        match boundary_method:
            
            case "hyperbolic":
                
                fit_p = hyperbolic_fit(variable, good_solves)
                fitted_boundary = hyperbolic(smoothed_x, *fit_p)
            
            case "quadratic":
                
                smoother = quadratic_fit(variable, good_solves)
                fitted_boundary = smoother(smoothed_x)
                
        #breakpoint()
    
        boundary_norm = fitted_boundary - np.min(sigmas)
        divider = np.unique(np.round(np.abs(np.diff(sigmas)), 8))
        boundary_y = (1/divider) * boundary_norm + 0.5
        
        sns.lineplot(x = boundary_x, y = boundary_y, ax = ax, color = 'black',
                     linewidth = 3)
        
    def sigma_vs_stability(ax, variable, pivot, sigmas, xlabel,
                           example_M = 225):
        
        stability = pivot.loc[:, example_M].to_frame()
        stability.reset_index(inplace = True)
        stability.rename(columns = {example_M : 'P(stability)'}, inplace = True)
        
        sns.lineplot(data = stability, x = variable, y = 'P(stability)',
                     ax = ax, linewidth = 3, color = 'black',
                     err_style = "bars", errorbar = ("pi", 100),
                     marker = "o", markersize = 9)

        ax.set_xticks(sigmas[::2], labels = sigmas[::2], fontsize = 10, 
                      rotation = 0)
        
        ax.yaxis.set_tick_params(labelsize = 10)
        
        ax.set_xlabel(xlabel, fontsize = 10, weight = 'bold')
        ax.set_ylabel('Probability(stability)', fontsize = 10, weight = 'bold')
        
        sns.despine(ax = ax)
        
    ############################################
    
    def stability_condition(ax, variable, sigmas, sces, boundary, boundary_method,
                            xlabel, example_M = 225):

        df_plot = sces.loc[sces['M'] == example_M, :]
        
        dfl = pd.melt(df_plot[[variable, 'rho', 'Species packing']], [variable])
        
        dfl.loc[dfl['variable'] == 'Species packing', 'value'] = \
            np.sqrt(dfl.loc[dfl['variable'] == 'Species packing', 'value'])
            
        good_solves = boundary.loc[boundary['loss'] <= -28, :]
            
        match boundary_method:
            
            case "hyperbolic":
                
                fit_p = hyperbolic_fit(variable, good_solves)
                fitted_boundary = hyperbolic(example_M, *fit_p)
            
            case "quadratic":
                
                smoother = quadratic_fit(variable, good_solves)
                fitted_boundary = smoother(example_M)
                
        if variable == "sigma_c":
    
            ax.add_patch(Rectangle((np.min(sigmas) - 0.1, np.min(dfl['value']) - 0.1),
                                   fitted_boundary - np.min(sigmas) + 0.1,
                                   np.max(dfl['value']) - np.min(dfl['value']) + 0.12,
                                   fill = True, color = '#6950a3ff', zorder = 0))
            
            ax.set_xlim([np.min(sigmas) - 0.1, np.max(sigmas) + 0.1])
            
        elif variable == "sigma_y":
            
            ax.add_patch(Rectangle((fitted_boundary, np.min(dfl['value']) - 0.1),
                                   np.max(sigmas) + 0.1 - fitted_boundary,
                                   np.max(dfl['value']) + 0.12 - np.min(dfl['value']),
                                   fill = True, color = '#6950a3ff', zorder = 0))
            
            ax.set_xlim([np.min(sigmas) - 0.01, np.max(sigmas) + 0.01])
        
        ax.vlines(fitted_boundary, np.min(dfl['value']) - 0.02,
                  np.max(dfl['value']) + 0.02,
                  color = 'black', linewidth = 2.5, zorder = 0)
        
        
        subfig1 = sns.lineplot(dfl, x = variable, y = 'value', hue = 'variable',
                               ax = ax, linewidth = 3,
                               palette = sns.color_palette(['black', 'black'], 2),
                               zorder = 10)
    
        subfig2 = sns.lineplot(dfl, x = variable, y = 'value', hue = 'variable',
                               ax = ax, linewidth = 2, marker = 'o', markersize = 8,
                               palette = sns.color_palette(['#00557aff', '#3dc27aff'], 2),
                               zorder = 10, markeredgewidth = 0.4, markeredgecolor = 'black')
        
        ax.set_ylim([np.min(dfl['value']) - 0.02, np.max(dfl['value']) + 0.02])
        
        ax.set_xlabel(xlabel, fontsize = 10, weight = 'bold')
        ax.set_ylabel('')
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.set_xticks(sigmas[::2], labels = [str(sigma) #+ '^2$' 
                                             for sigma in sigmas[::2]])
        
        ax.legend_.remove()
        

    #############################################################################################
    
    df_simulation_c = df_simulation_c.loc[df_simulation_c['sigma_c'] > 0.75, :]
    globally_solved_sces_c = globally_solved_sces_c.loc[globally_solved_sces_c['sigma_c'] > 0.75, :]
    solved_boundary_c = solved_boundary_c.loc[solved_boundary_c['sigma_c'] > 0.75, :]
    
    df_simulation_y = df_simulation_y.loc[df_simulation_y['sigma_y'] < 0.225, :]
    globally_solved_sces_y = globally_solved_sces_y.loc[globally_solved_sces_y['sigma_y'] < 0.225, :]
    solved_boundary_y = solved_boundary_y.loc[solved_boundary_y['sigma_y'] < 0.225, :]
    
    resource_pool_sizes = np.unique(df_simulation_c['M'])
    sigma_cs = np.unique(df_simulation_c['sigma_c'])
    sigma_ys = np.unique(df_simulation_y['sigma_y'])
    
    sns.set_style('ticks')
  
    mosaic = [["P_c", ".", "I_C_c"],
              ["P_y", ".", "I_C_y"]]
    
    fig, axs = plt.subplot_mosaic(mosaic, figsize = (6.5, 5.4),
                                  width_ratios = [5.2, 1.8, 5.4],
                                  height_ratios = [1, 1],
                                  gridspec_kw = {'hspace' : 0.5, 'wspace' : 0.1})
    
    ######################## Stability diagram ######################################
    
    # Simulation data
    
    stability_pivot_c = le_pivot_r(df_simulation_c, columns = 'M',
                                   index = 'sigma_c')[0]
    
    stability_pivot_y = le_pivot_r(df_simulation_y, columns = 'M',
                                     index = 'sigma_y')[0]
        
    #stability_diagram(axs["P_c"], 'sigma_c', stability_pivot_c, sigma_cs,
    #                  'variance in total consumption\nrate, ' + r'$\sigma_c^2$',
    #                  solved_boundary_c, "hyperbolic", cbar = False) 
    
    #stability_diagram(axs["P_y"], 'sigma_y', stability_pivot_y, sigma_ys,
    #                  'variance in yield conversion,\n' + r'$\sigma_y^2$',
    #                  solved_boundary_y, "quadratic") 
    
    sigma_vs_stability(axs["P_c"], 'sigma_c', stability_pivot_c, sigma_cs,
                       'std. deviation in total\nconsumption rate, ' + r'$\sigma_c$')
    
    sigma_vs_stability(axs["P_y"], 'sigma_y', stability_pivot_y, sigma_ys,
                       'std. deviation in yield\nconversion, ' + r'$\sigma_y$')
    
    axs["P_c"].set_title("Different sources of interaction heterogeneity" + \
                          "\ninduce opposing stability transitions",
                          fontsize = 11, weight = "bold", y = 1.1)
        
    #axs["P_c"].set_xlabel("")
        
    #################### Instability condition vs M #####################
    
    stability_condition(axs["I_C_c"], 'sigma_c', sigma_cs, globally_solved_sces_c,
                        solved_boundary_c, "hyperbolic",
                        'std. deviation in total\nconsumption rate, ' + r'$\sigma_c$')
    
    stability_condition(axs["I_C_y"], 'sigma_y', sigma_ys, globally_solved_sces_y,
                        solved_boundary_y, "quadratic",
                        'std. deviation in yield\nconversion, ' + r'$\sigma_y$')
    
    axs["I_C_c"].set_title("by having opposing " + \
                           "effects on\nreciprocity and the packing ratio",
                           fontsize = 11, weight = "bold", y = 1.1)
    
    ###############################################
    
    plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/self_limit_M_vs_sigmas_digram_condition.png",
                bbox_inches='tight')
    plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/self_limit_M_vs_sigmas_digram_condition.svg",
                bbox_inches='tight')
    
    plt.show()

Stability_Plot(df_simulation_c, globally_solved_sces_c, solved_boundary_c,
               df_simulation_y, globally_solved_sces_y, solved_boundary_y)
