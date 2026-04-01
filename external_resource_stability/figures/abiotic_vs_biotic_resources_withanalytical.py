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
from scipy.optimize import curve_fit

from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import matplotlib.patheffects as patheffects

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
    
    df.rename(columns = {"maxLe" : "Max. lyapunov exponent"}, inplace = True)
    df = np.round(df, 7)

    stability_pivot = le_pivot_r(df, index = "rho", columns = "sigma_c")[0]
    
    return df, stability_pivot
    
# %%

def load_clean_sces(filename):
    
    sces = pd.read_csv("C:/Users/jamil/Documents/PhD/Data/external_resource_stability/self_consistency_equations/" + \
                       filename + ".csv",
                       na_values=["Missing[Failed]", "Missing[]"])
    sces = np.round(sces.apply(pd.to_numeric, errors="coerce"), 7)
    
    return sces

# %%

def feasible_region(df, index = 'rho', columns = 'sigma_c'):
    
    def prop_feasible(x,
                      feasibility_threshold = 200.0):
        
        return np.count_nonzero(x == feasibility_threshold)/len(x)

        
    return pd.pivot_table(df, index = index, columns = columns,
                          values = 'EndTime', aggfunc = prop_feasible)

# %%

def compare_abiotic_biotic(stability_ab,
                           stability_b,
                           sces,
                           example_rho=0.8):
    
    ################
    
    def ndimension_fit(xvals, yvals,
                       dim = 10):
        
        smoother = np.poly1d(np.polyfit(xvals, yvals, dim))
        
        return smoother
    
    def log_fit(xvals, yvals):
        
        fit_p, _ = curve_fit(log,
                             xvals, yvals,
                             bounds = [0, 1e6])
        
        return fit_p
    
    def log(x,
            a, b):
        
        return a + b*np.log(x) #a - b/x
    
    def fitted_threshold(x,
                         stability_distance = 'stability_distance'):
        
        smoothed_x = np.arange(np.min(x['sigma_c']),
                               np.max(x['sigma_c']) + 2,
                               0.01)
        
        smoother = ndimension_fit(x['sigma_c'].to_numpy(),
                                  x[stability_distance].to_numpy())
        
        stability_threshold = smoothed_x[np.abs(smoother(smoothed_x)).argmin()]
        
        return stability_threshold
    
    ################
    
    def stability_plot(pivot,
                       title,
                       ax):
        
        subfig = sns.heatmap(pivot, ax = ax,
                             vmin = 0, vmax = 1, cbar = True, cmap = 'Purples_r')
        
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
        cbar.set_label(label = 'Prob. (stability)',
                       size = '10', horizontalalignment = 'center', 
                       verticalalignment = 'top', weight = 'bold')
        cbar.ax.tick_params(labelsize = 10)
        
    def stability_boundary(sces,
                           rhos,
                           sigmas,
                           ax,
                           fit_method = "log"):
        
        sces['stability_distance'] = sces['Packing ratio'] - sces['Stability Term']
        
        sces_thresh = (sces.loc[(sces['loss'] < 10**(-2)) & (sces['rho'] < 1),
                               ['rho','sigma_c', 'stability_distance']]
                       .groupby('rho').apply(fitted_threshold, include_groups=False)
                       .to_frame("sigma_c_thresh")
                       .reset_index()
                       )
        
        match fit_method:
            
            case "polynomial":    
            
                estimated_boundary = ndimension_fit(sces_thresh['sigma_c_thresh'],
                                                   sces_thresh['rho'],
                                                   2)(sigmas)
            
            case "log":
        
                estimated_boundary = log(sigmas,
                                         *log_fit(sces_thresh['sigma_c_thresh'],
                                                  sces_thresh['rho']))
        
        y_phase = estimated_boundary - np.min(rhos)
        y_vals = (1/np.unique(np.round(np.abs(np.diff(rhos)), 7)))*y_phase + 0.5
        
        x_vals = 0.5 + np.arange(0, len(sigmas), 1)
        
        sns.lineplot(x = x_vals,
                     y = y_vals,
                     ax = ax,
                     color = 'black',
                     linewidth = 3)
        
        return sces_thresh
                                                                            
    def example_pstability(pivot,
                           sces,
                           stable_threshold,
                           example_rho,
                           axPstable,
                           axCondition,
                           axCompareCondition):
        
        stability_threshold = stable_threshold.loc[stable_threshold['rho'] == example_rho,
                                                   'sigma_c_thresh'].to_numpy()[0]
        
        sigma_labels = pivot.columns.to_numpy()[::9]
        
        example_stability = pivot.loc[example_rho, :].to_frame()
        example_stability.reset_index(inplace = True)
        example_stability.rename(columns = {example_rho : 'P(stability)'},
                                 inplace = True)
        
        axPstable.vlines(stability_threshold,
                         0, 1,
                         zorder = 1,
                         color = 'black',
                         linewidth = 3)
        
        sns.lineplot(data = example_stability,
                     x = 'sigma_c', y = 'P(stability)',
                     ax = axPstable,
                     color = '#404040ff',
                     linewidth = 1.5,
                     err_style = "bars", errorbar = ("pi", 100),
                     marker = "o")
        
        se_95 = 1.96*np.sqrt((example_stability['P(stability)'] * \
                              (1 - example_stability['P(stability)']))/30)
        
        axPstable.errorbar(x = example_stability['sigma_c'], 
                           y =  example_stability['P(stability)'],
                           yerr = se_95,
                           fmt = 'none',
                           ecolor = '#404040ff',
                           linewidth = 1.5) 
        
        axPstable.xaxis.set_tick_params(labelsize = 10)
        
        axPstable.set_xticks(sigma_labels,
                             labels = sigma_labels,
                             fontsize = 10, rotation = 0)
        
        axPstable.yaxis.set_tick_params(labelsize = 10)
        
        axPstable.set_xlabel('')
        axPstable.set_ylabel('prob. (stability)',
                             fontsize = 10, weight = 'bold',
                             verticalalignment = 'center', horizontalalignment = 'center')
        
        sns.despine(ax = axPstable)
        
        ############
        
        dfl = pd.melt(sces.loc[sces['rho'] == example_rho,
                                   ['sigma_c',
                                    'Stability Term',
                                    'Packing ratio']],
                      ['sigma_c'])
        
        axCondition.add_patch(Rectangle((stability_threshold,
                                         np.min(dfl['value']) - 0.05),
                                        np.max(dfl['sigma_c']) + 0.1 - stability_threshold,
                                        np.max(dfl['value']) + 0.15 - np.min(dfl['value']),
                                        fill = True, color = '#8f8cc0ff', zorder = 0))
        
        axCondition.vlines(stability_threshold,
                           np.min(dfl['value']) - 0.05, np.max(dfl['value']) + 0.09,
                           zorder = 1,
                           color = 'black', linewidth = 3)
        
        sns.lineplot(dfl,
                     x = 'sigma_c', y = 'value', hue = 'variable',
                     ax = axCondition, zorder = 10,
                     linewidth = 3,
                     palette = sns.color_palette(['black', 'black'], 2))
        
        sns.lineplot(dfl,
                     x = 'sigma_c', y = 'value', hue = 'variable',
                     ax = axCondition, zorder = 10,
                     palette = sns.color_palette(['#00557aff', '#3dc27aff'], 2),
                     linewidth = 2.5, 
                     marker = 'o', markersize = 8,
                     markeredgewidth = 0.4, markeredgecolor = 'black')
        
        axCondition.set_xlabel('')
        axCondition.set_xticks(sigma_labels,
                               labels = sigma_labels,
                               fontsize = 10, rotation = 0)
        axCondition.set_ylabel('')
        axCondition.set_xlim([np.min(dfl['sigma_c']) - 0.1, np.max(dfl['sigma_c']) + 0.1])
        axCondition.set_ylim([np.min(dfl['value']) - 0.03, np.max(dfl['value']) + 0.09])
        axCondition.tick_params(axis='both', which='major', labelsize=10)
        axCondition.legend_.remove()

        axCondition.text(-0.22, 0,
                        'stability term', fontsize = 10, weight = 'bold', color = '#00557aff',
                        path_effects = [patheffects.withStroke(linewidth=0.5, foreground='black')],
                        verticalalignment = 'bottom', horizontalalignment = 'left',
                        rotation = 90, transform = axCondition.transAxes)
        
        axCondition.text(-0.22, 0.5,
                         'packing ratio',
                         fontsize = 10, weight = 'bold',
                         color = '#3dc27aff', 
                         path_effects = [patheffects.withStroke(linewidth=0.5, foreground='black')],
                         verticalalignment = 'bottom', horizontalalignment = 'left',
                         rotation = 90, transform = axCondition.transAxes)
        
        axCondition.text(-0.5, -0.1,
                         'std. deviation in consumption or growth, ' + r'$\sigma$',
                         fontsize = 10, weight = 'bold',
                         verticalalignment = 'top', horizontalalignment = 'center')
        
        #############
        
        axCompareCondition.hlines(example_rho**2,
                                  np.min(dfl['sigma_c']), np.max(dfl['sigma_c']),
                                  zorder = 1,
                                  color = '#00557aff',
                                  linewidth = 4)
        
        axCompareCondition.hlines(example_rho**2,
                                  np.min(dfl['sigma_c']), np.max(dfl['sigma_c']),
                                  zorder = 2,
                                  color = '#00557aff',
                                  linewidth = 3.5)
        
        sns.lineplot(dfl[dfl['variable'] == 'Packing ratio'],
                     x = 'sigma_c', y = 'value',
                     color = 'black',
                     ax = axCompareCondition, zorder = 10,
                     linewidth = 3)
        
        sns.lineplot(dfl[dfl['variable'] == 'Packing ratio'],
                     x = 'sigma_c', y = 'value',
                     color = '#3dc27aff',
                     ax = axCompareCondition, zorder = 12,
                     linewidth = 2.5, 
                     marker = 'o', markersize = 8,
                     markeredgewidth = 0.4, markeredgecolor = 'black')
        
        axCompareCondition.set_xlabel('')
        axCompareCondition.set_xticks(sigma_labels,
                               labels = sigma_labels,
                               fontsize = 10, rotation = 0)
        axCompareCondition.set_ylabel('')
        axCompareCondition.set_ylim([np.min(dfl['value']) - 0.03, np.max(dfl['value']) + 0.1])
        axCompareCondition.tick_params(axis='both', which='major', labelsize=10)

        axCompareCondition.text(-0.22, 0,
                        'correlation' + r'${}^2$', fontsize = 10, weight = 'bold', color = '#00557aff',
                        path_effects = [patheffects.withStroke(linewidth=0.5, foreground='black')],
                        verticalalignment = 'bottom', horizontalalignment = 'left',
                        rotation = 90, transform = axCompareCondition.transAxes)
        
        axCompareCondition.text(-0.22, 0.5,
                         'packing ratio',
                         fontsize = 10, weight = 'bold',
                         color = '#3dc27aff', 
                         path_effects = [patheffects.withStroke(linewidth=0.5, foreground='black')],
                         verticalalignment = 'bottom', horizontalalignment = 'left',
                         rotation = 90, transform = axCompareCondition.transAxes)
        
    ##################################
     
    rhos = stability_ab.index.to_numpy()
    sigmas = stability_ab.columns.to_numpy()
    
    sns.set_style('ticks')
    
    mosaic = [["PA", ".", "P(stable)A", ".", "SC_A", ".", "SC_B"]] #,
              #[".", ".", ".", ".", "."],
              #["PB", ".", "P(stable)B", ".", "SC_B"]]
       
    fig, axs = plt.subplot_mosaic(mosaic, figsize = (12, 2.5), #(8, 5),
                                  width_ratios = [4.5, 0.5, 2.25, 0.5, 2.25, 0.5, 2.25],
                                  height_ratios = [1], #[4, 1, 4],
                                  gridspec_kw = {'hspace' : 0.0, 'wspace' : 0})
    
    stability_plot(stability_ab, "Externally-supplied resources", axs["PA"])
    #stability_plot(stability_b, "Self-limiting resources", axs["PB"])
    
    axs["PA"].set_facecolor('grey')
    #axs["PB"].set_facecolor('grey')
    
    stable_thresh_abiotic = stability_boundary(sces,
                                               np.unique(rhos),
                                               np.unique(sigmas),
                                               axs["PA"])
    
    example_pstability(stability_ab.loc[:, 0.2:],
                       sces,
                       stable_thresh_abiotic,
                       example_rho,
                       axs["P(stable)A"], axs["SC_A"], axs['SC_B'])
    
    plt.savefig("C:/Users/jamil/Documents/PhD/Figures/externally_supplied_resources/simulations_analytics_rho_sigma_small_mu.png",
                bbox_inches='tight')
    plt.savefig("C:/Users/jamil/Documents/PhD/Figures/externally_supplied_resources/simulations_analytics_rho_sigma_small_mu.svg",
                bbox_inches='tight')
                 
    plt.show()

# %%

simulations_abiotic, stability_abiotic  = load_clean_simulations("rho_vs_sigma_abiotic_smallmu.csv")
simulations_biotic, stability_biotic = load_clean_simulations("rho_vs_sigma_biotic_smallmu.csv")

sces_abiotic = load_clean_sces("rho_sigma_smaller_mu")

# %%

feasibility_abiotic = feasible_region(simulations_abiotic)
feasibility_biotic = feasible_region(simulations_biotic)

# %%

compare_abiotic_biotic(stability_abiotic, # .mask(feasibility_abiotic < 1),
                       stability_biotic, #.mask(feasibility_biotic < 1),
                       sces_abiotic)

