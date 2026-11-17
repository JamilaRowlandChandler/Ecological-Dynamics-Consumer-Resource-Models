# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 09:52:23 2026

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
    
    full_location = "C:/Users/jamil/Documents/PhD/Data/MiCRM_rho_vs_sigma"
    
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

def stability_diagram(df,
                      stability,
                      feasibility):
    
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
        
    def ndimension_fit(xvals,
                       yvals,
                       dim = 2):
        
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
                         not_groupby_var,
                         stability_distance = 'stability_distance'):
        
        #smoothed_x = np.arange(np.min(x['sigma_c']),
        #                       np.max(x['sigma_c']) + 2,
        #                       0.01)
        
        x[x == np.inf] = 1e5
        x[x == -np.inf] = -1e5
        
        smoothed_x = np.arange(0, 1, 0.01) \
                     if not_groupby_var == "rho" \
                     else np.arange(np.min(x[not_groupby_var]),
                                    np.max(x[not_groupby_var]) + 2,
                                    0.01)
        
        #smoother = ndimension_fit(x['sigma_c'].to_numpy(),
        smoother = ndimension_fit(x[not_groupby_var].to_numpy(),
                                  x[stability_distance].to_numpy())
        
        stability_threshold = smoothed_x[np.abs(smoother(smoothed_x)).argmin()]
        
        return stability_threshold
    
    def stability_boundary(df,
                           rhos,
                           sigmas,
                           groupby_var,
                           ax,
                           fit_method = "log"):
        
        not_groupby_var = "rho" if groupby_var == "sigma_c" else "sigma_c"
        
        df['Packing ratio'] = df['phi_N']/df['phi_R']
        df['stability_distance'] = df['rho']**2 - df['Packing ratio'] 
        
        sces_thresh = (df.loc[~np.isnan(df['Max. lyapunov exponent']),
                              ['rho','sigma_c', 'stability_distance']]
                       .groupby(['rho', 'sigma_c'])
                       .apply('mean',
                              include_groups=False)
                       .reset_index()
                       .groupby(groupby_var)
                       .apply(fitted_threshold,
                              not_groupby_var=not_groupby_var,
                              include_groups=False)
                       .to_frame(not_groupby_var)
                       .reset_index())
        
        sigma_diff = np.unique(np.abs(np.diff(sigmas)))[0]
        
        plot_sigmas = np.arange(np.min(sigmas) - sigma_diff,
                                np.max(sigmas) + 2*sigma_diff,
                                sigma_diff)
        
        match fit_method:
            
            case "polynomial":    
            
                estimated_boundary = ndimension_fit(sces_thresh['sigma_c'],
                                                   sces_thresh['rho'],
                                                   2)(plot_sigmas)
                 
            case "log":
        
                estimated_boundary = log(plot_sigmas,
                                         *log_fit(sces_thresh['sigma_c'],
                                                  sces_thresh['rho']))
                
        y_phase = estimated_boundary - np.min(rhos)
        y_vals = (1/np.unique(np.round(np.abs(np.diff(rhos)), 7)))*y_phase + 0.5
        
        x_vals = -0.5 + np.arange(0, len(plot_sigmas), 1)
        
        sns.lineplot(x = x_vals,
                     y = y_vals,
                     ax = ax,
                     color = 'black',
                     linewidth = 3)
        
    ##################################
     
    rhos = stability.index.to_numpy()
    sigmas = stability.columns.to_numpy()
    
    sns.set_style('ticks')
    
       
    fig, axs = plt.subplots(1, 1,
                            figsize = (3, 2), #(8, 5),
                            layout = "constrained")
    
    stability_plot(stability,
                   feasibility,
                   "Cross-fed resources",
                   axs)
    
    #stability_boundary(df,
    #                   rhos,
    #                   sigmas,
    #                   "rho",
    #                   axs)
    
    
    axs.set_facecolor('white')
         

    #plt.savefig("C:/Users/jamil/Documents/PhD/Figures/externally_supplied_resources/simulations_rho_sigma_large_mu.png",
    #            bbox_inches='tight')
    #plt.savefig("C:/Users/jamil/Documents/PhD/Figures/externally_supplied_resources/simulations_rho_sigma_large_mu.svg",
    #            bbox_inches='tight')
                 
    plt.show()

# %%

simulations, stability = load_clean_simulations("MiCRM_rho_vs_sigma")

feasibility = feasible_region(simulations)

# %%
stability_diagram(simulations,
                  stability,
                  feasibility)


