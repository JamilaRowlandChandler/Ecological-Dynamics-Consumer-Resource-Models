# -*- coding: utf-8 -*-
"""
Created on Tue May 12 15:59:29 2026

@author: jamil
"""

import numpy as np
import sys
import os
from matplotlib import pyplot as plt 
from matplotlib.colors import colorConverter, LinearSegmentedColormap
import pandas as pd 
import seaborn as sns

# %%

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
    
    df.rename(columns = {"maxLe" : "Max. lyapunov exponent",
                         'max. le' : "Max. lyapunov exponent",},
              inplace = True)
    df = np.round(df, 9)
    
    df.loc[df["Divergence measure"] < 1000, "Max. lyapunov exponent"] = np.nan

    return df

def stability_diagram(df,
                      index,
                      columns):
    
    return le_pivot_r(df,
                      index = index,
                      columns = columns)[0]

def feasibility_diagram(df,
                        index,
                        columns):
    
    def prop_feasible(x,
                      feasibility_threshold = 1000):
        
        return np.count_nonzero(x == feasibility_threshold)/len(x)
    
    
    return pd.pivot_table(df,
                          index = index,
                          columns = columns,
                          values = "Divergence measure",
                          aggfunc = prop_feasible)   

# %%

def influx_figure(stability_sli,
                  feasibility_sli,
                  stability_ir,
                  feasibility_ir,
                  example_sensitivities_df,
                  example_sensitivity):
    
    def stability_plot(stability_pivot,
                       feasibility_pivot,
                       ax,
                       log_x = False):
        
        cmap_stable = LinearSegmentedColormap.from_list("cmap_feasible",
                                                        [(0.0, colorConverter.to_rgba('#30007dff', alpha = 1)),
                                                         #(0.999, colorConverter.to_rgba('#e4e4e4ff', alpha=1)),
                                                         (1.0, colorConverter.to_rgba('white', alpha=0))]) 
      
        cmap_feasible = LinearSegmentedColormap.from_list("cmap_feasible",
                                                          [(0.0, colorConverter.to_rgba('#595959ff', alpha = 1)),
                                                           #(0.999, colorConverter.to_rgba('#e4e4e4ff', alpha=1)),
                                                           (1.0, colorConverter.to_rgba('white', alpha=0))]) 
        
        sns.heatmap(stability_pivot,
                    ax = ax,
                    vmin = 0, vmax = 1,
                    cbar = False,
                    cmap = cmap_stable)
    
        sns.heatmap(feasibility_pivot,
                    ax = ax,
                    vmin = 0, vmax = 1,
                    cbar = False,
                    cmap = cmap_feasible)
        
        if log_x == True:
            
            ax.set_xticks(np.arange(0.5, 
                                    len(stability_pivot.columns.to_numpy() + 0.5),
                                    2),
                          labels = [f"$10^{{{e}}}$" 
                                    for e in np.int64(stability_pivot.columns.to_numpy()[::2])],
                          #labels = stability_pivot.columns.to_numpy()[::2],
                          fontsize = 8,
                          rotation = 0)
        
        else: 
        
            ax.set_xticks(np.arange(0.5, 
                                    len(stability_pivot.columns.to_numpy() + 0.5),
                                    3),
                          labels = stability_pivot.columns.to_numpy()[::3],
                          fontsize = 8,
                          rotation = 0)
        ax.set_yticks(np.arange(0.5, 
                                len(stability_pivot.index.to_numpy() + 0.5),
                                2),
                      labels = stability_pivot.index.to_numpy()[::2],
                      fontsize = 8,
                      rotation = 0)
        ax.invert_yaxis()
        ax.set_facecolor('white')
        ax.set_xlabel('')
        ax.set_ylabel('')
        
        for _, spine in ax.spines.items(): spine.set_visible(True)
        
    def sensitivities_plot(df,
                           ax):
        
        sns.lineplot(df,
                     x = 'b',
                     y = 'max peak',
                     color = "#609800ff",
                     err_style='bars',
                     errorbar=("ci", 95),
                     linewidth = 3,
                     marker="o",
                     markersize=5,
                     ax=ax)
        
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.tick_params(axis='both', which='minor', labelsize=8)
        ax.set(xscale='log', yscale='log')
        #ax.set_ylim([10**(-10), np.ceil(np.nanmax(df['max peak']) + 150)])
        ax.set_ylim([10**(-3), np.ceil(np.nanmax(df['max peak']))])
        
    def example_sensitivities_plot(df,
                                   example_sensitivity, 
                                   ax):
        
        sns.lineplot(x = example_sensitivity['x'],
                     y = example_sensitivity['dRdx2'],
                     color = 'black',
                     marker="o",
                     markersize=5,
                     linewidth = 2,
                     ax=ax)
        
        xlabels = np.round(np.arange(0.9, 1.5, 0.1), 7)

        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticks(xlabels,
                      labels = xlabels,
                      fontsize = 8,
                      rotation = 0)
        ax.tick_params(axis='y', which='major', labelsize=8)
        ax.tick_params(axis='y', which='minor', labelsize=8)
        
        ax.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
        
        #ax.set(yscale='log')
        #ax.sharey(axs['S'])
        ax.set_xlim([0.88, 1.1])
        #ax.set_ylim([10**(-5), np.ceil(np.nanmax(df['max peak']))])
        
    mosaic = [["P_ll", "P_lh", "P_lh", "P_hl", "P_hl", "P_hh", "P_hh"],
              #[".", ".", ".", ".", ".", ".", "."],
              ["P_ir", "P_ir", ".", ".", "Temp", "Temp", "."],
              ["P_ir", "P_ir", ".", ".", "S", "S", "."]]
       
    fig, axs = plt.subplot_mosaic(mosaic, figsize = (6.8, 4.5),
                                  width_ratios = [1, 0.8, 0.2, 0.2, 0.8, 0.7, 0.3],
                                  height_ratios = [0.3, 0.25, 0.35],
                                  #layout = 'constrained',
                                  gridspec_kw = {'hspace' : 0.3, 'wspace' : 0.0})
    
    
    for stability_pivot, feasibility_pivot, ax in zip(stability_sli.values(),
                                                      feasibility_sli.values(),
                                                      ["P_lh",
                                                       "P_hh",
                                                       "P_ll",
                                                       "P_hl"]):
        
        stability_plot(stability_pivot,
                       feasibility_pivot,
                       axs[ax])
        
    stability_plot(stability_ir,
                   feasibility_ir,
                   axs["P_ir"],
                   log_x=True)
    
    sensitivities_plot(example_sensitivities_df,
                       axs["S"])
    
    example_sensitivities_plot(example_sensitivities_df,
                               example_sensitivity,
                               axs["Temp"])
    
    plt.savefig("C:/Users/jamil/Documents/PhD/Figures/externally_supplied_resources/simulations_influx.png",
                bbox_inches='tight')
    plt.savefig("C:/Users/jamil/Documents/PhD/Figures/externally_supplied_resources/simulations_influx.svg",
                bbox_inches='tight')
                 
    plt.show()
    
# %%

################ load in data #################

# influx vs self-inhibition

simulations_sli = {key : load_clean_simulations(data_location)
                   for key, data_location in zip(["high_si_low_i",
                                                  "high_si_high_i",
                                                  "low_si_low_i",
                                                  "low_si_high_i"],
                                                 ["rho_sigma_mu50_sl",
                                                  "rho_sigma_mu50_h",
                                                  "rho_sigma_noinflux",
                                                  "rho_sigma_largeinflux"])}

stability_sli = {key : stability_diagram(df,
                                         'rho',
                                         'sigma_c')
                 for key, df in simulations_sli.items()}

feasibility_sli = {key : feasibility_diagram(df,
                                             'rho',
                                             'sigma_c')
                   for key, df in simulations_sli.items()}

# influx vs rho

simulations_ir = load_clean_simulations("hybrid_influx_rho")
simulations_ir['b_exponent'] = np.round(np.log10(simulations_ir['b_val']),
                                        3)

stability_ir = stability_diagram(simulations_ir,
                                         "rho",
                                         "b_exponent")

feasibility_ir = feasibility_diagram(simulations_ir,
                                             "rho",
                                             "b_exponent")

# influx vs sensitivities to community regulation

#sensitivities_df = load_clean_simulations("rho_influx_sensitivities.csv")
#sensitivities_df['log_dRdx2'] = np.log10(np.select([sensitivities_df['dRdx2'] == 0,
#                                                    sensitivities_df['dRdx2'] > 0],
#                                                   [1e-8,
#                                                    sensitivities_df['dRdx2']],
#                                                   np.nan))

#sensitivities_pivot = pd.pivot_table(sensitivities_df,
#                                     index = 'rho',
#                                     columns = 'b',
#                                     values = 'log_dRdx2',
#                                     aggfunc = 'median')

#sensitivities_stability  = stability_diagram(sensitivities_df.mask(sensitivities_df['max. le'] < 0),
#                                             index = 'rho',
#                                             columns = 'b')

example_sensitivities = pd.read_pickle("C:/Users/jamil/Documents/PhD/Data/external_resource_stability/simulations/rho_influx_example_sensitivities_cusp_2.pkl")
example_df = pd.DataFrame([{key : sensitivities_dict[key] 
                            for key in ['b', 'rho', 'max peak', 'max. le']}
                           for sensitivities_dict in example_sensitivities])
#example_df['max peak'] += np.nanmin(np.concatenate([es['dRdx2']
#                                                    for es in example_sensitivities]))

example_df.loc[example_df['max peak'] == 0, 'max peak']  = 0.01

# %%

############## plot figure ##########

influx_figure(stability_sli,
              feasibility_sli,
              stability_ir,
              feasibility_ir,
              example_df,
              example_sensitivities[19])

# %%

for i, es in enumerate(example_sensitivities):
    
    if np.nanmax(es['dRdx2']) > 10**5.5:
        
        print(i)
        
        break

