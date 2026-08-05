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

    df = np.round(df, 7)

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
                          values = "Divergence measure",
                          aggfunc = prop_feasible)

# %%

def compare_abiotic_biotic(stability_ab,
                           feasibility_ab,
                           stability_b,
                           feasibility_b):

    def stability_plot(stability_pivot,
                       feasibility_pivot,
                       title,
                       ax):

        cmap_stable = LinearSegmentedColormap.from_list("cmap_feasible",
                                                        [(0.0, colorConverter.to_rgba('#30007dff', alpha = 1)),
                                                         (1.0, colorConverter.to_rgba('white', alpha=0))])

        cmap_feasible = LinearSegmentedColormap.from_list("cmap_feasible",
                                                          [(0.0, colorConverter.to_rgba('#595959ff', alpha = 1)),
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

    ##################################

    rhos = stability_ab.index.to_numpy()
    sigmas = stability_ab.columns.to_numpy()

    sns.set_style('ticks')

    mosaic = [["PA", "PB"]]

    fig, axs = plt.subplot_mosaic(mosaic, figsize = (6, 3),
                                  gridspec_kw = {'wspace' : 0.2})

    stability_plot(stability_ab,
                   feasibility_ab,
                   "Externally-supplied resources",
                   axs["PA"])

    stability_plot(stability_b,
                   feasibility_b,
                   "Self-limiting resources",
                   axs["PB"])

    axs["PA"].set_facecolor('white')
    axs["PB"].set_facecolor('white')

    plt.savefig("C:/Users/jamil/Documents/PhD/Figures/externally_supplied_resources/simulations_rho_sigma_large_mu_allplus.png",
                bbox_inches='tight')
    plt.savefig("C:/Users/jamil/Documents/PhD/Figures/externally_supplied_resources/simulations_rho_sigma_large_mu_allplus.svg",
                bbox_inches='tight')

    plt.show()

# %%

simulations_abiotic, stability_abiotic = load_clean_simulations("rho_sigma_mu50_es_allplus")
simulations_biotic, stability_biotic = load_clean_simulations("rho_sigma_mu50_sl_allplus")

feasibility_abiotic = feasible_region(simulations_abiotic)
feasibility_biotic = feasible_region(simulations_biotic)

# %%

compare_abiotic_biotic(stability_abiotic,
                       feasibility_abiotic,
                       stability_biotic,
                       feasibility_biotic)
