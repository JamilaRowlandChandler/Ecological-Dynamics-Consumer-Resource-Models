# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  3 10:49:24 2025

@author: jamil
"""

# %%

import numpy as np
import pandas as pd
import seaborn as sns
import os
import sys

from matplotlib import pyplot as plt

from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as patheffects

os.chdir('C:/Users/jamil/Documents/PhD/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models/resource_diversity_stability(sl)/cavity_solutions_vs_simulations')

sys.path.insert(0, "C:/Users/jamil/Documents/PhD/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models/" + \
                    "resource_diversity_stability(sl)")
from complete_simulation_functions import generate_simulation_df, le_pivot_r

# %%

# read_eLV_data() output columns - kept the same regardless of whether a
# directory holds pickled community objects (eLV directories, v1) or v3 csvs
# of pre-computed statistics (gLV directories, since gLV_M()/gLV_M_averaged()
# switched to v3 saving - see mean_variance_test.py/averaged_stats_gLV.py)
_READ_ELV_DATA_COLUMNS = ['M', 'mu_c', 'Max. lyapunov exponent', 'mu_Aij',
                         'sigma_Aij', 'rho_D', 'rho_R', 'rho_C', 'rho_1idx',
                         'corr_violate', 'mu_Aii', 'sigma_Aii', 'mu_r',
                         'sigma_r', 'Divergence']

def read_eLV_data(directory):

    def read_pkl(filepath):

        egLV_communities = pd.read_pickle(filepath)

        df = pd.DataFrame({'M' : np.repeat(egLV_communities[0].no_resources,
                                                         len(egLV_communities)),
                                       'mu_c' : np.repeat(np.round(egLV_communities[0].no_resources * egLV_communities[0].mu_c, 4),
                                                            len(egLV_communities)),
                 'Max. lyapunov exponent' : [gLV_community.max_lyapunov_exponent
                          for gLV_community in egLV_communities],
                 'mu_Aij' : [gLV_community.mu_Aij
                          for gLV_community in egLV_communities],
                 'sigma_Aij' : [gLV_community.sigma_Aij
                          for gLV_community in egLV_communities],
                 'rho_D' : [gLV_community.rho_D
                          for gLV_community in egLV_communities],
                 'rho_R' : [gLV_community.rho_R
                          for gLV_community in egLV_communities],
                 'rho_C' : [gLV_community.rho_C
                          for gLV_community in egLV_communities],
                 'rho_1idx' : [gLV_community.rho_1idx
                          for gLV_community in egLV_communities],
                 'corr_violate' : [gLV_community.corr_violate
                                   if hasattr(gLV_community, "corr_violate")
                                   else False
                              for gLV_community in egLV_communities],
                 'mu_Aii' : [gLV_community.mu_Aii
                          for gLV_community in egLV_communities],
                 'sigma_Aii' : [gLV_community.sigma_Aii
                          for gLV_community in egLV_communities],
                 'mu_r' : [gLV_community.mu_r
                           if hasattr(gLV_community, "mu_r")
                           else gLV_community.r
                           for gLV_community in egLV_communities],
                 'sigma_r' : [gLV_community.sigma_r
                              if hasattr(gLV_community, "sigma_r")
                              else 0
                              for gLV_community in egLV_communities],
                 'Divergence' : [gLV_community.ODE_sols[0].t[-1]
                              for gLV_community in egLV_communities]
                 })

        return df[_READ_ELV_DATA_COLUMNS]

    def read_csv(filepath):

        df = pd.read_csv(filepath)

        # 'M'/'mu_c' aren't stored directly by eLV_gLV_df_from_communities()
        # (which keeps the per-resource no_resources/mu_c instead) - derive
        # them the same way the pickled-object path above does
        df['M'] = df['no_resources']
        df['mu_c'] = np.round(df['no_resources'] * df['mu_c'], 4)

        return df[_READ_ELV_DATA_COLUMNS]

    egLV_df = pd.concat([read_csv(os.path.join(directory, file))
                         if file.endswith('.csv')
                         else read_pkl(os.path.join(directory, file))
                         for file in os.listdir(directory)],
                        axis = 0, ignore_index = True)

    return egLV_df

# %%

def _is_eLV_directory(directory):

    '''

    Inspect one file in directory to decide whether it holds eLV/gLV data
    (read_eLV_data()) or raw CRM data (generate_simulation_df()) - v3 csvs
    are always eLV/gLV (eLV_gLV_df_from_communities() output), and pickled
    community lists are eLV/gLV only if their communities carry interaction
    statistics (mu_Aij), which raw CRM community objects never have.

    '''

    example_file = next(file for file in os.listdir(directory)
                        if file.endswith('.csv') or file.endswith('.pkl'))

    if example_file.endswith('.csv'):

        return True

    example_communities = pd.read_pickle(os.path.join(directory, example_file))

    return hasattr(example_communities[0], 'mu_Aij')

def load_stability_dataset(directory):

    '''

    Load a directory of simulation output into a common-schema dataframe
    (at minimum: 'M', 'mu_c', 'Max. lyapunov exponent'), automatically
    picking read_eLV_data() (eLV/gLV: pickled community lists or v3 csvs)
    or generate_simulation_df() (raw CRM pickles) depending on what
    directory actually holds.

    Parameters
    ----------
    directory : str
        Full filepath of the directory to load.

    Returns
    -------
    pd.DataFrame

    '''

    if _is_eLV_directory(directory):

        return read_eLV_data(directory)

    return generate_simulation_df(directory)

# %%

def Stability_Plot(dfs,
                   titles,
                   mu_c_max = 220,
                   savepath = None):

    '''

    Plot a row of stability-probability heatmaps (M vs mu_c), one per
    dataset, for any number of datasets.

    Parameters
    ----------
    dfs : list of pd.DataFrame
        Each dataframe needs 'M', 'mu_c' and 'Max. lyapunov exponent'
        columns (see load_stability_dataset()).
    titles : list of str
        Subplot title for each dataset in dfs (same length as dfs).
    mu_c_max : float, optional
        Communities with mu_c >= mu_c_max are excluded. The default is 220.
    savepath : str, optional
        If given, save the figure to savepath + '.png'/'.svg'. The default
        is None (figure is not saved to file).

    '''

    if len(dfs) != len(titles):

        raise ValueError("dfs and titles must be the same length.")

    n = len(dfs)

    resource_pool_sizes = np.unique(dfs[0]['M'])
    mu_cs = np.unique(dfs[0].loc[dfs[0]['mu_c'] < mu_c_max, 'mu_c'])

    ######################## Phase diagram ######################################

    stability_pivots = [le_pivot_r(df.loc[df['mu_c'] < mu_c_max, :],
                                  columns = 'M',
                                  index = 'mu_c')[0]
                        for df in dfs]

    sns.set_style('ticks')

    fig, axs = plt.subplots(1, n,
                            sharex=True, sharey=True,
                            layout='constrained',
                            figsize=(2.3 * n, 2.6))

    if n == 1:

        axs = [axs]

    for i, (stability_pivot, title, ax) in enumerate(zip(stability_pivots,
                                                         titles,
                                                         axs)):

        subfig = sns.heatmap(stability_pivot,
                             ax = ax,
                             vmin = 0,
                             vmax = 1,
                             cbar = (i == n - 1),
                             cmap = 'Purples_r')

        subfig.axhline(0, 0, 1, color = 'black', linewidth = 2)
        subfig.axhline(stability_pivot.shape[0], 0, 1,
                       color = 'black', linewidth = 2)
        subfig.axvline(0, 0, 1, color = 'black', linewidth = 2)
        subfig.axvline(stability_pivot.shape[1], 0, 1,
                       color = 'black', linewidth = 2)

        ax.set_xticks(np.arange(0.5, len(resource_pool_sizes) + 0.5, 2),
                            labels = resource_pool_sizes[::2], fontsize = 10,
                            rotation = 0)

        ax.set_yticks(np.arange(0.5, len(mu_cs) + 0.5, 2), labels = np.int32(mu_cs[::2]),
                            fontsize = 10, rotation = 0)

        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.invert_yaxis()

        ax.set_title(title,
                     fontsize=10,
                     weight="bold",
                     horizontalalignment = 'center',
                     verticalalignment = 'top')

    cbar = axs[-1].collections[0].colorbar
    cbar.set_label(label = 'Probability(stability)',
                   size = '8', horizontalalignment = 'center',
                   verticalalignment = 'top')
    cbar.ax.tick_params(labelsize = 10)

    fig.supxlabel('resource pool size, ' + r'$M$', fontsize = 10,
                  weight = 'bold')
    fig.supylabel('avg. total consumption coeff., ' + r'$\mu_c$',
                  fontsize = 10, weight = 'bold')

    if savepath is not None:

        plt.savefig(savepath + ".png", bbox_inches='tight')
        plt.savefig(savepath + ".svg", bbox_inches='tight')

    plt.show()

# %%

def _lyapunov_exponent(community):

    '''

    eLV/gLV communities store this as max_lyapunov_exponent, raw CRM
    communities as lyapunov_exponent.

    '''

    return getattr(community, 'max_lyapunov_exponent',
                   getattr(community, 'lyapunov_exponent', None))

def _first_chaotic(communities):

    for community in communities:

        if _lyapunov_exponent(community) > 0:

            return community

    return communities[0]

def _first_stable(communities):

    for community in communities:

        if _lyapunov_exponent(community) <= 0:

            return community

    return communities[0]

def _has_resource_dynamics(community):

    '''

    True if community.ODE_sols track resource abundances as well as
    species abundances (raw CRM communities) rather than species alone
    (eLV/gLV communities).

    '''

    return community.ODE_sols[0].y.shape[0] > community.no_species

def _indices_and_cmap(M):

    colour_index = np.arange(M)
    np.random.shuffle(colour_index)

    cmap = LinearSegmentedColormap.from_list('custom YlGBl',
                                             ['#e9a100ff','#1fb200ff',
                                              '#1f5a00ff','#00e9e9ff','#001256fd'],
                                               N = M)

    return colour_index, cmap

def _plot_dynamics(ax, simulation, colour_index_cmap, title):

    data = simulation.ODE_sols[0]

    colour_index, cmap = colour_index_cmap
    var_pos = np.arange(len(colour_index))

    if title == "resource":

        var_pos += len(colour_index)

    for i, v in zip(colour_index, var_pos):

        ax.plot(data.t, data.y[v,:].T, color = 'black', linewidth = 0.5)
        ax.plot(data.t, data.y[v,:].T, color = cmap(i), linewidth = 0.45)

        ax.set_title(title, fontsize = 9, y = 0.85)

    return ax

def population_dynamics(datasets,
                        chaotic_file = "simulations_75_1.9333.pkl",
                        stable_file = "simulations_225_0.6444.pkl",
                        example_M = (75, 225),
                        savepath = None):

    '''

    Plot example chaotic/stable population dynamics for any number of
    datasets, side by side. Datasets that track resource abundances (raw
    CRM communities) additionally get a resource-dynamics row; datasets
    that don't (eLV/gLV communities) don't.

    Parameters
    ----------
    datasets : list of (str, str) tuples
        (title, directory) pairs. Each directory must contain
        chaotic_file/stable_file - pickled lists of communities (one file =
        one (M, mu_c) combination, with ODE_sols already simulated).
    chaotic_file : str, optional
        Filename (within each dataset's directory) to search for a
        community with a chaotic (positive) Lyapunov exponent. The default
        is "simulations_75_1.9333.pkl".
    stable_file : str, optional
        Filename (within each dataset's directory) to search for a
        community with a stable (non-positive) Lyapunov exponent. The
        default is "simulations_225_0.6444.pkl".
    example_M : (int, int), optional
        Resource pool sizes corresponding to chaotic_file/stable_file
        respectively (used to build consistent colour maps across
        datasets). The default is (75, 225).
    savepath : str, optional
        If given, save the figure to savepath + '.png'/'.svg'. The default
        is None (figure is not saved to file).

    '''

    n = len(datasets)

    loaded = []

    for title, directory in datasets:

        chaotic_communities = pd.read_pickle(os.path.join(directory, chaotic_file))
        stable_communities = pd.read_pickle(os.path.join(directory, stable_file))

        chaotic_community = _first_chaotic(chaotic_communities)
        stable_community = _first_stable(stable_communities)

        loaded.append((title, chaotic_community, stable_community,
                      _has_resource_dynamics(chaotic_community)))

    colour_index_cmap_by_M = {M : _indices_and_cmap(M) for M in example_M}

    col_labels = []
    resource_row = []

    for i, (_, _, _, has_resources) in enumerate(loaded):

        col_labels += [f"d{i}_chaoticC", f"d{i}_stableC"]

        if has_resources:

            resource_row += [f"d{i}_chaoticR", f"d{i}_stableR"]

        else:

            resource_row += ['.', '.']

    mosaic = [col_labels, resource_row]

    fig, axs = plt.subplot_mosaic(mosaic,
                                  layout='constrained',
                                  sharex=True,
                                  sharey=True,
                                  figsize=(2.3 * n, 2.4))

    for i, (title, chaotic_community, stable_community, has_resources) in enumerate(loaded):

        _plot_dynamics(axs[f"d{i}_chaoticC"], chaotic_community,
                      colour_index_cmap_by_M[example_M[0]], 'consumer')
        sns.despine(ax = axs[f"d{i}_chaoticC"])

        _plot_dynamics(axs[f"d{i}_stableC"], stable_community,
                      colour_index_cmap_by_M[example_M[1]], 'consumer')
        sns.despine(ax = axs[f"d{i}_stableC"])

        if has_resources:

            _plot_dynamics(axs[f"d{i}_chaoticR"], chaotic_community,
                          colour_index_cmap_by_M[example_M[0]], 'resource')
            sns.despine(ax = axs[f"d{i}_chaoticR"])

            _plot_dynamics(axs[f"d{i}_stableR"], stable_community,
                          colour_index_cmap_by_M[example_M[1]], 'resource')
            sns.despine(ax = axs[f"d{i}_stableR"])

    if savepath is not None:

        plt.savefig(savepath + ".png", bbox_inches='tight')
        plt.savefig(savepath + ".svg", bbox_inches='tight')

    plt.show()

# %%

# example usage (guarded so importing this module never runs it):

#if __name__ == "__main__":

base = "C:/Users/jamil/Documents/PhD/Data/resource_diversity_stability/simulations/"

df_CRM = load_stability_dataset(base + "M_vs_mu_c")
df_eLV_phiR = load_stability_dataset(base + "eLV/M_vs_mu_c")
df_eLV_ar = load_stability_dataset(base + "eLV/M_vs_mu_c(all_resource)")

Stability_Plot([df_CRM, df_eLV_ar, df_eLV_phiR],
               ["Consumer-resource model",
                "Consumer-only model",
                "Consumer-only model\n(inc. extinct resources)"],
               savepath = "C:/Users/jamil/Documents/PhD/Figures/" + \
                   "resource_diversity_stability/M_vs_mu_c_eLVs")
    
# %%

population_dynamics([("consumer", base + "M_vs_mu_c"),
                     ("consumer", base + "eLV/M_vs_mu_c(all_resource)"),
                     ("consumer", base + "eLV/M_vs_mu_c"),
                     ("consumer", base + "M_vs_mu_c")])
