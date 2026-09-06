# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 00:00:00 2026

@author: jamil

Plot eLV ablation results (produced by ablate_eLV_directory() in
ablate_eLV_correlations.py) as a series of heatmaps - one panel per
ablation - following the same heatmap style as Stability_Plot() in
eLV_vs_CRM_all.py.

Note: deliberately does NOT import ablate_eLV_correlations.py (it currently
has a live, unguarded __main__ block running real ablation sweeps - see its
own module docstring/history) or eLV_vs_CRM_all.py (same "unguarded
top-level code runs on import" issue). The ablation label list and the
M/mu_c-column derivation logic are duplicated here instead, to avoid
triggering either.

"""

import numpy as np
import pandas as pd
import seaborn as sns
import os
import sys
from matplotlib import pyplot as plt
from typing import Union

os.chdir('C:/Users/jamil/Documents/PhD/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models/resource_diversity_stability(sl)/effective_gLV')

sys.path.insert(0, "C:/Users/jamil/Documents/PhD/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models" + \
                    "/resource_diversity_stability(sl)")
from complete_simulation_functions import le_pivot_r, agg_pivot

# %%

# every ablation label ablate_eLV_directory() can produce (c.f.
# _ALL_ABLATIONS in ablate_eLV_correlations.py - duplicated here rather
# than imported, see module docstring)
ALL_ABLATIONS = ["rho_R", "rho_C", "rho_D", "rho_r_Aij", "rho_Aii_Aij",
                 "mu_r", "mu_Aij", "sigma_r", "sigma_Aij", "sigma_Aii"]

# %%

def read_ablation_directory(directory : str):

    '''

    Load every csv in `directory` (one ablation's v3 output, from
    ablate_eLV_directory()) into a single dataframe, adding 'M' and scaled
    'mu_c' columns (c.f. read_eLV_data() in eLV_vs_CRM_all.py, whose csv
    branch does the same derivation).

    Parameters
    ----------
    directory : str
        Full path to one ablation's output subdirectory
        (e.g. ".../eLV_ablations/M_vs_mu_c/rho_D").

    Returns
    -------
    df : pd.DataFrame
        Concatenated interaction statistics across every csv in directory.

    '''

    dfs = []

    for file in os.listdir(directory):

        df = pd.read_csv(os.path.join(directory, file))

        df['M'] = df['no_resources']
        df['mu_c'] = np.round(df['no_resources'] * df['mu_c'], 4)

        dfs.append(df)

    return pd.concat(dfs, axis = 0, ignore_index = True)

# %%

def read_pickled_eLV_directory(directory : str):

    '''

    Load every pickled file in `directory` (full eLV_SL/eLV_ES community
    objects, e.g. as saved by eLV_M() in all_mu_c_vs_M_egLV.py - this data
    was deliberately NOT switched to v3 csv saving, unlike the gLV/ablation
    directories, so it needs its own reader) into a single dataframe with
    the same M/mu_c/'Max. lyapunov exponent' columns
    read_ablation_directory() produces, so it can be plotted alongside the
    ablations for comparison (c.f. read_eLV_data()'s pkl branch in
    eLV_vs_CRM_all.py, which this mirrors).

    species_survival_fraction is only included if calculate_community_
    properties() was called on these communities (not true of the base
    eLV_M() pipeline as of writing) - falls back to NaN otherwise, rather
    than raising.

    Parameters
    ----------
    directory : str
        Full path to a directory of pickled eLV community lists (e.g.
        ".../eLV/M_vs_mu_c").

    Returns
    -------
    df : pd.DataFrame
        Concatenated interaction statistics across every file in directory.

    '''

    dfs = []

    for file in os.listdir(directory):

        eLV_communities = pd.read_pickle(os.path.join(directory, file))

        df = pd.DataFrame({
            'M' : [c.no_resources for c in eLV_communities],
            'mu_c' : [np.round(c.no_resources * c.mu_c, 4) for c in eLV_communities],
            'Max. lyapunov exponent' : [c.max_lyapunov_exponent for c in eLV_communities],
            'mu_Aij' : [c.mu_Aij for c in eLV_communities],
            'sigma_Aij' : [c.sigma_Aij for c in eLV_communities],
            'mu_Aii' : [c.mu_Aii for c in eLV_communities],
            'sigma_Aii' : [c.sigma_Aii for c in eLV_communities],
            'mu_r' : [c.mu_r for c in eLV_communities],
            'sigma_r' : [c.sigma_r for c in eLV_communities],
            'rho_D' : [c.rho_D for c in eLV_communities],
            'rho_R' : [c.rho_R for c in eLV_communities],
            'rho_C' : [c.rho_C for c in eLV_communities],
            'rho_1idx' : [c.rho_1idx for c in eLV_communities],
            'species_survival_fraction' : [getattr(c, 'species_survival_fraction', [np.nan])[0]
                                           for c in eLV_communities],
            })

        dfs.append(df)

    return pd.concat(dfs, axis = 0, ignore_index = True)

# %%

def read_all_ablations(ablation_base_directory : str,
                       eLV_directory : Union[str, None] = None,
                       ablations : list = ALL_ABLATIONS):

    '''

    Load every ablation subdirectory of ablation_base_directory (as saved
    by ablate_eLV_directory()) into a dict of dataframes.

    Ablations without a subdirectory yet (or with an empty one - e.g. a
    sweep still in progress) are skipped, with a message, rather than
    raising.

    Parameters
    ----------
    ablation_base_directory : str
        Full path to the directory ablate_eLV_directory() was given as
        output_directory (e.g. ".../eLV_ablations/M_vs_mu_c").
    eLV_directory : str, optional
        If given, also load the un-ablated eLV data from this directory
        (pickled eLV community objects, e.g. ".../eLV/M_vs_mu_c") as an
        'original' entry, for comparison alongside the ablations. Placed
        first in the returned dict. The default is None (not included).
    ablations : list, optional
        Which ablation labels to look for. The default is ALL_ABLATIONS.

    Returns
    -------
    ablation_dfs : dict
        {label : dataframe}, one entry per ablation with data present, plus
        'original' first if eLV_directory was given.

    '''

    ablation_dfs = {}

    if eLV_directory is not None:

        ablation_dfs['original'] = read_pickled_eLV_directory(eLV_directory)

    for ablation in ablations:

        directory = os.path.join(ablation_base_directory, ablation)

        if not os.path.exists(directory) or len(os.listdir(directory)) == 0:

            print(f"skipping '{ablation}': {directory} does not exist or is empty")
            continue

        ablation_dfs[ablation] = read_ablation_directory(directory)

    return ablation_dfs

# %%

def survival_fraction_pivot(df : pd.DataFrame):

    '''

    pivot_func for plot_ablation_heatmaps(): mean species_survival_fraction
    (mu_c vs M), via agg_pivot().

    species_survival_fraction may not be available for every dataframe -
    the base eLV pipeline (read_pickled_eLV_directory(), eLV_M() in
    all_mu_c_vs_M_egLV.py) never calls calculate_community_properties(), so
    that column is all-NaN there, and agg_pivot()'s mean will silently
    produce NaN cells (left blank by seaborn) rather than raising.

    '''

    return agg_pivot(df, values = 'species_survival_fraction',
                     index = 'mu_c', columns = 'M')[0]

# %%

def plot_ablation_heatmaps(ablation_dfs : dict,
                           pivot_func = None,
                           cbar_label : str = "Probability(stability)",
                           cmap : str = "Purples_r",
                           vmin : float = 0,
                           vmax : float = 1,
                           mu_c_max : Union[float, None] = None,
                           ncols : int = 5,
                           figsize : Union[tuple, None] = None,
                           savepath : Union[str, None] = None):

    '''

    Plot one heatmap per ablation (mu_c vs M), arranged in a grid, using
    the same heatmap style as Stability_Plot() in eLV_vs_CRM_all.py.

    Parameters
    ----------
    ablation_dfs : dict
        {ablation label : dataframe}, as returned by read_all_ablations().
    pivot_func : callable, optional
        df -> pivoted 2D table (index=mu_c, columns=M). The default plots
        probability(stability) via le_pivot_r(). To plot a different
        quantity instead, pass e.g.

            lambda df: agg_pivot(df, values='species_survival_fraction',
                                 index='mu_c', columns='M')[0]

        and set cbar_label/cmap/vmin/vmax to match.
    cbar_label : str, optional
        Colorbar label. The default is "Probability(stability)".
    cmap : str, optional
        Colormap. The default is "Purples_r".
    vmin, vmax : float, optional
        Colour scale limits. The default is 0, 1.
    mu_c_max : float, optional
        If given, only plot rows with mu_c below this cutoff (c.f.
        Stability_Plot()'s df['mu_c'] < 220 filter). The default is None
        (no filtering).
    ncols : int, optional
        Number of columns in the panel grid. The default is 5.
    figsize : tuple, optional
        Figure size. The default is (2.6*ncols, 2.6*nrows).

    Returns
    -------
    fig, axs

    '''

    if pivot_func is None:

        pivot_func = lambda df: le_pivot_r(df, index = 'mu_c', columns = 'M')[0]

    labels = list(ablation_dfs.keys())
    n = len(labels)
    nrows = int(np.ceil(n / ncols))

    if figsize is None:

        figsize = (2.3 * ncols, 2.31 * nrows)

    sns.set_style('ticks')

    fig, axs = plt.subplots(nrows, ncols,
                            sharex = True, sharey = True,
                            layout = 'constrained', figsize = figsize)

    axs_flat = np.atleast_1d(axs).flatten()

    for i, (label, ax) in enumerate(zip(labels, axs_flat)):

        df = ablation_dfs[label]

        if mu_c_max is not None:

            df = df.loc[df['mu_c'] < mu_c_max, :]

        resource_pool_sizes = np.unique(df['M'])
        mu_cs = np.unique(df['mu_c'])

        pivot = pivot_func(df)

        ax.set_facecolor('grey')

        subfig = sns.heatmap(pivot, ax = ax,
                             vmin = vmin, vmax = vmax,
                             cbar = (i == 0), cmap = cmap)

        subfig.axhline(0, 0, 1, color = 'black', linewidth = 2)
        subfig.axhline(pivot.shape[0], 0, 1, color = 'black', linewidth = 2)
        subfig.axvline(0, 0, 1, color = 'black', linewidth = 2)
        subfig.axvline(pivot.shape[1], 0, 1, color = 'black', linewidth = 2)

        ax.set_xticks(np.arange(0.5, len(resource_pool_sizes) + 0.5, 2),
                     labels = resource_pool_sizes[::2], fontsize = 10, rotation = 0)
        ax.set_yticks(np.arange(0.5, len(mu_cs) + 0.5, 2),
                     labels = np.round(mu_cs[::2], 0).astype(int), fontsize = 10)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.invert_yaxis()

        title = "original (no ablation)" if label == "original" else f"no {label}"
        ax.set_title(title, fontsize = 10, weight = 'bold')

    # hide any unused panels (n not a multiple of ncols)
    for ax in axs_flat[n:]:

        ax.set_visible(False)

    cbar = axs_flat[0].collections[0].colorbar
    cbar.set_label(label = cbar_label, size = 9)
    cbar.ax.tick_params(labelsize = 8)

    fig.supxlabel('resource pool size, ' + r'$M$', fontsize = 12, weight = 'bold')
    fig.supylabel('avg. total consumption coeff., ' + r'$\mu_c$', fontsize = 12, weight = 'bold')
    
    if savepath is not None:

        plt.savefig(savepath + ".png", bbox_inches='tight')
        plt.savefig(savepath + ".svg", bbox_inches='tight')

    plt.show()

    return fig, axs

# %%

# example usage (guarded so importing this module never runs it):

ablation_dfs = read_all_ablations(
    "C:/Users/jamil/Documents/PhD/Data/resource_diversity_stability/simulations/eLV_ablations/M_vs_mu_c",
    eLV_directory = "C:/Users/jamil/Documents/PhD/Data/resource_diversity_stability/simulations/eLV/M_vs_mu_c")
 
# %%

#     # probability(stability)
fig, axs = plot_ablation_heatmaps(ablation_dfs[1:],
                                  mu_c_max=210,
                                 savepath = "C:/Users/jamil/Documents/PhD/Figures/" + \
                                     "resource_diversity_stability/M_vs_mu_c_ablations_stability")
 
# %%

#     # mean species survival fraction (NaN/blank for 'original' - not
#     # available for the base eLV pipeline, see survival_fraction_pivot())


fig, axs = plot_ablation_heatmaps(ablation_dfs,
                                  pivot_func = survival_fraction_pivot,
                                  cmap='Greens_r',
                                  vmax=np.max(np.concatenate([df.loc[:, 'species_survival_fraction'].to_numpy() 
                                                              for df in ablation_dfs.values()])),
                                  cbar_label = "Mean species survival fraction",
                                  mu_c_max=210,
                                 savepath = "C:/Users/jamil/Documents/PhD/Figures/" + \
                                     "resource_diversity_stability/M_vs_mu_c_ablations_diversity")
