# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 16:10:44 2026

@author: jamil

Loads and plots the eigenspec_stats/example trajectory data saved by
ts_eigenspectrum.py and ts_eigenspectrum_eLV.py (save_eigenspec_stats() and
save_example_trajectories()) - it no longer depends on those scripts having
been run in the same interactive session, and no longer imports any
simulation code (Consumer_Resource_Model, eigenspectrum(), etc.), since
everything it needs is already computed and saved to disk.
"""

import numpy as np
import numpy.typing as npt
import pandas as pd
from typing import Literal, Union
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# %%

eigenspec_directory = "C:/Users/jamil/Documents/PhD/Data/resource_diversity_stability/simulations/CRM_TS/eigenspectra"
figure_directory = "C:/Users/jamil/Documents/PhD/Figures/resource_diversity_stability"

CRM_eigenspec_stats = pd.read_pickle(eigenspec_directory + "/M_vs_mu_c_eigenspec_stats.pkl")
GC_eigenspec_stats = pd.read_pickle(eigenspec_directory + "/M_vs_mu_c_eigenspec_stats_GC.pkl")
CRM_example_trajectories = pd.read_pickle(eigenspec_directory + "/M_vs_mu_c_example_trajectories.pkl")

eLV_eigenspec_stats = pd.read_pickle(eigenspec_directory + "/M_vs_mu_c_eLV_eigenspec_stats.pkl")
eLV_GC_eigenspec_stats = pd.read_pickle(eigenspec_directory + "/M_vs_mu_c_eLV_eigenspec_stats_GC.pkl")
eLV_example_trajectories = pd.read_pickle(eigenspec_directory + "/M_vs_mu_c_eLV_example_trajectories.pkl")

# %%

def example_eigenspectra(idx : int,
                         filename : str) -> None:

    '''

    Plot the leading eigenspectrum for each (M, epsilon) combination, for
    one of the saved communities (idx), alongside the corresponding eLV_SL
    community's eigenspectrum (the CRM's epsilon -> 0 limit) as a final
    reference panel.

    '''

    def collate_eigenspectra(idx):

        eigenspectra = [list(reversed([community_data['eigenspec_stats'][0]['eigenspectrum']
                                       for community_data in communities[idx].values()])) + \
                        [eLV_eigenspec_stats[M][idx]['eigenspec_stats'][0]['eigenspectrum']]
                        for M, communities in CRM_eigenspec_stats.items()]

        return [eig_spec
                for eig_spec_M in eigenspectra
                for eig_spec in eig_spec_M]

    def collate_titles(idx):

        def format_title_from_dict(title_data):

            M, e, max_le = list(title_data.values())
            stability = "\n(stable)" if max_le < 0 else "\n(unstable)"

            title = r'$M = $' + f'${{{M}}}$, ' + \
                         r'$1/\epsilon = $' + f'$10^{{{e}}}$, ' + \
                         stability

            return title

        titles_data = [[{'M' : float(str1),
                       'e' : np.abs(np.log10(float(str2))),
                       'max. le' : np.round(val2['lyapunov_exponent'], 5)
                       }
                      for str2, val2 in val1[idx].items()]
                      for str1, val1 in CRM_eigenspec_stats.items()]

        eLV_titles = [r'$M = $' + f'${{{float(M)}}}$, ' + '\neLV_SL' + \
                     ("\n(stable)"
                      if eLV_eigenspec_stats[M][idx]['lyapunov_exponent'] < 0
                      else "\n(unstable)")
                     for M in CRM_eigenspec_stats.keys()]

        titles = [list(reversed([format_title_from_dict(title_data)
                                 for title_data in titles_data_M])) + [eLV_title]
                  for titles_data_M, eLV_title in zip(titles_data, eLV_titles)]

        return [title
                for titles_M in titles
                for title in titles_M]

    def full_spectra(eigenspectra,
                     titles):

        fig, axs = plt.subplots(2, int(len(eigenspectra)/2),
                               layout="constrained",
                               figsize=(8.5, 4))

        for ax, data, title in zip(axs.flatten(),
                                   eigenspectra,
                                   titles):

            ax.scatter(data.real,
                       data.imag,
                       c='black',
                       s=2)

            ax.axhline(0, color='grey', linewidth=0.5)
            ax.axvline(0, color='grey', linewidth=0.5)

            ax.set_xlabel('')
            ax.set_ylabel('')

            ax.set_title(title,
                         fontsize=10)

        fig.supxlabel('Re(λ)', weight="bold", fontsize=10)
        fig.supylabel('Im(λ)', weight="bold", fontsize=10)

        plt.savefig(figure_directory + "/" + filename + ".png",
                    bbox_inches='tight')
        plt.savefig(figure_directory + "/" + filename + ".svg",
                    bbox_inches='tight')

        plt.show()

    def zoom_spectra(eigenspectra,
                     titles):

        fig, axs = plt.subplots(2, int(len(eigenspectra)/2),
                               layout="constrained",
                               figsize=(1.7*int(len(eigenspectra)/2), 4))

        for ax, data, title in zip(axs.flatten(),
                                   eigenspectra,
                                   titles):

            ax.scatter(data.real,
                       data.imag,
                       c='black',
                       s=2)

            ax.axhline(0, color='grey', linewidth=0.5)
            ax.axvline(0, color='grey', linewidth=0.5)

            ax.set_xlabel('')
            ax.set_ylabel('')

            ax.set_title(title,
                         fontsize=10)

            ax.set_xlim([-0.3, 0.13])
            ax.set_ylim([-0.11, 0.11])

        fig.supxlabel('Re(λ)', weight="bold", fontsize=10)
        fig.supylabel('Im(λ)', weight="bold", fontsize=10)

        plt.savefig(figure_directory + "/" + filename + "_smallrange.png",
                    bbox_inches='tight')
        plt.savefig(figure_directory + "/" + filename + "_smallrange.svg",
                    bbox_inches='tight')

        plt.show()

    eigenspectra = collate_eigenspectra(idx)
    titles = collate_titles(idx)

    full_spectra(eigenspectra, titles)
    zoom_spectra(eigenspectra, titles)

example_eigenspectra(8,
                     "eigenspectrum_ts_chaos")

example_eigenspectra(7,
                     "eigenspectrum_ts_stable")

# %%

def example_eigenspectra_GC(idx : int,
                            filename : str) -> None:

    '''

    Plot the leading eigenspectrum for each (M, epsilon) combination, for
    one of the saved communities (idx), alongside the corresponding eLV_SL
    community's eigenspectrum (the CRM's epsilon -> 0 limit) as a final
    reference panel.

    '''

    def collate_eigenspectra(idx):

        eigenspectra = [[GC_eigenspec_stats[M][idx]['eigenspectrum'],
                        eLV_GC_eigenspec_stats[M][idx]['eigenspectrum']]
                        for M in GC_eigenspec_stats.keys()]

        return [eig_spec
                for eig_spec_M in eigenspectra
                for eig_spec in eig_spec_M]

    def collate_titles(idx):

        def format_title_from_dict(title_data):

            model, M, max_le = list(title_data.values())
            stability = "\n(stable)" if max_le < 0 else "\n(unstable)"

            title = model + "," + \
                    r'$M = $' + f'${{{M}}}$, ' + \
                    stability

            return title

        titles_data = [[{'model' : r'$-GC^T$',
                         'M' : float(M),
                       'max. le' : np.round(CRM_eigenspec_stats[M][idx]["1.0"]['lyapunov_exponent'], 5)
                        },
                        {'model' : r'$-A^T$',
                         'M' : float(M),
                         'max. le' : np.round(eLV_eigenspec_stats[M][idx]['lyapunov_exponent'], 5)
                         }]
                      for M in CRM_eigenspec_stats.keys()]
        
        titles_data_flat = [title_data 
                            for titles_data_M in titles_data
                            for title_data in titles_data_M]
        
        titles = [format_title_from_dict(title_data)
                  for title_data in titles_data_flat]

        return titles

    def full_spectra(eigenspectra,
                     titles):

        fig, axs = plt.subplots(2, 2,
                               layout="constrained",
                               figsize=(3.4, 4))

        for ax, data, title in zip(axs.flatten(),
                                   eigenspectra,
                                   titles):

            ax.scatter(data.real,
                       data.imag,
                       c='black',
                       s=2)

            ax.axhline(0, color='grey', linewidth=0.5)
            ax.axvline(0, color='grey', linewidth=0.5)

            ax.set_xlabel('')
            ax.set_ylabel('')

            ax.set_title(title,
                         fontsize=10)

        fig.supxlabel('Re(λ)', weight="bold", fontsize=10)
        fig.supylabel('Im(λ)', weight="bold", fontsize=10)

        plt.savefig(figure_directory + "/" + filename + ".png",
                    bbox_inches='tight')
        plt.savefig(figure_directory + "/" + filename + ".svg",
                    bbox_inches='tight')

        plt.show()

    def zoom_spectra(eigenspectra,
                     titles):

        fig, axs = plt.subplots(2, 2,
                               layout="constrained",
                               figsize=(3.4, 4))

        for ax, data, title in zip(axs.flatten(),
                                   eigenspectra,
                                   titles):

            ax.scatter(data.real,
                       data.imag,
                       c='black',
                       s=2)

            ax.axhline(0, color='grey', linewidth=0.5)
            ax.axvline(0, color='grey', linewidth=0.5)

            ax.set_xlabel('')
            ax.set_ylabel('')

            ax.set_title(title,
                         fontsize=10)

            ax.set_xlim([-6, 0.7])
            ax.set_ylim([-1.1, 1.1])

        fig.supxlabel('Re(λ)', weight="bold", fontsize=10)
        fig.supylabel('Im(λ)', weight="bold", fontsize=10)
        
        plt.savefig(figure_directory + "/" + filename + "_smallrange.png",
                    bbox_inches='tight')
        plt.savefig(figure_directory + "/" + filename + "_smallrange.svg",
                    bbox_inches='tight')

        plt.show()

    eigenspectra = collate_eigenspectra(idx)
    titles = collate_titles(idx)
    
    full_spectra(eigenspectra, titles)
    zoom_spectra(eigenspectra, titles)

example_eigenspectra_GC(8,
                        "eigenspectrum_GC_chaos")

example_eigenspectra_GC(7,
                        "eigenspectrum_GC_stable")


# %%

def example_dynamics(idx : Union[list[int], npt.NDArray],
                     filename_suffix : str = "") -> None:

    '''

    Plot example trajectories for one of the saved communities (idx) - only
    available for community indices that were passed to
    save_example_trajectories() when ts_eigenspectrum.py (model = 'CRM') or
    ts_eigenspectrum_eLV.py (model = 'eLV') was run.

    For the CRM, resource dynamics are plotted above consumer dynamics (two
    stacked rows per M, one column per epsilon), since the CRM's state
    includes both. eLV_SL has no separate resource dynamics to split out, so
    each M's full dynamics are instead plotted together in a single panel.

    '''
    
    def sort_rows_by_mean_desc(arr):
        
        row_means = arr.mean(axis=1)
        order = np.argsort(row_means)[::-1]
        
        ordered_arr = arr[order, :]
        
        #return order, ordered_arr
        return np.arange(arr.shape[0]), arr
        
    def colour_map(length):
        
        colour_index = np.arange(length)
        np.random.shuffle(colour_index)

        cmap = LinearSegmentedColormap.from_list('custom YlGBl',
                                                 ['#e9a100ff','#1fb200ff',
                                                  '#1f5a00ff','#00e9e9ff','#001256fd'],
                                                   N = length)
        
        shuffled_colours = cmap(np.arange(length))[colour_index]
    
        shuffled_cmap = LinearSegmentedColormap.from_list('custom_YlGBl_shuffled',
                                                          shuffled_colours,
                                                          N = length)

        return shuffled_cmap
    
    def ordered_arr_cmap(arr, cmap):
        
        order, ordered_arr = sort_rows_by_mean_desc(arr)
        
        ordered_colours = cmap(np.arange(arr.shape[0]))[order]
        #ordered_cmap = LinearSegmentedColormap.from_list('custom_YlGBl_ordered',
        #                                                 ordered_colours,
        #                                                 N = len(order))
        return ordered_arr, ordered_colours

    resource_pool_sizes = list(CRM_example_trajectories.keys())

    fig, axs = plt.subplots(2 * len(resource_pool_sizes),
                            5,
                           layout="constrained",
                           #figsize=(1.7*(len(CRM_example_trajectories["50"][idx[0]]) + 1), 6))
                           figsize=(4.2*1.7*(len(CRM_example_trajectories["50"][idx[0]]) + 1), 24))

    for row_pair, (M, idx_M) in enumerate(zip(resource_pool_sizes,
                                              idx)):
        
        cmap_r = colour_map(int(M))
        cmap_s = colour_map(int(M))

        resource_axs = axs[2 * row_pair]
        species_axs = axs[2 * row_pair + 1]

        for ax_r, ax_s, (epsilon, (t, y)) in zip(resource_axs,
                                                 species_axs,
                                                 reversed(CRM_example_trajectories[M][idx_M].items())):
            
            sort_resources, ordered_cmap_r = ordered_arr_cmap(y[int(M):, 10:], 
                                                              cmap_r)
            sort_species, ordered_cmap_s = ordered_arr_cmap(y[:int(M), 10:],
                                                            cmap_s)

            log_epsilon = np.abs(np.round(np.log10(float(epsilon)), 1))

            #ax_r.plot(t, y[int(M):, :].T)
            #ax_s.plot(t, y[:int(M), :].T)
            ax_r.stackplot(t[10:], sort_resources,
                           colors=ordered_cmap_r)
            ax_s.stackplot(t[10:], sort_species,
                           colors=ordered_cmap_s)
            ax_r.tick_params(axis="both", which="major", labelsize=7)
            ax_r.tick_params(axis="both", which="minor", labelsize=7)
            ax_s.tick_params(axis="both", which="major", labelsize=7)
            ax_s.tick_params(axis="both", which="minor", labelsize=7)

            if row_pair == 0:

                ax_r.set_title(f"$10^{{{log_epsilon}}}$",
                             weight="bold",
                             fontsize=10)
            
        resource_axs[0].set_ylabel('resources', fontsize=7)
        species_axs[0].set_ylabel('consumers', fontsize=7)
        
        for ax, M, idx_M in zip(axs[np.arange(1,
                                              len(resource_pool_sizes) + 3,
                                              2),
                                    -1], resource_pool_sizes, idx):

            t, y = eLV_example_trajectories[M][idx_M]
            
            sort_y, ordered_cmap = ordered_arr_cmap(y[:, 10:],
                                                    cmap_s)

            #ax.plot(t, y.T)
            ax.stackplot(t[10:], sort_y,
                         colors=ordered_cmap)
            ax.set_title("CM",
                         weight="bold",
                         fontsize=10)
            ax.tick_params(axis="both", which="major", labelsize=7)
            ax.tick_params(axis="both", which="minor", labelsize=7)

    fig.supxlabel('time', weight="bold", fontsize=7)
    fig.supylabel('abundance', weight="bold", fontsize=7)

    filename = "ts_example" + filename_suffix
    plt.savefig(figure_directory + "/" + filename + ".png",
                bbox_inches='tight')
    plt.savefig(figure_directory + "/" + filename + ".svg",
                bbox_inches='tight')
    plt.show()

example_dynamics([8, 8],
                 filename_suffix="_chaos")

example_dynamics([7, 7],
                 filename_suffix="_stable")

# %%

def absolute_eigenvec_contribution(CRM_eigenspec_stats : dict) -> pd.DataFrame:

    '''

    Flatten the loaded CRM eigenspec_stats into a per-(M, epsilon, community)
    DataFrame of the leading eigenvector's species/resource contributions.

    '''

    def contribution(eigenvector, M):

        species_contribution = np.mean(np.abs(eigenvector[:M]))
        resource_contribution = np.mean(np.abs(eigenvector[M:]))

        return species_contribution, resource_contribution

    rows = []

    for M_str, communities in CRM_eigenspec_stats.items():

        M = int(M_str)

        for community_ts in communities:

            for epsilon, community_data in community_ts.items():

                leading_eigenvector = community_data['eigenspec_stats'][0]['leading_vec']

                species_contribution, resource_contribution = \
                    contribution(leading_eigenvector, M)

                rows.append(dict(M = M,
                                 timescalar = float(epsilon),
                                 max_le = community_data['lyapunov_exponent'],
                                 species_contribution = species_contribution,
                                 resource_contribution = resource_contribution))

    return pd.DataFrame(rows)

eigenvec_contr_df = absolute_eigenvec_contribution(CRM_eigenspec_stats)

compare_contributions = (eigenvec_contr_df[['M',
                                            'timescalar',
                                            'species_contribution',
                                            'resource_contribution']]
                         .groupby(['M', 'timescalar'])
                         .apply('mean',
                                include_groups=False)
                         .reset_index(names=['M', 'timescalar'],
                                      drop=False))

def transform_timescalar(x):

    log_x = np.abs(np.log10(x))
    log_x_str = f"$10^{{{log_x}}}$"

    return log_x_str

def transform_contribution(x):

    return f"${x:.5f}$"

compare_contributions['timescalar'] = compare_contributions['timescalar'].apply(lambda x : transform_timescalar(x))
compare_contributions['species_contribution'] = compare_contributions['species_contribution'].apply(lambda x : transform_contribution(x))
compare_contributions['resource_contribution'] = compare_contributions['resource_contribution'].apply(lambda x : transform_contribution(x))


fig, ax = plt.subplots(1, 1)

fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')

table = ax.table(cellText=compare_contributions.values,
                 colLabels=['resource pool size, ' + r'$M$',
                            'timescale separation, ' + r'$1/\epsilon$',
                            'normalised consumer contribution to\nleading eigenvector',
                            'normalised resource contribution to\nleading eigenvector'],
                 )

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(2.5, 2.5)

plt.savefig(figure_directory + "/ts_abs_eigenvec_contr.png",
            bbox_inches='tight')
plt.savefig(figure_directory + "/ts_abs_eigenvec_contr.svg",
            bbox_inches='tight')
plt.show()

# %%

def eLV_vec_magnitude_table(eLV_eigenspec_stats : dict) -> pd.DataFrame:

    '''

    Flatten the loaded eLV eigenspec_stats into a per-(M, community) table of
    the leading eigenvector's mean absolute magnitude, for comparison against
    the CRM's species/resource contributions above.

    '''

    rows = [dict(M = int(M_str),
                community = idx,
                max_le = community_data['lyapunov_exponent'],
                vec_magnitude = np.mean(np.abs(community_data['eigenspec_stats'][0]['leading_vec'])))
           for M_str, communities in eLV_eigenspec_stats.items()
           for idx, community_data in enumerate(communities)]

    return pd.DataFrame(rows)

eLV_vec_magnitude_df = eLV_vec_magnitude_table(eLV_eigenspec_stats)

fig, ax = plt.subplots(1, 1)

fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')

eLV_table_data = eLV_vec_magnitude_df.copy()
eLV_table_data['max_le'] = eLV_table_data['max_le'].apply(lambda x : f"${x:.5f}$")
eLV_table_data['vec_magnitude'] = eLV_table_data['vec_magnitude'].apply(lambda x : f"${x:.5f}$")

table = ax.table(cellText=eLV_table_data.values,
                 colLabels=['resource pool size, ' + r'$M$',
                            'community index',
                            'max. lyapunov exponent',
                            'normalised leading eigenvector\nmagnitude'],
                 )

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(2.5, 2.5)

plt.savefig(figure_directory + "/ts_eLV_eigenvec_magnitude.png",
            bbox_inches='tight')
plt.savefig(figure_directory + "/ts_eLV_eigenvec_magnitude.svg",
            bbox_inches='tight')
plt.show()
