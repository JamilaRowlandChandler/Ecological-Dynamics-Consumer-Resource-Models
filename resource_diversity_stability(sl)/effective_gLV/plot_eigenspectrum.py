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
import pandas as pd
from matplotlib import pyplot as plt

# %%

eigenspec_directory = "C:/Users/jamil/Documents/PhD/Data/resource_diversity_stability/simulations/CRM_TS/eigenspectra"
figure_directory = "C:/Users/jamil/Documents/PhD/Figures/resource_diversity_stability"

CRM_eigenspec_stats = pd.read_pickle(eigenspec_directory + "/M_vs_mu_c_eigenspec_stats.pkl")
CRM_example_trajectories = pd.read_pickle(eigenspec_directory + "/M_vs_mu_c_example_trajectories.pkl")

eLV_eigenspec_stats = pd.read_pickle(eigenspec_directory + "/M_vs_mu_c_eLV_eigenspec_stats.pkl")
eLV_example_trajectories = pd.read_pickle(eigenspec_directory + "/M_vs_mu_c_eLV_example_trajectories.pkl")

# %%

def example_eigenspectr(idx : int,
                        filename : str) -> None:

    '''

    Plot the leading eigenspectrum for each (M, epsilon) combination, for
    one of the saved communities (idx), alongside the corresponding eLV_SL
    community's eigenspectrum (the CRM's epsilon -> 0 limit) as a final
    reference panel.

    '''

    def collate_eigenspectra(idx):

        eigenspectra = [[community_data['eigenspec_stats'][0]['eigenspectrum']
                         for community_data in communities[idx].values()] + \
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

        titles = [[format_title_from_dict(title_data)
                   for title_data in titles_data_M] + [eLV_title]
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

            ax.set_xlim([-0.3, 0.12])
            ax.set_ylim([-0.11, 0.11])

        fig.supxlabel('Re(λ)', weight="bold", fontsize=10)
        fig.supylabel('Im(λ)', weight="bold", fontsize=10)

        axs[0, int(len(eigenspectra)/2)-1].set_ylim([-0.75, 0.75])
        axs[0, int(len(eigenspectra)/2)-1].set_xlim([-7, 7])
        axs[1, int(len(eigenspectra)/2)-1].set_ylim([-0.75, 0.75])
        axs[1, int(len(eigenspectra)/2)-1].set_xlim([-7, 7])

        plt.savefig(figure_directory + "/" + filename + "_smallrange.png",
                    bbox_inches='tight')
        plt.savefig(figure_directory + "/" + filename + "_smallrange.svg",
                    bbox_inches='tight')

        plt.show()

    eigenspectra = collate_eigenspectra(idx)
    titles = collate_titles(idx)

    full_spectra(eigenspectra, titles)
    zoom_spectra(eigenspectra, titles)

example_eigenspectr(1,
                    "eigenspectrum_ts_chaos")

example_eigenspectr(0,
                    "eigenspectrum_ts_stable")

# %%

def example_dynamics(idx : int) -> None:

    '''

    Plot example CRM trajectories (across epsilons) for one of the saved
    communities (idx) - only available for community indices that were
    passed to save_example_trajectories() when ts_eigenspectrum.py was run.

    '''

    resource_pool_sizes = list(CRM_example_trajectories.keys())

    fig, axs = plt.subplots(len(resource_pool_sizes), 4,
                           layout="constrained",
                           figsize=(8.5, 3.5 * len(resource_pool_sizes)))

    for row, M in enumerate(resource_pool_sizes):

        for ax, (epsilon, (t, y)) in zip(axs[row],
                                         CRM_example_trajectories[M][idx].items()):

            log_epsilon = np.abs(np.round(np.log10(float(epsilon)), 1))

            ax.plot(t, y[:int(M), :].T)
            ax.set_title(f"$10^{{{log_epsilon}}}$" if row == 0 else "",
                         weight="bold",
                         fontsize=10)

    fig.supxlabel('time', weight="bold", fontsize=10)
    fig.supylabel('abundance', weight="bold", fontsize=10)

    plt.savefig(figure_directory + "/ts_example_chaos.png",
                bbox_inches='tight')
    plt.savefig(figure_directory + "/ts_example_chaos.svg",
                bbox_inches='tight')
    plt.show()

example_dynamics(0)

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
