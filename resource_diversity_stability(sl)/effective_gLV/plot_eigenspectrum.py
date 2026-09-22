# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 16:10:44 2026

@author: jamil
"""

import numpy as np
import pandas as pd
import seaborn as sns
import os
import sys
from typing import Literal, Union
import numpy.typing as npt
from copy import deepcopy
from tqdm import tqdm
from scipy.linalg import schur
from matplotlib import pyplot as plt

os.chdir('C:/Users/jamil/Documents/PhD/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models/resource_diversity_stability(sl)/effective_gLV')

sys.path.insert(0, "C:/Users/jamil/Documents/PhD/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models" + \
                    "/consumer_resource_modules")
from models import Consumer_Resource_Model
from community_level_properties import max_le, eigenspectrum

# %%

def extract_parameters(base_community : Literal["SL_CRM"]) -> dict:
    
    parameters = {attr : getattr(base_community, 
                                 attr, 
                                 None)
                  for attr in ['no_resources', 'mu_c', 'sigma_c',
                               'mu_g', 'sigma_g',
                               'consumption',
                               'growth',
                               'b', 'd']}
    
    return parameters

# %%

def separate_timescales(parameters : dict,
                        epsilon : float) -> dict:
    
    separated_parameters = deepcopy(parameters)
    
    separated_parameters['epsilon'] = epsilon
    
    return separated_parameters

# %%

def separate_parameter_timescales(base_community : Literal["SL_CRM"],
                                  epsilons : Union[list[float], npt.NDArray]):
    
    base_parameters = extract_parameters(base_community)
    
    parameters_sep_by_eps = [separate_timescales(base_parameters,
                                                 epsilon)
                             for epsilon in epsilons]
    
    return parameters_sep_by_eps

# %%

def resimulate_CRM_timescale(parameters : dict) -> None:
    
    M = parameters['no_resources']
    
    mu_c = parameters['mu_c']
    sigma_c = parameters['sigma_c']
    mu_y = parameters['mu_g']
    sigma_y = parameters['sigma_g']
    consumption = parameters['consumption']
    growth = parameters['growth']
    
    intrinsic_resource_growth = parameters['b']
    death = parameters['d']
    
    epsilon = parameters['epsilon']
    
    community = Consumer_Resource_Model("Self-limiting resource supply",
                                        M,
                                        M)
    
    community.growth_consumption_rates('user-supplied',
                                       mu_c = mu_c,
                                       sigma_c = sigma_c,
                                       mu_g = mu_y,
                                       sigma_g = sigma_y,
                                       consumption = consumption,
                                       growth = growth)
    community.model_specific_rates(death_method = "user-supplied",
                                   death_args = {'d' : death},
                                   resource_growth_method = "user-supplied",
                                   resource_growth_args = {'b' : intrinsic_resource_growth})
    
    community.timescale_separation(epsilon)
        
    # run simulations from randomly generated initial abundances
    community.simulate_community(t_end = 7000,
                                 no_init_cond = 1)
    
    community.calculate_community_properties()
       
    # numerically estimate the max. lyapunov exponent
    community.lyapunov_exponent = max_le(community,
                                         community.ODE_sols[0].y[:, -1],
                                         T = 1000,
                                         perturbation = 1e-6)
    
    community.eigenspec_stats = [eigenspectrum(community,
                                               ode_sol.y[:, -1])
                                 for ode_sol in community.ODE_sols]
    
    return community
    
# %%

def timescale_separation_eigenspec(CRM_directory : str,
                                   epsilons,
                                   resource_pool_sizes,
                                   mu_c):


    def read_call_timescale_separate(full_CRM_directory : str,
                                     epsilons):
        
        # read in consumer-resource model (CRM) communities
        CRM_communities = pd.read_pickle(full_CRM_directory)
        
        parameters_sep_by_eps = [separate_parameter_timescales(CRM_community,
                                                               epsilons)
                                 for CRM_community in CRM_communities[7:9]]
        
        CRM_ts_eigenspec = [{str(parameters["epsilon"]) : resimulate_CRM_timescale(parameters)
                             for parameters in parameter_sets}
                            for parameter_sets in tqdm(parameters_sep_by_eps,
                                                       leave=True,
                                                       position=0,
                                                       total=len(parameters_sep_by_eps))]
        
        return CRM_ts_eigenspec
    
    ###################################################################################
    
    full_CRM_directory = "C:/Users/jamil/Documents/PhD/Data/resource_diversity_stability/simulations/" + \
                           CRM_directory
            
    # generate filenames based on mu_c
    filenames = [full_CRM_directory + "/simulations_" + \
                 str(M) + "_" + str(np.round(mu_c/M, 4)) + ".pkl"
                 for M in resource_pool_sizes]
        
    CRM_ts_eigenspec = {str(M) : read_call_timescale_separate(filename,
                                                              epsilons)
                        for filename, M in zip(filenames, resource_pool_sizes)}
        
    return CRM_ts_eigenspec

# %%

def absolute_eigenvec_contribution(eigenspecs):
    
    df = pd.DataFrame([absolute_eigenvec_contr_stats(community)
                       for communities in eigenspecs.values()
                       for community_ts in communities
                       for community in community_ts.values()])
    
    return df

def absolute_eigenvec_contr_stats(community):
    
    def contribution(eigenvector, M):
        
        species_contribution = np.mean(np.abs(eigenvector[:M]))
        resource_contribution = np.mean(np.abs(eigenvector[M:]))
        
        return species_contribution, resource_contribution
    
    M = community.no_resources
    timescalar = community.timescalar
    max_le = community.lyapunov_exponent
    
    leading_eigenvector = community.eigenspec_stats[0]["leading_vec"]
    
    species_contribution, resource_contribution = contribution(leading_eigenvector,
                                                               M)
    
    return dict(M = M,
                timescalar = timescalar,
                max_le = max_le,
                species_contribution = species_contribution,
                resource_contribution = resource_contribution)

# %%

def eigenspectrum_matrix(community):
    
    surviving_species = community.ODE_sols[0].y[:community.no_resources, -1] > 1e-4
    surviving_resources = community.ODE_sols[0].y[community.no_resources:, -1] > 1e-4
    
    surviving_consumption = community.consumption[np.ix_(surviving_species,
                                                         surviving_resources)]
    
    surviving_growth = community.growth[np.ix_(surviving_resources,
                                               surviving_species)]
    
    matrix = -surviving_growth @ surviving_consumption
    
    T, Z = schur(matrix, output='complex')
    eigenvalues = np.diag(T)
    
    leading_val = eigenvalues[np.argmax(eigenvalues.real)]
    
    return {'eigenspectrum' : eigenvalues, 'leading_val' : leading_val}
    
# %%

epsilons = np.array([10**-5, 10**-2, 0.1, 1.0])

CRM_ts_eigenspec = timescale_separation_eigenspec(CRM_directory = "M_vs_mu_c",
                                                  epsilons = epsilons,
                                                  resource_pool_sizes = np.array([50, 250]),
                                                  mu_c = 145)

# %%

GC_eigenspec = {M : [eigenspectrum_matrix(community_dict['1.0'])
                     for community_dict in communities]
                for M, communities in CRM_ts_eigenspec.items()}

# %%

def example_eigenspectr(idx,
                        filename):
    
    def collate_eigenspectra(idx):
        
        eigenspectra = [[community.eigenspec_stats[0]['eigenspectrum']
                         for community in communities[idx].values()] + \
                        [GC_eigenspec[M][idx]['eigenspectrum']]
                        for M, communities in CRM_ts_eigenspec.items()]  
            
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
                       'max. le' : np.round(val2.lyapunov_exponent, 5)
                       }
                      for str2, val2 in val1[idx].items()]
                      for str1, val1 in CRM_ts_eigenspec.items()]
        
        titles = [[format_title_from_dict(title_data)
                   for title_data in titles_data_M] + [r"$-GC^T$"]
                  for titles_data_M in titles_data]
        
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
        
        plt.savefig("C:/Users/jamil/Documents/PhD/Figures/resource_diversity_stability/" + \
                    filename + ".png",
                    bbox_inches='tight')
        plt.savefig("C:/Users/jamil/Documents/PhD/Figures/resource_diversity_stability/" + \
                    filename + ".svg",
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
            
            #ax.set_xlim([np.quantile(data.real[data.real < 0], 0.1),
            #             0.05])
            #ax.set_ylim([np.quantile(data.imag[data.imag < 0], 0.01),
            #             np.quantile(data.imag[data.imag > 0], 0.99)])
        
        fig.supxlabel('Re(λ)', weight="bold", fontsize=10)
        fig.supylabel('Im(λ)', weight="bold", fontsize=10)
        
        axs[0, int(len(eigenspectra)/2)-1].set_ylim([-0.75, 0.75])
        axs[0, int(len(eigenspectra)/2)-1].set_xlim([-7, 7])
        axs[1, int(len(eigenspectra)/2)-1].set_ylim([-0.75, 0.75])
        axs[1, int(len(eigenspectra)/2)-1].set_xlim([-7, 7])
        
        plt.savefig("C:/Users/jamil/Documents/PhD/Figures/resource_diversity_stability/" + \
                    filename + "_smallrange.png",
                    bbox_inches='tight')
        plt.savefig("C:/Users/jamil/Documents/PhD/Figures/resource_diversity_stability/" + \
                    filename + "_smallrange.svg",
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

    fig, axs = plt.subplots(2, 4,
                           layout="constrained",
                           #sharex=True,
                           figsize=(8.5, 3.5))
    
    for ax, (epsilon, data) in zip(axs.flatten()[:4],
                                   CRM_ts_eigenspec["50"][idx].items()):
        
        log_epsilon = np.abs(np.round(np.log10(float(epsilon)), 1))
        
        ax.plot(data.ODE_sols[0].t, data.ODE_sols[0].y[:50, :].T)
        ax.set_title(f"$10^{{{log_epsilon}}}$" ,
                     weight="bold",
                     fontsize=10)
        
    for ax, (epsilon, data) in zip(axs.flatten()[4:],
                                   CRM_ts_eigenspec["250"][idx].items()):
        
        log_epsilon = np.abs(np.round(np.log10(float(epsilon)), 1))
        
        ax.plot(data.ODE_sols[0].t, data.ODE_sols[0].y[:250, :].T)
        ax.set_title("" ,
                     weight="bold",
                     fontsize=10)
    
        
    #ax.set_xlabel('')
    #ax.set_ylabel('')
    
    fig.supxlabel('time', weight="bold", fontsize=10)
    fig.supylabel('abundance', weight="bold", fontsize=10)
    
    plt.savefig("C:/Users/jamil/Documents/PhD/Figures/resource_diversity_stability/ts_example_chaos.png",
                bbox_inches='tight')
    plt.savefig("C:/Users/jamil/Documents/PhD/Figures/resource_diversity_stability/ts_example_chaos.svg",
                bbox_inches='tight')
    plt.show()
    
example_dynamics(8)

# %%

eigenvec_contr_df = absolute_eigenvec_contribution(CRM_ts_eigenspec)

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

plt.savefig("C:/Users/jamil/Documents/PhD/Figures/resource_diversity_stability/ts_abs_eigenvec_contr.png",
            bbox_inches='tight')
plt.savefig("C:/Users/jamil/Documents/PhD/Figures/resource_diversity_stability/ts_abs_eigenvec_contr.svg",
            bbox_inches='tight')