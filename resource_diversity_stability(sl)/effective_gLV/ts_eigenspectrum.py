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

sys.path.insert(0, "C:/Users/jamil/Documents/PhD/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models" + \
                    "/resource_diversity_stability(sl)")
from complete_simulation_functions import pickle_dump

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

def save_eigenspec_stats(CRM_ts_eigenspec : dict,
                         filepath : str) -> None:

    '''

    Save just the eigenspectrum stats (and max. lyapunov exponent) for every
    community in CRM_ts_eigenspec - a lightweight pickle that leaves out the
    much larger ODE_sols trajectories.

    Parameters
    ----------
    CRM_ts_eigenspec : dict
        {M : [{epsilon : community, ...}, ...]}, as returned by
        timescale_separation_eigenspec().
    filepath : str
        Full filepath (including filename and extension) to save to.

    Returns
    -------
    None.

    '''

    eigenspec_stats = {M : [{epsilon : dict(eigenspec_stats = community.eigenspec_stats,
                                            lyapunov_exponent = community.lyapunov_exponent)
                             for epsilon, community in community_ts.items()}
                            for community_ts in communities]
                       for M, communities in CRM_ts_eigenspec.items()}

    pickle_dump(filepath, eigenspec_stats)

# %%

def save_example_trajectories(CRM_ts_eigenspec : dict,
                              community_indices : list[int],
                              filepath : str) -> None:

    '''

    Save (t, y) from ODE_sols[0] - across every epsilon - for a chosen subset
    of communities, as a pickle.

    Parameters
    ----------
    CRM_ts_eigenspec : dict
        {M : [{epsilon : community, ...}, ...]}, as returned by
        timescale_separation_eigenspec().
    community_indices : list[int]
        Indices into each M's list of communities (i.e. into
        CRM_ts_eigenspec[M]) to save trajectories for.
    filepath : str
        Full filepath (including filename and extension) to save to.

    Returns
    -------
    None.

    '''

    example_trajectories = {M : {idx : {epsilon : (community.ODE_sols[0].t, community.ODE_sols[0].y)
                                        for epsilon, community in communities[idx].items()}
                                 for idx in community_indices}
                            for M, communities in CRM_ts_eigenspec.items()}

    pickle_dump(filepath, example_trajectories)

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

eigenspec_directory = "C:/Users/jamil/Documents/PhD/Data/resource_diversity_stability/simulations/CRM_TS/eigenspectra"

if not os.path.exists(eigenspec_directory):

    os.makedirs(eigenspec_directory)

save_eigenspec_stats(CRM_ts_eigenspec,
                     eigenspec_directory + "/M_vs_mu_c_eigenspec_stats.pkl")

save_example_trajectories(CRM_ts_eigenspec,
                          community_indices = [0],
                          filepath = eigenspec_directory + "/M_vs_mu_c_example_trajectories.pkl")