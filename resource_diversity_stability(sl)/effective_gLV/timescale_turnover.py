# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 16:22:14 2026

@author: jamil
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import os
import sys
from tqdm import tqdm
from typing import Literal, Union
import numpy.typing as npt
from copy import deepcopy
import itertools
from matplotlib import pyplot as plt

os.chdir('C:/Users/jamil/Documents/PhD/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models/resource_diversity_stability(sl)/effective_gLV')

sys.path.insert(0, "C:/Users/jamil/Documents/PhD/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models" + \
                    "/consumer_resource_modules")
from models import Consumer_Resource_Model
from effective_LV_models import eLV_SL
from community_level_properties import max_le
    
sys.path.insert(0, "C:/Users/jamil/Documents/PhD/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models" + \
                    "/resource_diversity_stability(sl)")
from complete_simulation_functions import pickle_dump, simulation_df_from_communities

# %%

def max_le_tau(community : Union["Consumer_Resource_Model", "eLV_SL"]):
    
    times = np.arange(50, 1050, 50)
    
    max_LEs = [max_le(community,
                      community.ODE_sols[0].y[:, -1],
                      T = t,
                      perturbation = 1e-6)
               for t in times]
    
    return dict(tau = np.concatenate([[0], times]),
                max_le = np.concatenate([[1e-6], max_LEs]))

# %%

def total_community_diversity(community : Union["Consumer_Resource_Model", "eLV_SL"]):

    def total_diversity(y):
        
        return np.any(y[:, 30:] > 1e-4, axis = 1).sum()/y.shape[0]
    
    abundances = community.ODE_sols[0].y
    
    match type(community).__name__:
    
        case "SL_CRM":
    
            M = community.no_resources
            tot_diverse = {'phiR_overtime' : total_diversity(abundances[M:]),
                            'phiN_overtime' : total_diversity(abundances[:M])}
        
        case "eLV_SL":
            
            tot_diverse = {'phiR_overtime' : np.nan,
                           'phiN_overtime' : total_diversity(abundances)}
        
    return tot_diverse
    
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

def effective_r_A(community : "Consumer_Resource_Model",
                  cavity_phi_R : Union[float, None]):
    
    def surviving_resources(resources,
                            M,
                            cavity_phi_R):
        
        potential_ext_thresh = np.linspace(-8, -2, 200)
        
        extinct_thresh = \
        10**(potential_ext_thresh[np.abs(np.array([np.sum(resources[:, -1] > 10**(e_t))/M
                                                   for e_t in potential_ext_thresh]) - cavity_phi_R).argmin()])
        
        surviving_resources = deepcopy(resources)
        
        surviving_resources[surviving_resources < extinct_thresh] = 0.0
        
        return surviving_resources
       
    S = community.no_species
    B = community.b
    C = community.consumption
    G = community.growth
    D = community.d
    
    abundances = community.ODE_sols[0].y
    species = deepcopy(abundances[:S, :])
    all_resources = deepcopy(abundances[S:, :])
    
    if cavity_phi_R:
        
        resources = surviving_resources(all_resources,
                                        S,
                                        cavity_phi_R)
        
    else:
        
        resources = all_resources
    
    #resources = all_resources
    
    #resources[resources < 1e-4] = 0
    
    B_eff = B[:, None] - C @ species
    
    shared_quant = resources / B_eff
    
    #np.divide(resources, B_eff,
                   #          out=np.zeros(resources.shape),
                   #          where=B_eff > 1e-12)
    
    r = (G @ (B[:, None] * shared_quant) - D[:, None]).T # dims (t_end, S)
    A = (G[None, :, :] * shared_quant.T[:, None, :]) @ C # dims (t_end, S, S)
    
    return r, A

# %%

def CRM_timescale(parameters : dict,
                  fast_variable : Literal["resources",
                                          "consumers"],
                  cavity_phi_R : Union[float, None] = None) -> any:
    
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
    
    if fast_variable == "resources":
    
        community.timescale_separation(epsilon_r = epsilon)
        
    elif fast_variable == "species":
        
        community.timescale_separation(epsilon_s = epsilon)
        
    # run simulations from randomly generated initial abundances
    community.simulate_community(t_end = 7000,
                                 no_init_cond = 1)
    
    community.calculate_community_properties()
       
    # numerically estimate the max. lyapunov exponent
    max_LEs_tau = max_le_tau(community)
    
    community.lyapunov_exponent = max_LEs_tau['max_le'][-1]
    stability = "stable" if community.lyapunov_exponent < 0 else "unstable"
    
    # effective consumer-only model
    r_eff, A_eff = effective_r_A(community,
                                 cavity_phi_R)
    
    # cumulative diversity
    cumulative_diversity = total_community_diversity(community)
    
    return dict(M = M,
                epsilon = epsilon,
                stability = stability,
                max_les_tau = max_LEs_tau,
                r = r_eff,
                A = A_eff,
                cumulative_diversity = cumulative_diversity,
                community = community)

# %%

def eLV_comparison(CRM_community : Literal["SL_CRM"],
                   cavity_phi_R : Union[float, None] = None) -> any:
    
    # initialise eLV (generate growth rates, interaction matrices, etc from the CRM)
    community = eLV_SL()
    
    if cavity_phi_R: 
    
        community.elv_from_crm(CRM_community,
                               cavity_phi_R)
    
    else:
        
        community.elv_from_crm(CRM_community)
        
    community.generate_elv_parameters()
    community.calculate_interaction_stats()
    
    # run simulations from randomly generated initial abundances
    community.simulate_community(t_end = 7000,
                                 no_init_cond = 1)
    
    # numerically estimate the max. lyapunov exponent
    max_LEs_tau = max_le_tau(community)
    
    community.lyapunov_exponent = max_LEs_tau['max_le'][-1]
    stability = "stable" if community.lyapunov_exponent < 0 else "unstable"
    
    # cumulative diversity
    cumulative_diversity = total_community_diversity(community)
    
    return dict(M = community.no_resources,
                epsilon = "eLV",
                stability = stability,
                max_les_tau = max_LEs_tau,
                r = community.r,
                A = community.interaction_matrix,
                cumulative_diversity = cumulative_diversity,
                community = community)
    
# %%

def CRM_timescale_separation(CRM_directory : str,
                             epsilons,
                             resource_pool_sizes,
                             mu_c,
                             extra_name : str = "",
                             fast_variable : Literal["resources",
                                                     "species"] = "resources",
                             all_resource_survive : bool = False):


    def read_call_timescale_separate(full_CRM_directory : str,
                                     epsilons : npt.NDArray,
                                     sces : Union[any, None]):
        
        # read in consumer-resource model (CRM) communities
        CRM_communities = pd.read_pickle(full_CRM_directory)
        
        parameters_sep_by_eps = np.array([separate_parameter_timescales(CRM_community,
                                                                        epsilons)
                                          for CRM_community in CRM_communities]).flatten()
        
        kwargs_comm = {}
        
        if sces is not None: 
            
            mu_c_temp = parameters_sep_by_eps[0]['mu_c']
            M_temp = parameters_sep_by_eps[0]['no_resources']
            
            kwargs_comm["cavity_phi_R"] = \
                    sces.loc[np.where((sces["mu_c"] == 
                                       np.round(mu_c_temp * M_temp, 4)) & \
                                      (sces["M"] == M_temp)),
                             "phi_R"].to_numpy()
        
        CRM_ts_communities = [CRM_timescale(parameters,
                                            fast_variable,
                                            **kwargs_comm)
                              for parameters in
                              tqdm(parameters_sep_by_eps,
                                   leave = True,
                                   position = 1,
                                   total = len(parameters_sep_by_eps))]
        
        eLV_communities = [eLV_comparison(CRM_community,
                                          **kwargs_comm)
                           for CRM_community in
                           tqdm(CRM_communities,
                                leave = True,
                                position = 1,
                                total = len(CRM_communities))]
        
        return CRM_ts_communities + eLV_communities
    
    ###################################################################################
    
    full_CRM_directory = "C:/Users/jamil/Documents/PhD/Data/resource_diversity_stability/simulations/" + \
                           CRM_directory
                                       
    # generate filenames based on mu_c
    filenames = ["simulations_" + \
                 str(M) + "_" + str(np.round(mu_c/M, 4)) + extra_name
                 for M in resource_pool_sizes]
        
    kwargs = {}
        
    if all_resource_survive is False:
    
        # get the resource survival fraction
        kwargs["sces"] = pd.read_pickle("C:/Users/jamil/Documents/PhD/Data/resource_diversity_stability/self_consistency_equations/" + \
                                        CRM_directory + ".pkl")
            
    communities_list = list(itertools.chain.from_iterable([read_call_timescale_separate(full_CRM_directory + "/" + filename + ".pkl",
                                                                                        epsilons,
                                                                                        **kwargs)
                                           for filename in tqdm(filenames,
                                                                leave = True,
                                                                position = 0,
                                                                total = len(filenames))]))
    
    return pd.DataFrame(communities_list).to_dict(orient="list")
        
# %%

epsilons = 10.0**np.array([-2.0, 0.0]) #10.0**np.array([-5.0, 1.0])

CRM_eLV_comp = CRM_timescale_separation(CRM_directory = "M_vs_mu_c",
                                        epsilons = epsilons,
                                        resource_pool_sizes = np.arange(50, 275, 25),
                                        mu_c = 145)

# %%

directory = "C:/Users/jamil/Documents/PhD/Data/resource_diversity_stability/simulations/" + \
            "CRM_TS/properties_over_time"

if not os.path.exists(directory):
    
    os.makedirs(directory)

CRM_eLV_comp_to_save = {key : val
                        for key, val in CRM_eLV_comp.items()
                        if key not in ["community",
                                       "A",
                                       "r",
                                       "max_les_tau",
                                       "cumulative_diversity"]}

CRM_eLV_comp_to_save["phiN_overtime"] = [community_data["phiN_overtime"]
                                         for community_data in CRM_eLV_comp['cumulative_diversity']]

CRM_eLV_comp_to_save["phiR_overtime"] = [community_data["phiR_overtime"]
                                         for community_data in CRM_eLV_comp['cumulative_diversity']]

pd.DataFrame(CRM_eLV_comp_to_save).to_csv(directory + "/timescalar_1_001_stats.csv")
del CRM_eLV_comp_to_save

#####

max_les_taus_df = pd.concat([pd.DataFrame({'M': M,
                                           'epsilon': epsilon,
                                           'community' : community_idx,
                                           't': max_le_tau_comm['tau'],
                                           'max_le': max_le_tau_comm['max_le'],
                                           })
                             for M, epsilon, community_idx, max_le_tau_comm in 
                             zip(CRM_eLV_comp['M'],
                                 CRM_eLV_comp['epsilon'],
                                 np.repeat(np.arange(0,
                                                     len(CRM_eLV_comp['M'])/2,
                                                     1),
                                           2),
                                 CRM_eLV_comp['max_les_tau'])],
                            ignore_index=True)

pd.DataFrame(max_les_taus_df).to_csv(directory + "/timescalar_1_001_maxles.csv")
del max_les_taus_df

def collect_r_Aij_M():

    def stability_checker(maxles_0,
                          epsilon_0,
                          epsilon_1,
                          rs_0,
                          rs_1,
                          As_0,
                          As_1,
                          community_0,
                          community_1):
        
        conditions = {
        'unstable': lambda x: x > 10**-3,
        'stable': lambda x: x < -10**-3,
        'oscillatory': lambda x: (x < 10**-3) & (x > 10**-3),
        }
    
        results = {}
        
        for stab0, eps0, eps1, r0, r1, A0, A1, comm0, comm1 in zip(maxles_0,
                                                                   epsilon_0,
                                                                   epsilon_1,
                                                                   rs_0,
                                                                   rs_1,
                                                                   As_0,
                                                                   As_1,
                                                                   community_0,
                                                                   community_1):
            
            final_max_le = stab0['max_le'][-1]
            
            for name, condition in conditions.items():
            
                if name in results:
                
                    continue 
        
                if condition(final_max_le):
                    
                    results[name] = dict(epsilon = [eps0, eps1],
                                         max_le = final_max_le,
                                         r = [r0, r1],
                                         A = [A0, A1],
                                         simulation = [(comm0.ODE_sols[0].t,
                                                        comm0.ODE_sols[0].y),
                                                       (comm1.ODE_sols[0].t,
                                                        comm1.ODE_sols[0].y)])
        
            if len(results) == len(conditions):
                
                break
                
        return results
    
    Ms, M_idx = np.unique(CRM_eLV_comp['M'], return_index=True)
    M_idx = np.append(M_idx, len(CRM_eLV_comp['M']))
    
    M_stab_stats = {str(M) : stability_checker(CRM_eLV_comp['max_les_tau'][M_i : M_i + 40][::2],
                                                 CRM_eLV_comp['epsilon'][M_i : M_i + 40][::2],
                                                 CRM_eLV_comp['epsilon'][M_i : M_i + 40][1::2],
                                                 CRM_eLV_comp['r'][M_i : M_i + 40][::2],
                                                 CRM_eLV_comp['r'][M_i : M_i + 40][1::2],
                                                 CRM_eLV_comp['A'][M_i : M_i + 40][::2],
                                                 CRM_eLV_comp['A'][M_i : M_i + 40][1::2],
                                                 CRM_eLV_comp['community'][M_i : M_i + 40][::2],
                                                 CRM_eLV_comp['community'][M_i : M_i + 40][1::2])
                    for M, M_i in zip(Ms,
                                      M_idx[:-1])}

    return M_stab_stats

M_stab_stats = collect_r_Aij_M()
                    
pickle_dump(directory + "/timescalar_2_1_001_example_interactions.pkl",
            M_stab_stats)
del M_stab_stats

