# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 14:13:06 2025

@author: jamil

"""

# %%

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from copy import deepcopy
from typing import TYPE_CHECKING, Union

########## type checking ########

if TYPE_CHECKING:
    
    from models import SL_CRM
    
# %%

class eLV_SL():
    
    def __init__(self,
                 CRM_community : "SL_CRM",
                 cavity_phi_R : float):
        
        self.no_resources = CRM_community.no_resources
        self.no_species = CRM_community.no_species
                 
        self.consumption = CRM_community.consumption
        self.consumer_growth = CRM_community.growth
        
        self.resource_b = CRM_community.b
        self.consumer_d = CRM_community.d
        
        resource_abundances = CRM_community.ODE_sols[0].y[CRM_community.no_species :, -1]
        
        #####
        
        potential_ext_thresh = np.linspace(-8, -2, 200)
        
        extinct_thresh = \
        10**(potential_ext_thresh[np.abs(np.array([np.sum(resource_abundances > 10**(e_t))/self.no_resources
                                                   for e_t in potential_ext_thresh]) - cavity_phi_R).argmin()])
        
        self.phi_R = np.sum(resource_abundances > extinct_thresh)/self.no_resources
        
        ###############
        
        resource_pa = np.int64(resource_abundances > extinct_thresh).astype(np.float64)
        
        self.species_growth = np.sum((self.consumer_growth * self.resource_b * resource_pa),
                                     axis = 1) - self.consumer_d
        self.interaction_matrix = np.dot(self.consumer_growth,
                                         (self.consumption.T * resource_pa).T)
        
        self.initial_abundances = CRM_community.ODE_sols[0].y[: CRM_community.no_species, -1]
        self.phi_N = np.sum(self.initial_abundances > 1e-4)/self.no_species
        
    def interaction_statistics(self):
        
        ### growth rates ###
        
        self.mu_r = np.mean(self.species_growth)
        self.sigma_r = np.std(self.species_growth)
        
        ### interactions ###
        
        self_inhibition = np.diagonal(self.interaction_matrix)
        
        inter_species_interactions = self.interaction_matrix[~np.eye(self.interaction_matrix.shape[0],
                                                                     dtype=bool)].reshape((self.no_species,
                                                                                           self.no_species - 1))
        
        self.mu_Aii = np.mean(self_inhibition)
        self.sigma_Aii = np.std(self_inhibition)
        self.mu_Aij = np.mean(inter_species_interactions)
        self.sigma_Aij = np.std(inter_species_interactions) 
         
        total_interact_per_species = np.sum(inter_species_interactions, axis = 1)
        
        self.interaction_statistics = dict(Aii = self_inhibition,
                                           sum_j_Aij = total_interact_per_species,
                                           mu_Aij_tot = self.mu_Aij * \
                                               np.float64(self.no_species), # (np.float64(self.no_species) * self.phi_N),
                                           sigma_Aij_tot = self.sigma_Aij * \
                                               np.sqrt(np.float64(self.no_species)))
                                           #    np.sqrt(np.float64(self.no_species) * self.phi_N))
        
    def simulation(self,
                   t_end : float,
                   initial_abundances : Union[None, np.typing.NDArray] = None,
                   assign : bool = True):
        
        def gLV(t, species, r, A):
            
            # change in consumer abundances over time
            dNdt = species * (r - np.sum(A * species, axis = 1))
        
            return dNdt + 1e-8
        
        def unbounded_growth(t, var, *args):
            
            
            # if any species or resource abundances are greater than some threshold
            # or if any species abundances are less than or equal to 0  
            if np.any(np.log10(np.abs(var) + 1e-20) > 4) or np.isnan(np.log10(np.abs(var) + 1e-20)).any():
                
                return 0 # when the returned value of an event function is 0, the ode 
                            #solver terminates.
            
            else: 
                
                return 1 # the ode solver continues because the returned value is non-zero.
            
        if initial_abundances is None:
            
            sol = solve_ivp(gLV,
                            [0, t_end],
                            self.initial_abundances, 
                            args = (self.species_growth, self.interaction_matrix),
                            method = 'LSODA',
                            rtol = 1e-7, atol = 1e-9,
                            t_eval = np.linspace(0, t_end, 200),
                            events = unbounded_growth)  
            
        else:
            
            sol = solve_ivp(gLV,
                            [0, t_end],
                            initial_abundances, 
                            args = (self.species_growth, self.interaction_matrix),
                            method = 'LSODA',
                            rtol = 1e-7, atol = 1e-9,
                            t_eval = np.linspace(0, t_end, 200),
                            events = unbounded_growth)
        
        if assign is True:
        
            self.ODE_sol = sol
            
        else:
         
            return sol 
        
# %%

class gLV():
    
    def __init__(self,
                 no_species : float):
        
        self.no_species = no_species
                  
    def generate_interaction_matrix(self,
                                    mu_a : float,
                                    sigma_a : float):
        
        self.mu_a = mu_a
        self.sigma_a = sigma_a
        
        ### interactions ###
        
        interaction_matrix = self.mu_a + self.sigma_a * np.random.randn(self.no_species,
                                                                        self.no_species)
        np.fill_diagonal(interaction_matrix, 1)
        
        self.interaction_matrix = interaction_matrix
        
    def simulation(self,
                   t_end : float,
                   initial_abundances : np.typing.NDArray,
                   assign : bool = True):
        
        def gLV(t, species, A):
            
            # change in consumer abundances over time
            dNdt = species * (1 - np.sum(A * species, axis = 1))
        
            return dNdt + 1e-8
        
        def unbounded_growth(t, var, *args):
            
            
            # if any species or resource abundances are greater than some threshold
            # or if any species abundances are less than or equal to 0  
            if np.any(np.log10(np.abs(var) + 1e-20) > 4) or np.isnan(np.log10(np.abs(var) + 1e-20)).any():
                
                return 0 # when the returned value of an event function is 0, the ode 
                            #solver terminates.
            
            else: 
                
                return 1 # the ode solver continues because the returned value is non-zero.
                
        sol = solve_ivp(gLV,
                        [0, t_end],
                        initial_abundances, 
                        args = (self.interaction_matrix, ),
                        method = 'LSODA',
                        rtol = 1e-7, atol = 1e-9,
                        t_eval = np.linspace(0, t_end, 200),
                        events = unbounded_growth)
        
        if assign is True:
        
            self.ODE_sol = sol
            
        else:
         
            return sol 
    
