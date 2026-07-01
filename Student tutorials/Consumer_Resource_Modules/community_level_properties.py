# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 10:24:02 2024

@author: jamil
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from typing import TYPE_CHECKING, Union
from copy import deepcopy
import sys
from scipy.stats import binned_statistic

########## type checking ########

if TYPE_CHECKING:
    
    sys.path.insert(0, "C:/Users/jamil/Documents/PhD/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models" + \
                        "/consumer_resource_modules")
    from models import SL_CRM, SL_SI_CRM, SL_TL_CRM, ES_CRM, Hybrid_CRM
    from effective_LV_models import eLV_SL, gLV
    
# %%

#print(f"eLV_SL id: {id(eLV_SL)}")
#print(f"eLV_SL module: {eLV_SL.__module__}")

# %%
########### Community properties #############

class CommunityPropertiesInterface:
        
    def calculate_community_properties(self,
                                       sensitivity_distribution : bool = False):
        '''
        
        Call methods that calculate the moments of the species and resource
        abundance distribution, then assign them as attributes to the 
        consumer resource model object.

        Parameters
        ----------
        average_property : Bool, optional
            Whether or not to calculate community properties at the end of simulations (False)
            or over as an average over some time period (True). The default is False.
        time_window : float, optional
            If average_property is True, this is the time window to calculate
            community properties over. The default is 500.

        Returns
        -------
        None.

        '''
        
        trophic_levels = getattr(self, "trophic_levels", None)
        
        if trophic_levels is None:
            
            self.assign_abundance_distribution("resource",
                                                 [0, self.no_consumers])
            
            self.assign_abundance_distribution("consumer",
                                                 [self.no_consumers, -1])
            
            if sensitivity_distribution == True:
            
                self.assign_sensitivity_distribution()
        
        else:
            
            pool_idx = np.append(0, np.cumsum(self.pool_sizes))
            
            for i in np.arange(0, trophic_levels):
                
                self.assign_abundance_distribution("TL_" + str(i + 1),
                                                     [pool_idx[i],
                                                      pool_idx[i+1]])
            
    def assign_abundance_distribution(self,
                                      attr_name : str,
                                      attr_idx : int):
        
        distributions = \
            [self.abundance_distribution(simulation.y[attr_idx[0] : attr_idx[1],
                                                      -1])    
             for simulation in self.ODE_sols] 
        
        # assign survival fraction
        setattr(self,
                attr_name + "_survival_fraction",
                [dist[0] for dist in distributions])
        
        # assign average abundance
        
        setattr(self,
                attr_name + "_avg_abundance",
                [dist[1] for dist in distributions])
        
        # assign 2nd moment in bundance distribution
        setattr(self,
                attr_name + "_abundance_fluctuations",
                [dist[2] for dist in distributions])
             
    def abundance_distribution(self,
                               abundances : npt.NDArray,
                               extinct_thresh : float = 1e-4):
        
        '''
        
        Calculate the moments of the species and resource abundance distribution

        Parameters
        ----------
        abundances : np.ndarray
            Abundances of some variable, either over a time frame or at the end of simulations.
        extinct_thresh : float, optional
            Extinction threshold for the variable. The default is 1e-4.

        Returns
        -------
        zeroth_moment : float
            0th moment of the abundance distribution.
        first_moment : float
            1st moment of the abundance distribution.
        second_moment : float
            1st moment of the abundance distribution.

        '''
        
        zeroth_moment = \
            np.count_nonzero(abundances > extinct_thresh)/len(abundances)
            
        first_moment = np.mean(abundances)
        
        second_moment = np.mean(abundances**2)
        
        return zeroth_moment, first_moment, second_moment
    
    def assign_sensitivity_distribution(self):
        
        distributions = \
            [self.sensitivity_distribution(simulation.y)    
             if np.ndim(simulation.y) > 1
             else {}
             for simulation in self.ODE_sols]
                        
        setattr(self,
                "consumption_sensitivity",
                distributions)
    
    def sensitivity_distribution(self,
                                 abundances : npt.NDArray):
        
        if abundances.shape[1] < 10:
            
            outputs = [self.dRdconsumption(abundances, i) 
                       for i in np.arange(-abundances.shape[1], 0, 1)]
        
        else:
        
            outputs = [self.dRdconsumption(abundances, i) 
                       for i in np.arange(-10, 0, 1)]
        
        concat_outputs = [np.concatenate([output[i] for output in outputs])
                          for i in range(3)]
        
        consumption_sensitivities = {key : binned_statistic(concat_outputs[0],
                                                            var,
                                                            statistic='mean',
                                                            bins=15).statistic
                                     for key, var in zip(['x',
                                                          'R',
                                                          'dRdx2'],
                                                         concat_outputs)}
        
        return consumption_sensitivities
    
    def dRdconsumption(self,
                       abundances : npt.NDArray,
                       i : int):
        
        consumers = abundances[self.no_consumers:, i]
        resources = abundances[:self.no_consumers, i]
        
        tot_c = np.sum(self.consumption * consumers, axis=1)
        
        sort_idx = np.argsort(tot_c)
        
        sorted_tot_c = tot_c[sort_idx]
        sorted_resources = resources[sort_idx]
        
        dRdtotC = np.gradient(sorted_resources, sorted_tot_c)**2
        
        return sorted_tot_c, sorted_resources, dRdtotC
            
###########################################################################################################

# %%

def max_le(community : Union["SL_CRM", "SL_SI_CRM", "SL_TL_CRM", "ES_CRM",
                             "Hybrid_CRM",
                             "eLV_SL", "gLV"],
           initial_conditions : npt.NDArray,
           T : float = 1000,
           perturbation : float = 1e-6):
    
    match type(community).__name__:
        
        case glv if glv in ["eLV_SL", "gLV"]:
            
            original_traj, perturbed_traj = trajectory_LV(community,
                                                          initial_conditions,
                                                          T,
                                                          perturbation)
            
        case CRM if CRM in ["SL_CRM", "SL_SI_CRM", "ES_CRM", "Hybrid_CRM"]:
            
            original_traj, perturbed_traj = trajectory(community,
                                                       initial_conditions,
                                                       T,
                                                       perturbation)
            
        case "SL_TL_CRM":
            
            original_traj, perturbed_traj = \
                trajectory_multi_trophic(community,
                                         initial_conditions,
                                         T,
                                         perturbation)
     
    try:
        
        # Calculated the new separation between the original and perturbated trajectory (d1)
        separation = np.log(np.linalg.norm(perturbed_traj.y - original_traj.y,
                                           axis=0))
   
        separation_grad_abs = np.abs(np.gradient(separation,
                                                 perturbed_traj.t))
        
        max_lyapunov_exponent = calculate_max_le(original_traj,
                                                 perturbed_traj,
                                                 separation,
                                                 separation_grad_abs)
        
    except (IndexError, ValueError, TypeError):
        
        max_lyapunov_exponent = np.nan
    
    return max_lyapunov_exponent

############

def trajectory(community : Union["SL_CRM", "SL_SI_CRM", "ES_CRM"],
               initial_conditions : npt.NDArray,
               T : float,
               perturbation : float):

# Set initial conditions of the original and perturbated trajectory
    original_conditions = deepcopy(initial_conditions)
    
    perturbed_conditions = deepcopy(initial_conditions)
    perturbed_conditions += perturbation * np.ones(len(perturbed_conditions)) #np.random.uniform(-1, 1, len(perturbed_conditions))
    
    # Simulate the original community trajectory for time = T
    original_traj = community.simulate_community(T, 1, init_cond_func='user-supplied',
                                                 assign = False,
                                                 initial_conditions = [original_conditions])
    # Simulate the perturbated community trajectory for time = T
    perturbed_traj = community.simulate_community(T, 1, init_cond_func='user-supplied',
                                                  assign = False,
                                                  initial_conditions = [perturbed_conditions])
    
    return original_traj[0], perturbed_traj[0]

#############

def trajectory_multi_trophic(community : "SL_TL_CRM",
                             initial_conditions : npt.NDArray,
                             T : float,
                             perturbation : float):

# Set initial conditions of the original and perturbated trajectory
    original_conditions = deepcopy(initial_conditions)

    perturbed_conditions = deepcopy(initial_conditions)
    perturbed_conditions += perturbation * np.ones(len(perturbed_conditions)) #np.random.uniform(-1, 1, len(perturbed_conditions))
    
    # Simulate the original community trajectory for time = T
    original_traj = community.simulate_community(T, 1, init_cond_func='user-supplied',
                                                 assign = False,
                                                 initial_conditions = [original_conditions])
    # Simulate the perturbated community trajectory for time = T
    perturbed_traj = community.simulate_community(T,
                                                  1,
                                                  init_cond_func='user-supplied',
                                                  assign = False,
                                                  initial_conditions = [perturbed_conditions])
    
    return original_traj[0], perturbed_traj[0]

#############
    
def trajectory_LV(community : Union["eLV_SL", "gLV"],
                  initial_conditions : npt.NDArray,
                  T : float,
                  perturbation : float):

# Set initial conditions of the original and perturbated trajectory
    original_conditions = deepcopy(initial_conditions)
    
    perturbed_conditions = deepcopy(initial_conditions)
    perturbed_conditions += perturbation * np.ones(len(perturbed_conditions))
    
    # Simulate the original community trajectory for time = T
    original_traj = community.simulate_community(T, 
                                                 1,
                                                 init_cond_func='user-supplied',
                                                 assign = False,
                                                 initial_conditions = [original_conditions])
    # Simulate the perturbated community trajectory for time = T
    perturbed_traj = community.simulate_community(T, 
                                                 1,
                                                 init_cond_func='user-supplied',
                                                 assign = False,
                                                 initial_conditions = [perturbed_conditions])
    
    return original_traj, perturbed_traj

#########

def calculate_max_le(original_traj : npt.NDArray,
                     perturbed_traj : npt.NDArray,
                     separation : float,
                     separation_grad_abs : npt.NDArray):
    
    if np.all(np.convolve(separation_grad_abs,
                          np.ones(40)/40,
                          mode='valid') > 0.001) == False:
        
        final_idx = -1
    
    else: 
        
        cutoff_t = np.convolve(perturbed_traj.t,
                               np.ones(40)/40,
                               mode='valid')[np.convolve(separation_grad_abs,
                                                         np.ones(40)/40,
                                                         mode='valid') > 0.001][-1]
                                        
        final_idx = np.abs(perturbed_traj.t - cutoff_t).argmin()
                                                     
    max_lyapunov_exponent, log_offset = np.polyfit(perturbed_traj.t[10 : final_idx],
                                                   separation[10 : final_idx],
                                                   1)
    
    return max_lyapunov_exponent