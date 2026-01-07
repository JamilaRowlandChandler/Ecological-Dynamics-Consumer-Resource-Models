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
from matplotlib import pyplot as plt
import sys

########## type checking ########

if TYPE_CHECKING:
    
    sys.path.insert(0, "C:/Users/jamil/Documents/PhD/GitHub projects/Ecological-Dynamics-and-Community-Selection/" + \
                        "Ecological-Dynamics/Consumer-Resource Models/consumer_resource_modules")
    from models import SL_CRM, SL_SI_CRM, ES_CRM
    from effective_LV_models import eLV_SL, gLV
    
# %%

#print(f"eLV_SL id: {id(eLV_SL)}")
#print(f"eLV_SL module: {eLV_SL.__module__}")

# %%
########### Community properties #############

class CommunityPropertiesInterface:
    
    def calculate_community_properties(self,
                                       average_property : bool = False,
                                       time_window : float = 500):
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
        
        if average_property is True:
            
            # calculate average community property over some time window
            
            start_idx = np.abs(self.ODE_sols[0].t - (self.t_end - time_window)).argmin()
            
        else:
            
            # calculate community properties at the end of simulations 
            
            start_idx = -1
            
        # moments of the species abundance distribution 
        
        # extract species abundance distribution and calculate its 0th-2nd moment
        species_distributions = \
            [self.__abundance_distribution(simulation.y[:self.no_species,
                                                        start_idx:])    
             for simulation in self.ODE_sols] 
        
        # assign species survival fraction (phi_N, or S*/S) 
        self.species_survival_fraction = [dist[0] 
                                          for dist in species_distributions]
        
        # assign average species abundance (<N>)
        self.species_avg_abundance = [dist[1] 
                                      for dist in species_distributions]
        
        # assign 2nd moment in speices abundance distribution (<N^2>)
        self.species_abundance_fluctuations = [dist[2] 
                                               for dist in species_distributions]
        
        # moments of the resource abundance distribution (same process as with species)
        
        resource_distributions = \
            [self.__abundance_distribution(simulation.y[self.no_species:,
                                                        start_idx:])    
             for simulation in self.ODE_sols] 
            
        self.resource_survival_fraction = [dist[0] 
                                          for dist in resource_distributions]
        
        self.resource_avg_abundance = [dist[1] 
                                      for dist in resource_distributions]

        self.resource_abundance_fluctuations = [dist[2] 
                                               for dist in resource_distributions]
            
    def __abundance_distribution(self,
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
        
        if abundances.shape[1] == 1: # one abundance distribution, not over time period
            
            zeroth_moment = \
                np.count_nonzero(abundances > extinct_thresh)/len(abundances)
                
            first_moment = np.mean(abundances)
            
            second_moment = np.mean(abundances**2)
        
        elif abundances.shape[1] > 1: # multiple abundance distributions over time period
            
            zeroth_moment = \
                np.mean(np.count_nonzero(abundances > extinct_thresh,
                                         axis = 0)/abundances.shape[0])
                
            first_moment = np.mean(np.count_nonzero(abundances > extinct_thresh,
                                    axis = 0)/abundances.shape[0])
            
            second_moment = np.mean(np.mean(abundances**2, axis = 0))
        
        return zeroth_moment, first_moment, second_moment
    
###########################################################################################################

# %%

def max_le(community : Union["SL_CRM", "SL_SI_CRM", "ES_CRM", "eLV_SL", "gLV"],
           initial_conditions : npt.NDArray,
           T : float = 100,
           perturbation : float = 1e-8):
    
    if type(community).__name__ == "eLV_SL" or type(community).__name__ == "gLV":
        
        original_traj, perturbed_traj = trajectory_LV(community,
                                                      initial_conditions,
                                                      T,
                                                      perturbation)
    
    else:
    
        original_traj, perturbed_traj = trajectory(community,
                                                   initial_conditions,
                                                   T,
                                                   perturbation)
    
    if original_traj.y.shape[1] != perturbed_traj.y.shape[1]:
        
        trajectories = [original_traj, perturbed_traj]
        
        long_idx = [traj.y.shape[1] for traj in trajectories].argmax()
        shortest_traj_length = np.min([traj.y.shape[1] for traj in trajectories])
        
        trajectories[long_idx].t = trajectories[long_idx].t[:shortest_traj_length]
        trajectories[long_idx].y = trajectories[long_idx].y[: , :shortest_traj_length]
        
        original_traj, perturbed_traj = trajectories
        
    # Calculated the new separation between the original and perturbated trajectory (d1)
    separation = np.log(np.linalg.norm(perturbed_traj.y - original_traj.y,
                                       axis=0))
    
    try:
   
        separation_grad_abs = np.abs(np.gradient(separation,
                                                 perturbed_traj.t))
        
        max_lyapunov_exponent = calculate_max_le(original_traj, perturbed_traj,
                                                 separation, separation_grad_abs)
        
    except IndexError:
        
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
    original_traj = community.simulate_community(T, 1, init_cond_func='user supplied',
                                                 assign = False,
                                                 user_supplied_init_cond = 
                                                 {'species' : original_conditions[:community.no_species],
                                                  'resources' : original_conditions[community.no_species:]})
    # Simulate the perturbated community trajectory for time = T
    perturbed_traj = community.simulate_community(T, 1, init_cond_func='user supplied',
                                                  assign = False,
                                                  user_supplied_init_cond = 
                                                  {'species' : perturbed_conditions[:community.no_species],
                                                   'resources' : perturbed_conditions[community.no_species:]})
    
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
    original_traj = community.simulation(T,
                                         original_conditions,
                                         assign = False)
    # Simulate the perturbated community trajectory for time = T
    perturbed_traj = community.simulation(T,
                                          perturbed_conditions,
                                          assign = False)
    
    return original_traj, perturbed_traj

#########

def calculate_max_le(original_traj : npt.NDArray,
                     perturbed_traj : npt.NDArray,
                     separation : float,
                     separation_grad_abs : npt.NDArray):
    
    if len(original_traj.t) < 200 or len(perturbed_traj.t) < 200:
        
        max_lyapunov_exponent, log_offset = np.polyfit(perturbed_traj.t,
                                                       separation,
                                                       1)
    else:
    
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

# %%


def max_le_2(community : Union["SL_CRM", "SL_SI_CRM", "ES_CRM"],
             T : float,
             initial_conditions : npt.NDArray,
             separation : float = 1e-9,
             dt : float = 1):
    
    '''
    
    Calculate the average maximum lyapunov exponent for a community.
    See Sprott (1997, revised 2015) 'Numerical Calculation of Largest Lyapunov Exponent' 
    for more details.
    
    Protocol:
        (1) Extract initial species abundances from a simulation of lineage dynamics.
        (2) Simulate community dynamics from aforementioned initial abundances for time = dt.
        (3) Select an extant species, and perturbate its initial species abundance by separation.
            Simulate community dynamics for time = dt.
        (4) Measure the new degree of separation between the original trajectory and the
            perturbated trajectory. This is d1:
                d1 = [(N_1 - N_(1,perturbated))^2 + (N_2 - N_(2,perturbated))^2 + ...]
        (5) Estimate the max. lyapunov exponent = (1/dt)*ln(|d1/separation|).
        (6) Reset the perturbated trajectories species abundaces so that the 
            original and perturbated trajectory are 'separation' apart:
                x_normalised = x_end + (separation/d1)*(x_(perturbated,end)-x_end).
        (7) Repeat steps 2, 4-6 n times, then calculate the average max. lyapunov exponent.
    
    Parameters
    ----------
    community : object of the Consumer_Resource_Model class
        community, for which the max. le is being calculated
    T : float
        the max. time the max. lyapunov exponent is calculated over
        No. iteractions = T/dt
    initial_conditions : np.ndarray
        the initial abundances/community composition to start calculating the max. le from
    separation : float
        the amount a community is perturbated. The default is 1e-9.
    dt : float, optional
    
    Returns
    -------
    max_lyapunov_exponent : float
        The average maximum lyapunov exponent calculate over T.
    
    '''
    
    # Initialise list of max. lyapunov exponents
    log_d1d0 = []
    
    # Set initial conditions as population abundances at the end of lineage simulations
    
    # Set initial conditions of the original and perturbated trajectory
    original_conditions = deepcopy(initial_conditions)
    
    perturbed_conditions = deepcopy(initial_conditions)
    perturbed_conditions += separation/len(perturbed_conditions)
    
    current_time = 0
    
    # set max. and min. amounts trajectories can separate before the function stops
    separation_dt = separation
    separation_min = 1e-3 * separation
    separation_max = 1e3 * separation
    
    for n in range(int(np.round(T/dt))):
        
        if separation_dt > separation_min and separation_dt < separation_max:
            
            # Simulate the original community trajectory for time = dt
            simulation1 = community.simulate_community(dt, 1, init_cond_func='user supplied',
                                                         assign = False,
                                                         user_supplied_init_cond = 
                                                         {'species' : original_conditions[:community.no_species],
                                                          'resources' : original_conditions[community.no_species:]})
            # Simulate the perturbated community trajectory for time = dt
            simulation2 = community.simulate_community(dt, 1, init_cond_func='user supplied',
                                                         assign = False,
                                                         user_supplied_init_cond = 
                                                         {'species' : perturbed_conditions[:community.no_species],
                                                          'resources' : perturbed_conditions[community.no_species:]})
            
            # Get species abundances at the end of simulation from the original and perturbed trajectories
            final_dynamics1 = simulation1[0].y[:,-1]
            final_dynamics2 = simulation2[0].y[:,-1]
            
            # Calculated the new separation between the original and perturbated trajectory (d1)
            separation_dt = np.sqrt(np.sum((final_dynamics1 - final_dynamics2)**2))
            
            # Calculate the max. lyapunov exponent
            log_d1d0.append(np.log(separation_dt/(separation)))
            
            # Reset the original trajectory's species abundances to the species abundances at dt.
            original_conditions = final_dynamics1
             
            # Reset the perturbated trajectory's species abundances so that the original
            #    and perturbated community are 'separation' apart.
            perturbed_conditions = final_dynamics1 + \
                (final_dynamics2 - final_dynamics1)*(separation/separation_dt)
                
            current_time += dt
            
        else:
            
            break
        
    # Calculate average max. lyapunov exponent
    max_lyapunov_exponent = (1/(current_time)) * np.sum(np.array(log_d1d0))
    
    return max_lyapunov_exponent