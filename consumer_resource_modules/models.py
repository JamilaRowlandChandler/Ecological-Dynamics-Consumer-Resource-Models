# -*- coding: utf-8 -*-
"""
Created on Sun May  4 13:01:05 2025

@author: jamil
"""

import numpy as np
import numpy.typing as npt
from typing import Literal, Union, TypedDict
from scipy.integrate import solve_ivp

from parameters import ParametersInterface
from differential_equations import DifferentialEquationsInterface, unbounded_growth #, ReloadedODEs
from community_level_properties import CommunityPropertiesInterface

# %%

def Consumer_Resource_Model(model : Literal["Self-limiting resource supply",
                                            "Self-limiting resource supply, leached",
                                            "Self-limiting resource supply, self-inhibition",
                                            "Self-limiting resource supply, multi-trophic level"
                                            "Externally-supplied resources"],
                            no_species : Union[int, None] = None,
                            no_resources : Union[int, None] = None,
                            trophic_levels : Union[int, None] = None,
                            pool_sizes : Union[int, None] = None):
    
    '''
    
    Wrapper for different consumer resource model classes

    Parameters
    ----------
    model : str
        Type of consumer resource model. Options are:
            "Self-limiting resource supply" - resources grow logistically
            "Self-limiting resource supply, self-inhibition" - same as 
            "Self-limiting resource supply", but with direct consumer self-inhibition
            "Externally-supplied resources" - chemostat-style resource dynamics
            (constant influx + dilution)
    no_species : int
        species pool size
    no_resources : int
        resource pool size

    Raises
    ------
    Exception
        If a non-existent model is selected.

    Returns
    -------
    instance : object of some consumer-resource model class
        Instance of some consumer-resource model class.

    '''
    
    match model:
        
        case "Self-limiting resource supply":
            
            instance = SL_CRM(no_species, no_resources)
            
        #case "Self-limiting resource supply, leached":
            
        #    instance = SL_CRPM(no_species, no_resources)
            
        case "Self-limiting resource supply, self-inhibition":
            
            instance = SL_SI_CRM(no_species, no_resources)
            
        case "Self-limiting resource supply, multi-trophic level":
            
            instance = SL_TL_CRM(trophic_levels, pool_sizes)
            
        case "Externally-supplied resources":
            
            instance = ES_CRM(no_species, no_resources)
            
        case "Hybrid resource supply":
            
            instance = Hybrid_CRM(no_species, no_resources)
            
        case "MiCRM":
            
            instance = MiCRM(no_species, no_resources)
            
        case "Leached biomolecules":
            
            instance = LB_CRM(no_species, no_resources)
            
        case _:
            
            raise Exception('You have not selected an exisiting model.\n' + \
                  'Please chose from either "Self-limiting resource supply"' + \
                      '"Self-limiting resource supply, self-inhibition"' + \
                      ' or "Externally-supplied resources"')
    return instance

# %%

class SL_CRM(ParametersInterface,
             DifferentialEquationsInterface,
             #ReloadedODEs,
             CommunityPropertiesInterface):
    
    '''
    
    Consumer-resource model (CRM) class with self-limiting resource supply
    
    '''
    
    def __init__(self, no_species : int, no_resources : int):
        
        '''
        
        Initiate model
        
        Parameters
        ----------
        no_species : int
            species pool size
        no_resources : int
            resource pool size

        Returns
        -------
        None.

        '''
        
        # assign species and resource pool size as class attributes
        self.no_species = no_species
        self.no_resources = no_resources
        
    def model_specific_rates(self,
                             death_method : 
                                 Literal['normal', 'constant', 'user-supplied'] 
                                 = 'constant',
                             death_args : Union[TypedDict('normal', {'mu' : float, 'sigma' : float}),
                                                TypedDict('constant', {'d' : float}),
                                                TypedDict('user-supplied', {'d' : npt.NDArray})]
                             = {'d' : 1},
                             resource_growth_method : 
                                 Literal['normal', 'constant', 'user-supplied'] 
                                 = 'constant',
                             resource_growth_args : 
                                 Union[TypedDict('normal', {'mu' : float, 'sigma' : float}),
                                       TypedDict('constant', {'b' : float}),
                                       TypedDict('user-supplied', {'b' : npt.NDArray})]
                                 = {'b' : 1}):
        
        '''
        
        Generate parameters specific to the CRM with self-limiting resource 
        dynamics - consumer death rates and intrinsic resource growth rates


        Parameters
        ----------
        death_method : str
            Method used to generate death rates. Options are:
                'normal' : normally distributed parameters
                'constant' : death rates are fixed
                'user-supplied' : supply your own death rates
        death_args : dict
            Arguments for death_method.
            If 'normal', first argument is the mean, second is the stand deviation
            e.g., {'mu': mean, 'sigma' : standard deviation}
            If 'constant', the key is the parameter name, argument is the fixed value
            e.g., {'d' : val}
            If 'used-supplied', argument is the array of death rates 
            e.g., {'d' : array_of_vals}
        resource_growth_method : str
            Method used to generate intrinsic resource growth rates. Options are
            the same as death_method, but named 'b' rather than 'd'
        resource_growth_args : dict
            Arguments for resource_growth_method. Options are the same as 
            death_method args.

        Returns
        -------
        None.

        '''
        
        # labels used to assign parameters as object attributes
        p_labels = ['d', 'b']
        
        # dimensions for death rates and intrinsic growth rates
        dims_list = [(self.no_species, ), (self.no_resources, )]
        
        # generate parameters used the other_parameter_methods method
        for p_method, p_args, p_label, dims in \
            zip([death_method, resource_growth_method],
                [death_args, resource_growth_args],
                p_labels, dims_list):
                
                self.other_parameter_methods(p_method, p_args, p_label, dims)
                
    def collate_parameters(self):
        
        return (self.no_species, self.growth, self.consumption,
                self.d, self.b,
                getattr(self, 'timescalar_r', 1),
                getattr(self, 'timescalar_s', 1),)
    
    #####################################################################
    
    def simulation(self,
                   t_end : float,
                   initial_abundance : npt.NDArray):
        
        '''
        
        Simulate community dynamics
        
        Parameters
        ----------
        t_end : float
            Simulation end time.
        initial_abundance : np.ndarray
            Initial abundances of species and resources.

        Returns
        -------
        Bunch object produced by scipy.integrate.solve_ivp
            Simulation.

        '''
        
        # call the ODE solver with the unbounded growth event function
        # the ode solver stops when the event function is true (returns 0)
        return solve_ivp(self.model, [0, t_end], initial_abundance,
                         args = (self.no_species, self.growth, self.consumption,
                                 self.d, self.b,
                                 getattr(self, "timescalar_r", 1),
                                 getattr(self, "timescalar_s", 1)),
                         method = 'LSODA',
                         rtol = 1e-7,
                        # atol= 1e-9,
                         atol = np.concatenate([np.full(self.no_species, 10.0**-15),
                                                np.full(self.no_resources, 10.0**-15)]),
                         first_step=np.min([getattr(self, "timescalar_r", 1e-4),
                                            getattr(self, "timescalar_s", 1e-4),
                                            1e-4]),
                         t_eval = np.linspace(0, t_end, 200),
                         jac = self.jacobian,
                         events = unbounded_growth)
    
    
    def model(self,
              t, y,
              S, G, C, D, B,
              e_r, e_s):
        
        '''
        
        ODE for CRM with self-limiting resource supply

        Parameters
        ----------
        t : float
            time
        y : np.ndarray
            consumer and resource abundances at time t
        S : int
            species pool size (used to separate y into species and resource 
                               abundances)
        G : np.ndarray
            matrix of consumer growth rates
        C : np.ndarray
            matrix of resource consumption rates
        D : np.ndarray
            consumer death rates
        B : np.ndarray
            intrinsic resource growth rates

        Returns
        -------
        np.ndarray
            Rate of change in species and resource abundances over time 
            (dNdt and dRdt)

        '''
        
        # extinction threshold
        #y[y < 1e-5] = 0
        
        # separate species and resource abundances
        species, resources = y[:S], y[S:]
        
        # change in consumer abundances over time
        dNdt = (1.0/e_s) * (species * (np.sum(G * resources, axis = 1) - D))
        
        # change in resource abundances over time
        dRdt = (1.0/e_r) * ((resources * (B - resources)) - \
                            (resources * np.sum(C * species, axis=1)))
            
        if e_r or e_s > 1e-6: immigration = 1e-8 
        else: immigration = 1e-12
            
        return np.concatenate((dNdt, dRdt)) + immigration

    def jacobian(self,
                 t, y,
                 S, G, C, D, B,
                 e_r, e_s):

        species, resources = y[:S], y[S:]

        growth_term = np.sum(G * resources, axis = 1) - D
        consumption_term = np.sum(C * species, axis = 1)

        J = np.zeros((y.size, y.size))

        J[:S, :S] = (1.0 / e_s) * (np.diag(growth_term))
        J[:S, S:] = (1.0 / e_s) * (G * species[:, np.newaxis])
        J[S:, :S] = (1.0 / e_r) * (-C * resources[:, np.newaxis])
        J[S:, S:] = (1.0 / e_r) * (np.diag(B - 2*resources - consumption_term))
        
        return J
    
# %%

class SL_SI_CRM(ParametersInterface, DifferentialEquationsInterface,
                CommunityPropertiesInterface):
    
    '''
    
    Consumer-resource model (CRM) class with self-limiting resource supply and
    direct consumer self-inhibition
    
    '''
    
    def __init__(self, no_species : int, no_resources : int):
        
        '''
        
        Initiate model
        
        Parameters
        ----------
        no_species : int
            species pool size
        no_resources : int
            resource pool size

        Returns
        -------
        None.

        '''
        
        self.no_species = no_species
        self.no_resources = no_resources
        
    def model_specific_rates(self,
                             death_method : 
                                 Literal['normal', 'constant', 'user-supplied'] 
                                 = 'constant',
                             death_args : Union[TypedDict('normal', {'mu' : float, 'sigma' : float}),
                                                TypedDict('constant', {'d' : float}),
                                                TypedDict('user-supplied', {'d' : npt.NDArray})]
                             = {'d' : 1},
                             resource_growth_method : 
                                 Literal['normal', 'constant', 'user-supplied'] 
                                 = 'constant',
                             resource_growth_args : 
                                 Union[TypedDict('normal', {'mu' : float, 'sigma' : float}),
                                       TypedDict('constant', {'b' : float}),
                                       TypedDict('user-supplied', {'b' : npt.NDArray})]
                                 = {'b' : 1},
                             si_method : 
                                 Literal['normal', 'constant', 'user-supplied'] = 'constant',
                             si_args : 
                                 Union[TypedDict('normal', {'mu' : float, 'sigma' : float}),
                                       TypedDict('constant', {'si' : float}),
                                       TypedDict('user-supplied', {'si' : npt.NDArray})]
                                 = {'si' : 1}):
        
        '''
        
        Generate parameters specific to the CRM with self-limiting resource 
        dynamics - consumer death rates and intrinsic resource growth rates


        Parameters
        ----------
        death_method : str
            Method used to generate death rates. Options are:
                'normal' : normally distributed parameters
                'constant' : death rates are fixed
                'user-supplied' : supply your own death rates
        death_args :  Arguments for death_method.
             If 'normal', first argument is the mean, second is the stand deviation
             e.g., {'mu': mean, 'sigma' : mean}
             If 'constant', the key is the parameter name, argument is the fixed value
             e.g., {'d' : val}
             If 'used-supplied', argument is the array of death rates 
             e.g., {'d' : array_of_vals}
        resource_growth_method : str
            Method used to generate intrinsic resource growth rates. Options are
            the same as death_method, but named 'b' rather than 'd'
        resource_growth_args : dict
            Arguments for resource_growth_method. Options are the same as 
            death_method args.
        si_method : str
            Method used to generate direct self-interaction coefficients between consumers.
            Options are the same as death_method
        si_args : dict
            Arguments for si_method. Options are the same as death_method args,
            but named 'si' rather than 'd'

        Returns
        -------
        None.

        '''
        
        # labels used to assign parameters as object attributes
        p_labels = ['d', 'b', 'si']
        
        # dimensions for death rates, intrinsic growth rates, and self-interaction coeffients
        dims_list = [(self.no_species, ), (self.no_resources, ),
                     (self.no_species, )]
        
        # generate parameters used the other_parameter_methods method
        for p_method, p_args, p_label, dims in \
            zip([death_method, resource_growth_method, si_method],
                [death_args, resource_growth_args, si_args],
                p_labels, dims_list):
                
                self.other_parameter_methods(p_method, p_args, p_label, dims)
    
    #####################################################################
    
    def collate_parameters(self):
        
        return (self.no_species, self.growth, self.consumption, 
                self.d, self.b, self.si)
    
    def simulation(self,
                   t_end : float,
                   initial_abundance : npt.NDArray):
        
        '''
        
        Simulate community dynamics
        
        Parameters
        ----------
        t_end : float
            Simulation end time.
        initial_abundance : np.ndarray
            Initial abundances of species and resources.

        Returns
        -------
        Bunch object produced by scipy.integrate.solve_ivp
            Simulation.

        '''
        
        unbounded_growth.terminal = True
        
        # call the ODE solver with the unbounded growth event function
        # the ode solver stops when the event function is true (returns 0)           
        return solve_ivp(self.model, [0, t_end], initial_abundance, 
                         args = (self.no_species, self.growth, self.consumption, 
                                 self.d, self.b, self.si),
                         method = 'LSODA', rtol = 1e-7, atol = 1e-9,
                         t_eval = np.linspace(0, t_end, 200), events = unbounded_growth)
    
    def model(self,
              t, y,
              S, G, C, D, B, SI):
        
        '''
        
        ODE for CRM with self-limiting resource supply

        Parameters
        ----------
        t : float
            time
        y : np.ndarray
            consumer and resource abundances at time t
        S : int
            species pool size (used to separate y into species and resource 
                               abundances)
        G : np.ndarray
            matrix of consumer growth rates
        C : np.ndarray
            matrix of resource consumption rates
        D : np.ndarray
            consumer death rates
        B : np.ndarray
            intrinsic resource growth rates
        SI : np.ndarray
            consumer self-interaction coefficients 

        Returns
        -------
        np.ndarray
            Rate of change in species and resource abundances over time 
            (dNdt and dRdt)

        '''
        
        # separate species and resource abundances
        species, resources = y[:S], y[S:]
        
        # change in consumer abundances over time
        dNdt = species * ((np.sum(G * resources, axis = 1) - D) - SI*species)
       
        # change in resource abundances over time
        dRdt = (resources * (B - resources)) - \
            (resources * np.sum(C * species, axis=1))

        return np.concatenate((dNdt, dRdt)) + 1e-8
    
# %%

class SL_TL_CRM(ParametersInterface,
                DifferentialEquationsInterface,
                CommunityPropertiesInterface):
    
    '''
    
    Consumer-resource model (CRM) class with self-limiting resource supply
    
    '''
    
    def __init__(self,
                 trophic_levels : int,
                 pool_sizes = Union[tuple[int], list[int], npt.NDArray]):
        
        '''
        
        Initiate model
        
        Parameters
        ----------
        no_species : int
            species pool size
        no_resources : int
            resource pool size

        Returns
        -------
        None.

        '''
        
        # assign species and resource pool size as class attributes
        
        self.trophic_levels = trophic_levels
        
        self.pool_sizes = pool_sizes
        
    def model_specific_rates(self,
                             death_methods : 
                                 list[Literal['normal', 'constant', 'user-supplied']],
                             death_args : list[Union[TypedDict('normal', {'mu' : float, 'sigma' : float}),
                                                TypedDict('constant', {'d' : float}),
                                                TypedDict('user-supplied', {'d' : npt.NDArray})]],
                             resource_growth_method : 
                                 Literal['normal', 'constant', 'user-supplied'] 
                                 = 'constant',
                             resource_growth_args : 
                                 Union[TypedDict('normal', {'mu' : float, 'sigma' : float}),
                                       TypedDict('constant', {'b' : float}),
                                       TypedDict('user-supplied', {'b' : npt.NDArray})]
                                 = {'b' : 1},
                            resource_interaction_method : 
                                Literal['normal', 'constant', 'user-supplied'] 
                                = 'constant',
                            resource_interaction_args : 
                                Union[TypedDict('normal', {'mu' : float, 'sigma' : float}),
                                      TypedDict('constant', {'Aij' : float}),
                                      TypedDict('user-supplied', {'Aij' : npt.NDArray})]
                                = {'Aij' : 0}):
        
        '''
        
        Generate parameters specific to the CRM with self-limiting resource 
        dynamics - consumer death rates and intrinsic resource growth rates


        Parameters
        ----------
        death_method : str
            Method used to generate death rates. Options are:
                'normal' : normally distributed parameters
                'constant' : death rates are fixed
                'user-supplied' : supply your own death rates
        death_args : dict
            Arguments for death_method.
            If 'normal', first argument is the mean, second is the stand deviation
            e.g., {'mu': mean, 'sigma' : standard deviation}
            If 'constant', the key is the parameter name, argument is the fixed value
            e.g., {'d' : val}
            If 'used-supplied', argument is the array of death rates 
            e.g., {'d' : array_of_vals}
        resource_growth_method : str
            Method used to generate intrinsic resource growth rates. Options are
            the same as death_method, but named 'b' rather than 'd'
        resource_growth_args : dict
            Arguments for resource_growth_method. Options are the same as 
            death_method args.

        Returns
        -------
        None.

        '''
        
        # labels used to assign parameters as object attributes
        p_labels = ['d_' + str(tl) 
                    for tl in np.arange(2, self.trophic_levels + 1)] + \
                    ['b', 'Aij']
        
        # dimensions for death rates and intrinsic growth rates
        dims_list = [(pool_size, ) for pool_size in self.pool_sizes] + \
                        [(self.pool_sizes[0], self.pool_sizes[0])]
        
        methods_list = death_methods + [resource_growth_method,
                                        resource_interaction_method]
        
        args_list = death_args + [resource_growth_args,
                                  resource_interaction_args]
        
        # generate parameters used the other_parameter_methods method
        for p_method, p_args, p_label, dims in \
            zip(methods_list, args_list,
                p_labels, dims_list):
                
                self.other_parameter_methods(p_method, p_args, p_label, dims)
                
        np.fill_diagonal(self.Aij, 1)
        
    def collate_parameters(self):
        
        return (np.append(0, np.cumsum(self.pool_sizes)),
                [getattr(self, "growth_" + str(i)) 
                 for i in np.arange(2, self.trophic_levels + 1)],
                [getattr(self, "consumption_" + str(i)) 
                 for i in np.arange(2, self.trophic_levels + 1)],
                [getattr(self, "d_" + str(i)) 
                 for i in np.arange(2, self.trophic_levels + 1)],
                self.b, self.Aij)
    
    #####################################################################
    
    def simulation(self,
                   t_end : float,
                   initial_abundance : npt.NDArray):
        
        '''
        
        Simulate community dynamics
        
        Parameters
        ----------
        t_end : float
            Simulation end time.
        initial_abundance : np.ndarray
            Initial abundances of species and resources.

        Returns
        -------
        Bunch object produced by scipy.integrate.solve_ivp
            Simulation.

        '''
        
        unbounded_growth.terminal = True
        
        # call the ODE solver with the unbounded growth event function
        # the ode solver stops when the event function is true (returns 0)           
        return solve_ivp(self.model, [0, t_end], initial_abundance, 
                         args = (np.append(0, np.cumsum(self.pool_sizes)),
                                 [getattr(self, "growth_" + str(i)) 
                                  for i in np.arange(2, self.trophic_levels + 1)],
                                 [getattr(self, "consumption_" + str(i)) 
                                  for i in np.arange(2, self.trophic_levels + 1)],
                                 [getattr(self, "d_" + str(i)) 
                                  for i in np.arange(2, self.trophic_levels + 1)],
                                 self.b, self.Aij),
                         method = 'LSODA', rtol = 1e-7, atol = 1e-9,
                         t_eval = np.linspace(0, t_end, 200), events = unbounded_growth)
    
    def model(self,
              t, N,
              pool_idx, Gs, Cs, Ds, B, A):
        
        '''
        
        ODE for CRM with self-limiting resource supply

        Parameters
        ----------
        t : float
            time
        N : np.ndarray
            consumer and resource abundances at time t
        P : int
            predator pool size (used to separate y into predator, species and resource 
                               abundances)
        S : int
            species pool size (used to separate y into predator, species and resource 
                               abundances)
        G : np.ndarray
            matrix of consumer growth rates
        C : np.ndarray
            matrix of resource consumption rates
        D : np.ndarray
            consumer death rates
        B : np.ndarray
            intrinsic resource growth rates

        Returns
        -------
        np.ndarray
            Rate of change in species and resource abundances over time 
            (dNdt and dRdt)

        '''
        
        def toplevel_dynamics(N_i, N_iminus1, 
                              G_i, D_i):
            
            dNdt = N_i * (np.sum(G_i * N_iminus1, axis = 1) - D_i)
            
            return dNdt
        
        def middlelevel_dynamics(N_i, N_iminus1, N_iadd1,
                                 G_i, D_i, C_i):
            
            dNdt = N_i * (np.sum(G_i * N_iminus1, axis = 1) - D_i) - \
                (N_i * np.sum(C_i * N_iadd1, axis=1))
            
            return dNdt
        
        def bottomlevel_dynamics(N_i, N_iadd1,
                                 B, A, C_i):
            
            dNdt = (N_i * (B - A @ N_i)) - (N_i * np.sum(C_i * N_iadd1, axis=1))
                
            return dNdt
        
        # change in resource abundances over time
        dRdt = bottomlevel_dynamics(N[:pool_idx[1]],
                                    N[pool_idx[1] : pool_idx[2]],
                                    B, A, Cs[0])
        
        # change in consumer abundances over time
        dNdt = np.array([middlelevel_dynamics(N[pool_idx[i] : pool_idx[i+1]],
                                              N[pool_idx[i-1] : pool_idx[i]],
                                              N[pool_idx[i+1] : pool_idx[i+2]],
                                              Gs[i-1], Ds[i-1], Cs[i])
                         for i in np.arange(1, len(pool_idx[1:-1]))])
        
        # change predators abundances over time
        dPdt = toplevel_dynamics(N[pool_idx[-2]:],
                                 N[pool_idx[-3] : pool_idx[-2]],
                                 Gs[-1],
                                 Ds[-1])
        
        return np.concatenate((dRdt, dNdt.flatten(), dPdt)) + 1e-8
    
# %%

class ES_CRM(ParametersInterface, DifferentialEquationsInterface,
             CommunityPropertiesInterface):
    
    def __init__(self, no_species, no_resources):
        
        self.no_species = no_species
        self.no_resources = no_resources
    
    def model_specific_rates(self, 
                             death_method : 
                                 Literal['normal', 'constant', 'user-supplied'] 
                                 = 'constant',
                             death_args : Union[TypedDict('normal', {'mu' : float, 'sigma' : float}),
                                                TypedDict('constant', {'d' : float}),
                                                TypedDict('user-supplied', {'d' : npt.NDArray})]
                             = {'d' : 1},
                             influx_method: 
                                 Literal['normal', 'constant', 'user-supplied'] 
                                 = 'constant',
                             influx_args : Union[TypedDict('normal', {'mu' : float, 'sigma' : float}),
                                                TypedDict('constant', {'b' : float}),
                                                TypedDict('user-supplied', {'b' : npt.NDArray})]
                             = {'b' : 1},
                             outflux_method: 
                                 Literal['normal', 'constant', 'user-supplied'] 
                                 = 'constant',
                             outflux_args : Union[TypedDict('normal', {'mu' : float, 'sigma' : float}),
                                                TypedDict('constant', {'o' : float}),
                                                TypedDict('user-supplied', {'o' : npt.NDArray})]
                             = {'o' : 1}):
        
        '''
        
        Generate parameters specific to the CRM with self-limiting resource 
        dynamics - consumer death rates and intrinsic resource growth rates


        Parameters
        ----------
        death_method : str
            Method used to generate death rates. Options are:
                'normal' : normally distributed parameters
                'constant' : death rates are fixed
                'user-supplied' : supply your own death rates
        death_args : dict
            Arguments for death_method.
            If 'normal', first argument is the mean, second is the stand deviation
            e.g., {'mu': mean, 'sigma' : standard deviation}
            If 'constant', the key is the parameter name, argument is the fixed value
            e.g., {'d' : val}
            If 'used-supplied', argument is the array of death rates 
            e.g., {'d' : array_of_vals}
        influx_method : str
            Method used to generate intrinsic resource influx rates. Options are
            the same as death_method, but named 'b' rather than 'd'
        influx_args : dict
            Arguments for influx_method. Options are the same as 
            death_method args.
        outflux_method : str
            Method used to generate intrinsic resource outflux rates. Options are
            the same as death_method, but named 'o' rather than 'd'.
        outflux_args : dict
            Arguments for outflux_method. Options are the same as 
            death_method args.
        Returns
        -------
        None.

        '''
        
        # labels used to assign parameters as object attributes
        p_labels = ['d', 'b', 'o']
        
        # dimensions for death rates and intrinsic growth rates
        dims_list = [(self.no_species, ), (self.no_resources, ),
                     (self.no_resources, )]
        
        # generate parameters used the other_parameter_methods method
        for p_method, p_args, p_label, dims in \
            zip([death_method, influx_method, outflux_method],
                [death_args, influx_args, outflux_args],
                p_labels, dims_list):
                
                self.other_parameter_methods(p_method, p_args, p_label, dims)
                
    def collate_parameters(self):
        
        return (self.no_species, self.growth, self.consumption,
                self.d, self.b, self.o, getattr(self, "timescalar", 1))
        
    #########################################################
                
    def simulation(self,
                   t_end : float,
                   initial_abundance : npt.NDArray):
        
        '''
        
        Simulate community dynamics
        
        Parameters
        ----------
        t_end : float
            Simulation end time.
        initial_abundance : np.ndarray
            Initial abundances of species and resources.

        Returns
        -------
        Bunch object produced by scipy.integrate.solve_ivp
            Simulation.

        '''
        
        unbounded_growth.terminal = True

        # call the ODE solver with the unbounded growth event function
        # the ode solver stops when the event function is true (returns 0)
        return solve_ivp(self.model, [0, t_end], initial_abundance,
                         args = (self.no_species, self.growth, self.consumption,
                                 self.d, self.b, self.o,
                                 getattr(self, "timescalar", 1)),
                         method = 'LSODA',
                         rtol = 1e-7,
                         atol = np.concatenate([np.full(self.no_species, 10.0**-15),
                                                np.full(self.no_resources, 10.0**-15)]),
                         t_eval = np.linspace(0, t_end, 200),
                         jac = self.jacobian,
                         events = unbounded_growth)
    
    def model(self,
              t, y,
              S, G, C, D, B, O,
              e):
        
        '''
        
        ODE for CRM with self-limiting resource supply

        Parameters
        ----------
        t : float
            time
        y : np.ndarray
            consumer and resource abundances at time t
        S : int
            species pool size (used to separate y into species and resource 
                               abundances)
        G : np.ndarray
            matrix of consumer growth rates
        C : np.ndarray
            matrix of resource consumption rates
        D : np.ndarray
            consumer death rates
        B : np.ndarray
            intrinsic resource growth rates

        Returns
        -------
        np.ndarray
            Rate of change in species and resource abundances over time 
            (dNdt and dRdt)

        '''
        
        # extinction threshold
        #y[y < 1e-5] = 0
        
        # separate species and resource abundances
        species, resources = y[:S], y[S:]
        
        # change in consumer abundances over time
        dNdt = species * (np.sum(G * resources, axis = 1) - D)
        
        # change in resource abundances over time
        dRdt = (1.0/e) * ((B - O * resources) - \
                          (resources * np.sum(C * species, axis=1)))
            
        if e > 1e-6: immigration = 1e-8 
        else: immigration = 1e-12
            
        return np.concatenate((dNdt, dRdt)) + immigration

    def jacobian(self,
                 t, y,
                 S, G, C, D, B, O,
                 e):
        
        species, resources = y[:S], y[S:]

        growth_term = np.sum(G * resources, axis = 1) - D
        consumption_term = np.sum(C * species, axis = 1)

        J = np.zeros((y.size, y.size))

        J[:S, :S] = np.diag(growth_term)
        J[:S, S:] = G * species[:, np.newaxis]
        J[S:, :S] = (1.0 / e) * (-C * resources[:, np.newaxis])
        J[S:, S:] = (1.0 / e) * (np.diag(-O - consumption_term))

        return J

# %%

class Hybrid_CRM(ParametersInterface, DifferentialEquationsInterface,
                 CommunityPropertiesInterface):
    
    def __init__(self, no_species, no_resources):
        
        self.no_species = no_species
        self.no_resources = no_resources
    
    def model_specific_rates(self, 
                             death_method : 
                                 Literal['normal', 'constant', 'user-supplied'] 
                                 = 'constant',
                             death_args : Union[TypedDict('normal', {'mu' : float, 'sigma' : float}),
                                                TypedDict('constant', {'d' : float}),
                                                TypedDict('user-supplied', {'d' : npt.NDArray})]
                             = {'d' : 1},
                             influx_method: 
                                 Literal['normal', 'constant', 'user-supplied'] 
                                 = 'constant',
                             influx_args : Union[TypedDict('normal', {'mu' : float, 'sigma' : float}),
                                                TypedDict('constant', {'b' : float}),
                                                TypedDict('user-supplied', {'b' : npt.NDArray})]
                             = {'b' : 1},
                             outflux_method: 
                                 Literal['normal', 'constant', 'user-supplied'] 
                                 = 'constant',
                             outflux_args : Union[TypedDict('normal', {'mu' : float, 'sigma' : float}),
                                                TypedDict('constant', {'o' : float}),
                                                TypedDict('user-supplied', {'o' : npt.NDArray})]
                             = {'o' : 1},
                             resource_inhibition_method: 
                                 Literal['normal', 'constant', 'user-supplied'] 
                                 = 'constant',
                             resource_inhibition_args : Union[TypedDict('normal', {'mu' : float, 'sigma' : float}),
                                                              TypedDict('constant', {'a' : float}),
                                                              TypedDict('user-supplied', {'a' : npt.NDArray})]
                             = {'a' : 1}):
        
        '''
        
        Generate parameters specific to the CRM with self-limiting resource 
        dynamics - consumer death rates and intrinsic resource growth rates


        Parameters
        ----------
        death_method : str
            Method used to generate death rates. Options are:
                'normal' : normally distributed parameters
                'constant' : death rates are fixed
                'user-supplied' : supply your own death rates
        death_args : dict
            Arguments for death_method.
            If 'normal', first argument is the mean, second is the stand deviation
            e.g., {'mu': mean, 'sigma' : standard deviation}
            If 'constant', the key is the parameter name, argument is the fixed value
            e.g., {'d' : val}
            If 'used-supplied', argument is the array of death rates 
            e.g., {'d' : array_of_vals}
        influx_method : str
            Method used to generate intrinsic resource influx rates. Options are
            the same as death_method, but named 'b' rather than 'd'
        influx_args : dict
            Arguments for influx_method. Options are the same as 
            death_method args.
        outflux_method : str
            Method used to generate intrinsic resource outflux rates. Options are
            the same as death_method, but named 'o' rather than 'd'.
        outflux_args : dict
            Arguments for outflux_method. Options are the same as 
            death_method args.
        Returns
        -------
        None.

        '''
        
        # labels used to assign parameters as object attributes
        p_labels = ['d', 'b', 'o', 'a']
        
        # dimensions for death rates and intrinsic growth rates
        dims_list = [(self.no_species, ), (self.no_resources, ),
                     (self.no_resources, ), (self.no_resources, )]
        
        # generate parameters used the other_parameter_methods method
        for p_method, p_args, p_label, dims in \
            zip([death_method, influx_method, outflux_method, resource_inhibition_method],
                [death_args, influx_args, outflux_args, resource_inhibition_args],
                p_labels, dims_list):
                
                self.other_parameter_methods(p_method, p_args, p_label, dims)
                
    def collate_parameters(self):
        
        return (self.no_species, self.growth, self.consumption, 
                self.d, self.b, self.o, self.a)
        
    ##################################################################
                
    def simulation(self,
                   t_end : float,
                   initial_abundance : npt.NDArray):
        
        '''
        
        Simulate community dynamics
        
        Parameters
        ----------
        t_end : float
            Simulation end time.
        initial_abundance : np.ndarray
            Initial abundances of species and resources.

        Returns
        -------
        Bunch object produced by scipy.integrate.solve_ivp
            Simulation.

        '''
        
        unbounded_growth.terminal = True
        
        # call the ODE solver with the unbounded growth event function
        # the ode solver stops when the event function is true (returns 0)           
        return solve_ivp(self.model, [0, t_end], initial_abundance, 
                         args = (self.no_species, self.growth, self.consumption, 
                                 self.d, self.b, self.o, self.a),
                         method = 'LSODA', # 'Radau', #'RK45',
                         rtol = 1e-9, atol = 1e-7,
                         t_eval = np.linspace(0, t_end, 200), events = unbounded_growth)
    
    def model(self,
              t, y,
              S, G, C, D, B, O, A):
        
        '''
        
        ODE for CRM with self-limiting resource supply

        Parameters
        ----------
        t : float
            time
        y : np.ndarray
            consumer and resource abundances at time t
        S : int
            species pool size (used to separate y into species and resource 
                               abundances)
        G : np.ndarray
            matrix of consumer growth rates
        C : np.ndarray
            matrix of resource consumption rates
        D : np.ndarray
            consumer death rates
        B : np.ndarray
            intrinsic resource growth rates

        Returns
        -------
        np.ndarray
            Rate of change in species and resource abundances over time 
            (dNdt and dRdt)

        '''
        
        # extinction threshold
        #y[y < 1e-5] = 0
        
        # separate species and resource abundances
        species, resources = y[:S], y[S:]
        
        # change in consumer abundances over time
        dNdt = species * (np.sum(G * resources, axis = 1) - D)
        
        # change in resource abundances over time
        dRdt = (B + O * resources - A * resources**2) - \
            (resources * np.sum(C * species, axis=1))
            
        return np.concatenate((dNdt, dRdt)) + 1e-8
    



# %%

class MiCRM(ParametersInterface, DifferentialEquationsInterface,
            CommunityPropertiesInterface):

    '''

    Microbial consumer-resource model (MiCRM) with cross-feeding. Consumers
    leak a fraction of consumed resources, which are converted into other
    resources by a metabolic matrix (see Marsland et al., 2020).

    '''

    def __init__(self, no_species, no_resources):

        self.no_species = no_species
        self.no_resources = no_resources

    def model_specific_rates(self,
                             death_method :
                                 Literal['normal', 'constant', 'user-supplied']
                                 = 'constant',
                             death_args : Union[TypedDict('normal', {'mu' : float, 'sigma' : float}),
                                                TypedDict('constant', {'d' : float}),
                                                TypedDict('user-supplied', {'d' : npt.NDArray})]
                             = {'d' : 1},
                             influx_method:
                                 Literal['normal', 'constant', 'user-supplied']
                                 = 'constant',
                             influx_args : Union[TypedDict('normal', {'mu' : float, 'sigma' : float}),
                                                TypedDict('constant', {'b' : float}),
                                                TypedDict('user-supplied', {'b' : npt.NDArray})]
                             = {'b' : 1},
                             outflux_method:
                                 Literal['normal', 'constant', 'user-supplied']
                                 = 'constant',
                             outflux_args : Union[TypedDict('normal', {'mu' : float, 'sigma' : float}),
                                                TypedDict('constant', {'o' : float}),
                                                TypedDict('user-supplied', {'o' : npt.NDArray})]
                             = {'o' : 1},
                             leakage_method:
                                 Literal['normal', 'constant', 'user-supplied']
                                 = 'constant',
                             leakage_args : Union[TypedDict('normal', {'mu' : float, 'sigma' : float}),
                                                  TypedDict('constant', {'l' : float}),
                                                  TypedDict('user-supplied', {'l' : npt.NDArray})]
                             = {'l' : 0.8},
                             energy_method:
                                 Literal['normal', 'constant', 'user-supplied']
                                 = 'constant',
                             energy_args : Union[TypedDict('normal', {'mu' : float, 'sigma' : float}),
                                                 TypedDict('constant', {'w' : float}),
                                                 TypedDict('user-supplied', {'w' : npt.NDArray})]
                             = {'w' : 1},
                             metabolic_method:
                                 Literal['dirichlet', 'user-supplied']
                                 = 'dirichlet',
                             metabolic_args : Union[TypedDict('dirichlet', {'s' : float}),
                                                    TypedDict('user-supplied', {'D' : npt.NDArray})]
                             = {'s' : 0.05}):

        '''

        Generate parameters specific to the MiCRM - consumer death
        (maintenance) rates, resource influx and outflux rates, resource
        leakage fractions, resource energy contents and the metabolic matrix.

        Parameters
        ----------
        death_method : str
            Method used to generate death rates. Options are:
                'normal' : normally distributed parameters
                'constant' : death rates are fixed
                'user-supplied' : supply your own death rates
        death_args : dict
            Arguments for death_method.
            If 'normal', {'mu': mean, 'sigma' : standard deviation}
            If 'constant', {'d' : val}
            If 'used-supplied', {'d' : array_of_vals}
        influx_method, outflux_method, leakage_method, energy_method : str
            Methods used to generate resource influx rates (b), resource
            outflux rates (o), leakage fractions (l) and resource energy
            contents (w). Options are the same as death_method, but named
            'b', 'o', 'l' and 'w' respectively.
        influx_args, outflux_args, leakage_args, energy_args : dict
            Arguments for the respective methods. Options are the same as
            death_args.
        metabolic_method : str
            Method used to generate the metabolic matrix D, where D[a, b] is
            the fraction of leaked resource b converted into resource a.
            Options are:
                'dirichlet' : columns are sampled from a Dirichlet
                distribution with sparsity s, {'s' : sparsity}
                'user-supplied' : {'D' : matrix}
            Columns of D should sum to 1 (mass conservation).
        metabolic_args : dict
            Arguments for metabolic_method.

        Returns
        -------
        None.

        '''

        # labels used to assign parameters as object attributes
        p_labels = ['d', 'b', 'o', 'l', 'w', 'D']

        # dimensions of each set of parameters
        dims_list = [(self.no_species, ), (self.no_resources, ),
                     (self.no_resources, ), (self.no_resources, ),
                     (self.no_resources, ),
                     (self.no_resources, self.no_resources)]

        # generate parameters used the other_parameter_methods method
        for p_method, p_args, p_label, dims in \
            zip([death_method, influx_method, outflux_method,
                 leakage_method, energy_method, metabolic_method],
                [death_args, influx_args, outflux_args,
                 leakage_args, energy_args, metabolic_args],
                p_labels, dims_list):

                self.other_parameter_methods(p_method, p_args, p_label, dims)

    def collate_parameters(self):

        return (self.no_species, self.growth, self.consumption,
                self.d, self.b, self.o, self.l, self.w, self.D,
                getattr(self, "timescalar", 1))

    #########################################################

    def simulation(self,
                   t_end : float,
                   initial_abundance : npt.NDArray):

        '''

        Simulate community dynamics

        Parameters
        ----------
        t_end : float
            Simulation end time.
        initial_abundance : np.ndarray
            Initial abundances of species and resources.

        Returns
        -------
        Bunch object produced by scipy.integrate.solve_ivp
            Simulation.

        '''

        unbounded_growth.terminal = True

        # call the ODE solver with the unbounded growth event function
        # the ode solver stops when the event function is true (returns 0)
        return solve_ivp(self.model, [0, t_end], initial_abundance,
                         args = self.collate_parameters(),
                         method = 'LSODA',
                         rtol = 1e-7,
                         atol = np.concatenate([np.full(self.no_species, 10.0**-15),
                                                np.full(self.no_resources, 10.0**-15)]),
                         t_eval = np.linspace(0, t_end, 200),
                         jac = self.jacobian,
                         events = unbounded_growth)

    def model(self,
              t, y,
              S, G, C, D, B, O, L, W, DM,
              e):

        '''

        ODE for the MiCRM

        Parameters
        ----------
        t : float
            time
        y : np.ndarray
            consumer and resource abundances at time t
        S : int
            species pool size (used to separate y into species and resource
                               abundances)
        G : np.ndarray
            matrix of consumer growth rates (species x resources)
        C : np.ndarray
            matrix of resource consumption rates (resources x species)
        D : np.ndarray
            consumer death rates
        B : np.ndarray
            resource influx rates
        O : np.ndarray
            resource outflux rates
        L : np.ndarray
            resource leakage fractions
        W : np.ndarray
            resource energy contents
        DM : np.ndarray
            metabolic matrix (resources x resources)
        e : float
            timescale of the resource dynamics relative to consumers

        Returns
        -------
        np.ndarray
            Rate of change in species and resource abundances over time
            (dNdt and dRdt)

        '''

        # separate species and resource abundances
        species, resources = y[:S], y[S:]

        # change in consumer abundances over time
        dNdt = species * (G @ ((1 - L) * W * resources) - D)

        # consumption flux of each resource
        consumption_flux = resources * (C @ species)

        # change in resource abundances over time
        dRdt = (1.0/e) * ((B - O * resources) - consumption_flux \
                          + (DM @ (W * L * consumption_flux))/W)

        if e > 1e-6: immigration = 1e-8
        else: immigration = 1e-12

        return np.concatenate((dNdt, dRdt)) + immigration

    def jacobian(self,
                 t, y,
                 S, G, C, D, B, O, L, W, DM,
                 e):

        species, resources = y[:S], y[S:]

        consumption_term = C @ species

        # (M x M) operator mapping consumption flux to net resource change
        flux_operator = (DM * (W * L)[np.newaxis, :])/W[:, np.newaxis] \
                        - np.eye(self.no_resources)

        J = np.zeros((y.size, y.size))

        J[:S, :S] = np.diag(G @ ((1 - L) * W * resources) - D)
        J[:S, S:] = (G * ((1 - L) * W)[np.newaxis, :]) * species[:, np.newaxis]
        J[S:, :S] = (1.0 / e) * (flux_operator @ (C * resources[:, np.newaxis]))
        J[S:, S:] = (1.0 / e) * (flux_operator * consumption_term[np.newaxis, :]
                                 - np.diag(O))

        return J

# %%

class LB_CRM(ParametersInterface, DifferentialEquationsInterface,
             CommunityPropertiesInterface):

    '''

    Consumer-resource model where consumers produce and leach essential
    biomolecules (e.g., amino acids) that they cannot consume themselves.

        dR_a/dt = sum_i l_ia P_ia N_i - sum_i c_ia (1 - P_ia) N_i R_a
        dN_i/dt = N_i (sum_a g_ia (1 - P_ia) R_a - d_i)

    '''

    def __init__(self, no_species, no_resources):

        self.no_species = no_species
        self.no_resources = no_resources

    def model_specific_rates(self,
                             death_method :
                                 Literal['normal', 'constant', 'user-supplied']
                                 = 'constant',
                             death_args : Union[TypedDict('normal', {'mu' : float, 'sigma' : float}),
                                                TypedDict('constant', {'d' : float}),
                                                TypedDict('user-supplied', {'d' : npt.NDArray})]
                             = {'d' : 1},
                             leach_method:
                                 Literal['normal', 'constant', 'user-supplied']
                                 = 'constant',
                             leach_args : Union[TypedDict('normal', {'mu' : float, 'sigma' : float}),
                                                TypedDict('constant', {'l' : float}),
                                                TypedDict('user-supplied', {'l' : npt.NDArray})]
                             = {'l' : 1},
                             production_method:
                                 Literal['bernoulli', 'user-supplied']
                                 = 'bernoulli',
                             production_args : Union[TypedDict('bernoulli', {'c' : float}),
                                                     TypedDict('user-supplied', {'P' : npt.NDArray})]
                             = {'c' : 1}):

        '''

        Generate parameters specific to the leached-biomolecule CRM - consumer
        death rates, leach rates and the production matrix.

        Parameters
        ----------
        death_method : str
            Method used to generate death rates. Options are:
                'normal' : normally distributed parameters
                'constant' : death rates are fixed
                'user-supplied' : supply your own death rates
        death_args : dict
            Arguments for death_method.
            If 'normal', {'mu': mean, 'sigma' : standard deviation}
            If 'constant', {'d' : val}
            If 'used-supplied', {'d' : array_of_vals}
        leach_method : str
            Method used to generate leach rates l (species x resources).
            Options are the same as death_method, but named 'l' rather than 'd'.
        leach_args : dict
            Arguments for leach_method. Options are the same as death_args.
        production_method : str
            Method used to generate the binary production matrix P
            (species x resources). Options are:
                'bernoulli' : P_ia ~ Bernoulli(c/M), so each consumer produces
                c resources on average, {'c' : expected no. produced resources}
                'user-supplied' : {'P' : binary matrix}
        production_args : dict
            Arguments for production_method.

        Returns
        -------
        None.

        '''

        # labels used to assign parameters as object attributes
        p_labels = ['d', 'l', 'P']

        # dimensions of each set of parameters
        dims_list = [(self.no_species, ),
                     (self.no_species, self.no_resources),
                     (self.no_species, self.no_resources)]

        # generate parameters used the other_parameter_methods method
        for p_method, p_args, p_label, dims in \
            zip([death_method, leach_method, production_method],
                [death_args, leach_args, production_args],
                p_labels, dims_list):

                self.other_parameter_methods(p_method, p_args, p_label, dims)

    def collate_parameters(self):

        return (self.no_species, self.growth, self.consumption,
                self.d, self.l, self.P, getattr(self, "timescalar", 1))

    #########################################################

    def simulation(self,
                   t_end : float,
                   initial_abundance : npt.NDArray):

        '''

        Simulate community dynamics

        Parameters
        ----------
        t_end : float
            Simulation end time.
        initial_abundance : np.ndarray
            Initial abundances of species and resources.

        Returns
        -------
        Bunch object produced by scipy.integrate.solve_ivp
            Simulation.

        '''

        unbounded_growth.terminal = True

        # call the ODE solver with the unbounded growth event function
        # the ode solver stops when the event function is true (returns 0)
        return solve_ivp(self.model, [0, t_end], initial_abundance,
                         args = self.collate_parameters(),
                         method = 'LSODA',
                         rtol = 1e-7,
                         atol = np.concatenate([np.full(self.no_species, 10.0**-15),
                                                np.full(self.no_resources, 10.0**-15)]),
                         t_eval = np.linspace(0, t_end, 200),
                         jac = self.jacobian,
                         events = unbounded_growth)

    def model(self,
              t, y,
              S, G, C, D, L, P,
              e):

        '''

        ODE for the CRM with leached biomolecules

        Parameters
        ----------
        t : float
            time
        y : np.ndarray
            consumer and resource abundances at time t
        S : int
            species pool size (used to separate y into species and resource
                               abundances)
        G : np.ndarray
            matrix of consumer growth rates (species x resources)
        C : np.ndarray
            matrix of resource consumption rates (resources x species)
        D : np.ndarray
            consumer death rates
        L : np.ndarray
            matrix of leach rates (species x resources)
        P : np.ndarray
            binary production matrix (species x resources)
        e : float
            timescale of the resource dynamics relative to consumers

        Returns
        -------
        np.ndarray
            Rate of change in species and resource abundances over time
            (dNdt and dRdt)

        '''

        # separate species and resource abundances
        species, resources = y[:S], y[S:]

        # change in consumer abundances over time
        dNdt = species * (((1 - P) * G) @ resources - D)

        # change in resource abundances over time
        dRdt = (1.0/e) * ((L * P).T @ species \
                          - resources * (((1 - P) * C.T).T @ species))

        if e > 1e-6: immigration = 1e-8
        else: immigration = 1e-12

        return np.concatenate((dNdt, dRdt)) + immigration

    def jacobian(self,
                 t, y,
                 S, G, C, D, L, P,
                 e):

        species, resources = y[:S], y[S:]

        # growth and consumption restricted to non-produced resources
        G_np = (1 - P) * G
        C_np = (1 - P) * C.T

        J = np.zeros((y.size, y.size))

        J[:S, :S] = np.diag(G_np @ resources - D)
        J[:S, S:] = G_np * species[:, np.newaxis]
        J[S:, :S] = (1.0 / e) * ((L * P).T - C_np.T * resources[:, np.newaxis])
        J[S:, S:] = (1.0 / e) * np.diag(-(C_np.T @ species))

        return J
