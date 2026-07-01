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
                                            "Externally-supplied resources",
                                            "Hybrid resource supply"],
                            pool_sizes : Union[list[int],
                                               dict[str : int],
                                               npt.NDArray[int]]):
    
    '''
    
    Wrapper for different consumer resource model classes

    Parameters
    ----------
    model : str
        Type of consumer resource model. Options are:
            "Self-limiting resource supply" - resources grow logistically
            "Self-limiting resource supply, leached" - producers grow logistically, leach resources
            "Self-limiting resource supply, self-inhibition" - same as
                "Self-limiting resource supply", but with direct consumer self-inhibition
            "Externally-supplied resources" - chemostat-style resource dynamics
            (constant influx + dilution)
            "Hybrid resource supply" - combined "Self-limiting resource supply"
                and "Externally-supplied resources"
    pool_sizes : Union[list[int], dict[str : int], npt.NDArray[int]]
        Size of each pool. 
        If given as a list, 1st entry is the resource pool size, 2nd entry
        is consumers pool size, etc

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
            
            instance = SL_CRM(pool_sizes)
            
        case "Self-limiting resource supply, self-inhibition":
            
            instance = SL_SI_CRM(pool_sizes)
            
        case "Self-limiting resource supply, multi-trophic level":
            
            instance = SL_TL_CRM(len(pool_sizes), pool_sizes)
            
        case "Externally-supplied resources":
            
            instance = ES_CRM(pool_sizes)
            
        case "Hybrid resource supply":
            
            instance = Hybrid_CRM(pool_sizes)
            
        case _:
            
            raise Exception('You have not selected an exisiting model.\n' + \
                  'Please chose from either "Self-limiting resource supply"' + \
                      '"Self-limiting resource supply, leached"' + \
                      '"Self-limiting resource supply, self-inhibition"' + \
                      '"Externally-supplied resources"'  + \
                      'or "Hybrid resource supply"')
    return instance

# %%

class SL_CRM(ParametersInterface,
             DifferentialEquationsInterface,
             CommunityPropertiesInterface):
    
    '''
    
    Consumer-resource model (CRM) class with self-limiting resource supply
    
    '''
    
    def __init__(self, pool_sizes : Union[list[int],
                                          dict[str : int],
                                          npt.NDArray[int]]):
        
        '''
        
        Initiate model
        
        Parameters
        ----------
        pool_sizes : Union[list[int], dict[str : int], npt.NDArray[int]]

        Returns
        -------
        None.

        '''
    
        # assign pool sizes as class attributes
        if isinstance(pool_sizes, (list, np.ndarray)):
            
            self.no_resources, self.no_consumers = pool_sizes
            
        else:
            
            self.no_resources, self.no_consumers = pool_sizes['no_resources'], pool_sizes['no_resources']
        
        # make sure pool sizes are in the right order for generating initial conditions
        #   and supplying them to your ODE
        self.pool_sizes = [self.no_resources, self.no_consumers]
            
        
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
        
        self.other_parameter_methods(resource_growth_method,
                                     resource_growth_args,
                                     'b',
                                     (self.no_resources, ))
        
        self.other_parameter_methods(death_method,
                                     death_args,
                                     'd',
                                     (self.no_consumers, ))
        
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
            Initial abundances of consumers and resources.

        Returns
        -------
        Bunch object produced by scipy.integrate.solve_ivp
            Simulation.

        '''
        
        def model(t, y,
                  M, G, C, D, B):
            
            '''
            
            ODE for CRM with self-limiting resource supply

            Parameters
            ----------
            t : float
                time
            y : np.ndarray
                consumer and resource abundances at time t
            M : int
                resource pool size (used to separate y into consumers and resource 
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
                Rate of change in consumers and resource abundances over time 
                (dNdt and dRdt)

            '''
    
            # separate consumers and resource abundances
            resources, consumers = y[:M], y[M:]
            
            # change in resource abundances over time
            dRdt = (resources * (B - resources)) - \
                (resources * np.sum(C * consumers, axis=1))
                
            # change in consumer abundances over time
            dNdt = consumers * (np.sum(G * resources, axis = 1) - D)
                
            return np.concatenate((dRdt, dNdt)) + 1e-8
        
        unbounded_growth.terminal = True
        
        # call the ODE solver with the unbounded growth event function
        # the ode solver stops when the event function is true (returns 0)           
        return solve_ivp(model,
                         [0, t_end],
                         initial_abundance, 
                         args = (self.no_resources, self.growth, self.consumption, 
                                 self.d, self.b),
                         method = 'LSODA', rtol = 1e-7, atol = 1e-9,
                         t_eval = np.linspace(0, t_end, 200),
                         events = unbounded_growth)
    
# %%

class SL_SI_CRM(ParametersInterface, DifferentialEquationsInterface,
                CommunityPropertiesInterface):
    
    '''
    
    Consumer-resource model (CRM) class with self-limiting resource supply and
    direct consumer self-inhibition
    
    '''
    
    def __init__(self, pool_sizes : Union[list[int],
                                          dict[str : int],
                                          npt.NDArray[int]]):
        
        '''
        
        Initiate model
        
        Parameters
        ----------
        pool_sizes : Union[list[int], dict[str : int], npt.NDArray[int]]

        Returns
        -------
        None.

        '''
        
        # assign pool sizes as class attributes
        if isinstance(pool_sizes, (list, np.ndarray)):
            
            self.no_resources, self.no_consumers = pool_sizes
            
        else:
            
            self.no_resources, self.no_consumers = pool_sizes['no_resources'], pool_sizes['no_resources']
        
        # make sure pool sizes are in the right order for generating initial conditions
        #   and supplying them to your ODE
        self.pool_sizes = [self.no_resources, self.no_consumers]
        
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
        
        self.other_parameter_methods(resource_growth_method,
                                     resource_growth_args,
                                     'b',
                                     (self.no_resources, ))
        
        self.other_parameter_methods(death_method,
                                     death_args,
                                     'd',
                                     (self.no_consumers, ))
        
        self.other_parameter_methods(si_method,
                                     si_args,
                                     'si',
                                     (self.no_consumers, ))
        
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
            Initial abundances of consumers and resources.

        Returns
        -------
        Bunch object produced by scipy.integrate.solve_ivp
            Simulation.

        '''
        
        def model(t, y,
                  M, G, C, D, B, SI):
            
            '''
            
            ODE for CRM with self-limiting resource supply

            Parameters
            ----------
            t : float
                time
            y : np.ndarray
                consumer and resource abundances at time t
            M : int
                resource pool size (used to separate y into consumer and resource 
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
                Rate of change in consumers and resource abundances over time 
                (dNdt and dRdt)

            '''
            
            # separate consumers and resource abundances
            resources, consumers = y[:M], y[M:]
            
            # change in consumer abundances over time
            dNdt = consumers * ((np.sum(G * resources, axis = 1) - D) - SI*consumers)
           
            # change in resource abundances over time
            dRdt = (resources * (B - resources)) - \
                (resources * np.sum(C * consumers, axis=1))

            return np.concatenate((dRdt, dNdt)) + 1e-8
        
        unbounded_growth.terminal = True
        
        # call the ODE solver with the unbounded growth event function
        # the ode solver stops when the event function is true (returns 0)           
        return solve_ivp(model,
                         [0, t_end],
                         initial_abundance, 
                         args = (self.no_resources, self.growth, self.consumption, 
                                 self.d, self.b, self.si),
                         method = 'LSODA', rtol = 1e-7, atol = 1e-9,
                         t_eval = np.linspace(0, t_end, 200),
                         events = unbounded_growth)
    
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
        no_consumers : int
            consumers pool size
        no_resources : int
            resource pool size

        Returns
        -------
        None.

        '''
        
        # assign consumers and resource pool size as class attributes
        
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
        p_labels = ['b', 'Aij'] + \
                    ['d_' + str(tl) 
                     for tl in np.arange(2, self.trophic_levels + 1)]
                    
        
        # dimensions for death rates and intrinsic growth rates
        dims_list = [(self.no_resources, self.no_resources)] + \
                    [(pool_size, ) for pool_size in self.pool_sizes]
                        
        methods_list = [resource_growth_method,resource_interaction_method] + \
                        death_methods
        
        args_list = [resource_growth_args, resource_interaction_args] + death_args
        
        # generate parameters used the other_parameter_methods method
        for p_method, p_args, p_label, dims in \
            zip(methods_list, args_list,
                p_labels, dims_list):
                
                self.other_parameter_methods(p_method, p_args, p_label, dims)
                
        np.fill_diagonal(self.Aij, 1)
    
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
            Initial abundances of consumers and resources.

        Returns
        -------
        Bunch object produced by scipy.integrate.solve_ivp
            Simulation.

        '''
        
        def model(t, N,
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
                predator pool size (used to separate y into predator, consumers and resource 
                                   abundances)
            S : int
                consumers pool size (used to separate y into predator, consumers and resource 
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
                Rate of change in consumers and resource abundances over time 
                (dNdt and dRdt)

            '''
            
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
        
        unbounded_growth.terminal = True
        
        # call the ODE solver with the unbounded growth event function
        # the ode solver stops when the event function is true (returns 0)           
        return solve_ivp(model, [0, t_end], initial_abundance, 
                         args = (np.append(0, np.cumsum(self.pool_sizes)),
                                 [getattr(self, "growth_" + str(i)) 
                                  for i in np.arange(2, self.trophic_levels + 1)],
                                 [getattr(self, "consumption_" + str(i)) 
                                  for i in np.arange(2, self.trophic_levels + 1)],
                                 [getattr(self, "d_" + str(i)) 
                                  for i in np.arange(2, self.trophic_levels + 1)],
                                 self.b, self.Aij),
                         method = 'LSODA', rtol = 1e-7, atol = 1e-9,
                         t_eval = np.linspace(0, t_end, 200),
                         events = unbounded_growth)

# %%

class ES_CRM(ParametersInterface,
             DifferentialEquationsInterface,
             CommunityPropertiesInterface):
    
    def __init__(self, pool_sizes : Union[list[int],
                                          dict[str : int],
                                          npt.NDArray[int]]):
        
        '''
        
        Initiate model
        
        Parameters
        ----------
        pool_sizes : Union[list[int], dict[str : int], npt.NDArray[int]]

        Returns
        -------
        None.

        '''
        
        if isinstance(pool_sizes, (list, np.ndarray)):
            
            self.no_resources, self.no_consumers = pool_sizes
            
        else:
            
            self.no_resources, self.no_consumers = pool_sizes['no_resources'], pool_sizes['no_resources']
        
        # make sure pool sizes are in the right order for generating initial conditions
        #   and supplying them to your ODE
        self.pool_sizes = [self.no_resources, self.no_consumers]
    
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
                                                TypedDict('constant', {'o' : float}),
                                                TypedDict('user-supplied', {'o' : npt.NDArray})]
                             = {'o' : 1},
                             outflux_method: 
                                 Literal['normal', 'constant', 'user-supplied'] 
                                 = 'constant',
                             outflux_args : Union[TypedDict('normal', {'mu' : float, 'sigma' : float}),
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
        
        self.other_parameter_methods(influx_method,
                                     influx_args,
                                     'o',
                                     (self.no_resources, ))
        
        self.other_parameter_methods(outflux_method,
                                     outflux_args,
                                     'b',
                                     (self.no_resources, ))
        
        self.other_parameter_methods(death_method,
                                     death_args,
                                     'd',
                                     (self.no_consumers, ))
                
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
            Initial abundances of consumers and resources.

        Returns
        -------
        Bunch object produced by scipy.integrate.solve_ivp
            Simulation.

        '''
        
        def model(t, y,
                  M, G, C, D, B, O):
            
            '''
            
            ODE for CRM with self-limiting resource supply

            Parameters
            ----------
            t : float
                time
            y : np.ndarray
                consumer and resource abundances at time t
            M : int
                resource pool size (used to separate y into consumer and resource 
                                   abundances)
            G : np.ndarray
                matrix of consumer growth rates
            C : np.ndarray
                matrix of resource consumption rates
            D : np.ndarray
                consumer death rates
            B : np.ndarray
                resource influx rate 
            O : np.ndarray
                intrinsic resource growth or outflux rates

            Returns
            -------
            np.ndarray
                Rate of change in consumers and resource abundances over time 
                (dNdt and dRdt)

            '''
            
            # extinction threshold
            #y[y < 1e-5] = 0
            
            # separate consumers and resource abundances
            resources, consumers = y[:M], y[M:]
            
            # change in consumer abundances over time
            dNdt = consumers * (np.sum(G * resources, axis = 1) - D)
        
            # change in resource abundances over time
            dRdt = (O - B * resources) - \
                (resources * np.sum(C * consumers, axis=1))
                
            return np.concatenate((dRdt, dNdt)) + 1e-8
        
        unbounded_growth.terminal = True
        
        # call the ODE solver with the unbounded growth event function
        # the ode solver stops when the event function is true (returns 0)           
        return solve_ivp(model, [0, t_end],
                         initial_abundance, 
                         args = (self.no_resources,
                                 self.growth,
                                 self.consumption, 
                                 self.d, self.b, self.o),
                         method = 'LSODA', rtol = 1e-7, atol = 1e-9,
                         t_eval = np.linspace(0, t_end, 200), events = unbounded_growth)
    
# %%
    
class Hybrid_CRM(ParametersInterface,
                 DifferentialEquationsInterface,
                 CommunityPropertiesInterface):
    
    def __init__(self, pool_sizes : Union[list[int],
                                          dict[str : int],
                                          npt.NDArray[int]]):
        
        '''
        
        Initiate model
        
        Parameters
        ----------
        pool_sizes : Union[list[int], dict[str : int], npt.NDArray[int]]

        Returns
        -------
        None.

        '''
        
        if isinstance(pool_sizes, (list, np.ndarray)):
            
            self.no_resources, self.no_consumers = pool_sizes
            
        else:
            
            self.no_resources, self.no_consumers = pool_sizes['no_resources'], pool_sizes['no_resources']
        
        # make sure pool sizes are in the right order for generating initial conditions
        #   and supplying them to your ODE
        self.pool_sizes = [self.no_resources, self.no_consumers]
    
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
        
        self.other_parameter_methods(influx_method,
                                     influx_args,
                                     'o',
                                     (self.no_resources, ))
        
        self.other_parameter_methods(outflux_method,
                                     outflux_args,
                                     'b',
                                     (self.no_resources, ))
        
        self.other_parameter_methods(resource_inhibition_method,
                                     resource_inhibition_args,
                                     'a',
                                     (self.no_resources, ))
        
        self.other_parameter_methods(death_method,
                                     death_args,
                                     'd',
                                     (self.no_consumers, ))
        
                
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
            Initial abundances of consumers and resources.

        Returns
        -------
        Bunch object produced by scipy.integrate.solve_ivp
            Simulation.

        '''
        
        def model(t, y,
                  M, G, C, D, B, O, A):
            
            '''
            
            ODE for CRM with self-limiting resource supply

            Parameters
            ----------
            t : float
                time
            y : np.ndarray
                consumer and resource abundances at time t
            M : int
                resource pool size (used to separate y into consumer and resource 
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
                Rate of change in consumers and resource abundances over time 
                (dNdt and dRdt)

            '''
            
            # extinction threshold
            #y[y < 1e-5] = 0
            
            # separate consumers and resource abundances
            resources, consumers = y[:M], y[M:]
            
            # change in consumer abundances over time
            dNdt = consumers * (np.sum(G * resources, axis = 1) - D)
            
            # change in resource abundances over time
            dRdt = (B + O * resources - A * resources**2) - \
                (resources * np.sum(C * consumers, axis=1))
                
            return np.concatenate((dRdt, dNdt)) + 1e-8
        
        unbounded_growth.terminal = True
        
        # call the ODE solver with the unbounded growth event function
        # the ode solver stops when the event function is true (returns 0)           
        return solve_ivp(model, [0, t_end], initial_abundance, 
                         args = (self.no_resources, self.growth, self.consumption, 
                                 self.d, self.b, self.o, self.a),
                         method = 'LSODA', # 'Radau', #'RK45',
                         rtol = 1e-9, atol = 1e-7,
                         t_eval = np.linspace(0, t_end, 200), events = unbounded_growth)
    
