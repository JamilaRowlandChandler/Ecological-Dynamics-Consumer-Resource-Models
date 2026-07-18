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
                                            "Hybrid resource supply",
                                            "Metabolic pathways"],
                            pool_sizes : Union[list[int], tuple[int], dict[str, int],
                                               npt.NDArray] = None):

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
            "Metabolic pathways" - consumer-resource dynamics with a structured
            metabolic network determining consumer growth and resource byproducts
    pool_sizes : list, tuple, dict, or np.ndarray
        Size of each pool. If given as a list/tuple/array, the 1st entry is
        the resource pool size, the 2nd entry is the species pool size (and,
        for "Self-limiting resource supply, multi-trophic level", subsequent
        entries are the pool sizes of higher trophic levels). If given as a
        dict, use the keys 'no_resources' and 'no_species'.

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

        case "Self-limiting resource supply, leached":

            instance = SL_CRPM(pool_sizes)

        case "Self-limiting resource supply, self-inhibition":

            instance = SL_SI_CRM(pool_sizes)

        case "Self-limiting resource supply, multi-trophic level":

            instance = SL_TL_CRM(pool_sizes)

        case "Externally-supplied resources":

            instance = ES_CRM(pool_sizes)

        case "Hybrid resource supply":

            instance = Hybrid_CRM(pool_sizes)

        case "Metabolic pathways":

            instance = MP_CRM(pool_sizes)

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
    
    def __init__(self, pool_sizes : Union[list[int], tuple[int], dict[str, int],
                                          npt.NDArray]):

        '''

        Initiate model

        Parameters
        ----------
        pool_sizes : list, tuple, dict, or np.ndarray
            Size of each pool - 1st entry (or 'no_resources') is the resource
            pool size, 2nd entry (or 'no_species') is the species pool size.

        Returns
        -------
        None.

        '''

        # assign species and resource pool size as class attributes
        if isinstance(pool_sizes, (list, tuple, np.ndarray)):

            self.no_resources, self.no_species = pool_sizes

        else:

            self.no_resources, self.no_species = \
                pool_sizes['no_resources'], pool_sizes['no_species']

        # make sure pool sizes are in the right order for generating initial
        # conditions and supplying them to your ODE
        self.pool_sizes = [self.no_resources, self.no_species]

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
                                     (self.no_species, ))

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
                resource pool size (used to separate y into resource and species
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

            # separate resource and species abundances
            resources, species = y[:M], y[M:]

            # change in consumer abundances over time
            dNdt = species * (np.sum(G * resources, axis = 1) - D)

            # change in resource abundances over time
            dRdt = (resources * (B - resources)) - \
                (resources * np.sum(C * species, axis=1))

            return np.concatenate((dRdt, dNdt)) + 1e-8

        unbounded_growth.terminal = True

        # call the ODE solver with the unbounded growth event function
        # the ode solver stops when the event function is true (returns 0)
        return solve_ivp(model, [0, t_end], initial_abundance,
                         args = (self.no_resources, self.growth, self.consumption,
                                 self.d, self.b),
                         method = 'LSODA', rtol = 1e-7, atol = 1e-9,
                         t_eval = np.linspace(0, t_end, 200),
                         events = unbounded_growth)

# %%

class SL_CRPM(ParametersInterface,
              DifferentialEquationsInterface,
              CommunityPropertiesInterface):
    
    '''
    
    Consumer-resource model (CRM) class with self-limiting resource supply
    
    '''
    
    def __init__(self, pool_sizes : Union[list[int], tuple[int], dict[str, int],
                                          npt.NDArray]):

        '''

        Initiate model

        Parameters
        ----------
        pool_sizes : list, tuple, dict, or np.ndarray
            Size of each pool - 1st entry (or 'no_resources') is the resource
            pool size, 2nd entry (or 'no_species') is the species pool size.
            The producer pool size is always equal to the resource pool size.

        Returns
        -------
        None.

        '''

        # assign species and resource pool size as class attributes
        if isinstance(pool_sizes, (list, tuple, np.ndarray)):

            self.no_resources, self.no_species = pool_sizes

        else:

            self.no_resources, self.no_species = \
                pool_sizes['no_resources'], pool_sizes['no_species']

        self.no_producers = self.no_resources

        # so this class works with the shared (pool_sizes-based) initial
        # condition generation machinery in InitialConditionsInterface -
        # this class keeps a species-then-resources-then-producers ODE layout
        self.pool_sizes = [self.no_species, self.no_resources, self.no_producers]

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
        p_labels = ['d', 'b', 'Aij']
        
        # dimensions for death rates and intrinsic growth rates
        dims_list = [(self.no_species, ),
                     (self.no_resources, ),
                     (self.no_resources, self.no_resources)]
        
        # generate parameters used the other_parameter_methods method
        for p_method, p_args, p_label, dims in \
            zip([death_method, resource_growth_method, resource_interaction_method],
                [death_args, resource_growth_args, resource_interaction_args],
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
            Initial abundances of species and resources.

        Returns
        -------
        Bunch object produced by scipy.integrate.solve_ivp
            Simulation.

        '''
        
        def model(t, y,
                  S, M, G, C, D, B, A):
            
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
            species, resources, producers = y[:S], y[S:S+M], y[S+M:]
            
            # change in consumer abundances over time
            dNdt = species * (np.sum(G * resources, axis = 1) - D)
            
            # change in resource abundances over time
            dRdt = B*producers - resources * np.sum(C * species, axis=1)
                
            dPdt = producers * (1 - producers - A @ producers)
                
            return np.concatenate((dNdt, dRdt, dPdt)) + 1e-8
        
        unbounded_growth.terminal = True
        
        # call the ODE solver with the unbounded growth event function
        # the ode solver stops when the event function is true (returns 0)           
        return solve_ivp(model, [0, t_end], initial_abundance, 
                         args = (self.no_species, self.no_resources,
                                 self.growth, self.consumption, 
                                 self.d, self.b, self.Aij),
                         method = 'LSODA', rtol = 1e-7, atol = 1e-9,
                         t_eval = np.linspace(0, t_end, 200),
                         events = unbounded_growth)
    
# %%

class SL_SI_CRM(ParametersInterface,
                DifferentialEquationsInterface,
                CommunityPropertiesInterface):
    
    '''
    
    Consumer-resource model (CRM) class with self-limiting resource supply and
    direct consumer self-inhibition
    
    '''
    
    def __init__(self, pool_sizes : Union[list[int], tuple[int], dict[str, int],
                                          npt.NDArray]):

        '''

        Initiate model

        Parameters
        ----------
        pool_sizes : list, tuple, dict, or np.ndarray
            Size of each pool - 1st entry (or 'no_resources') is the resource
            pool size, 2nd entry (or 'no_species') is the species pool size.

        Returns
        -------
        None.

        '''

        if isinstance(pool_sizes, (list, tuple, np.ndarray)):

            self.no_resources, self.no_species = pool_sizes

        else:

            self.no_resources, self.no_species = \
                pool_sizes['no_resources'], pool_sizes['no_species']

        # make sure pool sizes are in the right order for generating initial
        # conditions and supplying them to your ODE
        self.pool_sizes = [self.no_resources, self.no_species]

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
                                     (self.no_species, ))

        self.other_parameter_methods(si_method,
                                     si_args,
                                     'si',
                                     (self.no_species, ))

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
                resource pool size (used to separate y into resource and species
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

            # separate resource and species abundances
            resources, species = y[:M], y[M:]

            # change in consumer abundances over time
            dNdt = species * ((np.sum(G * resources, axis = 1) - D) - SI*species)

            # change in resource abundances over time
            dRdt = (resources * (B - resources)) - \
                (resources * np.sum(C * species, axis=1))

            return np.concatenate((dRdt, dNdt)) + 1e-8

        unbounded_growth.terminal = True

        # call the ODE solver with the unbounded growth event function
        # the ode solver stops when the event function is true (returns 0)
        return solve_ivp(model, [0, t_end], initial_abundance,
                         args = (self.no_resources, self.growth, self.consumption,
                                 self.d, self.b, self.si),
                         method = 'LSODA', rtol = 1e-7, atol = 1e-9,
                         t_eval = np.linspace(0, t_end, 200), events = unbounded_growth)

# %%

class SL_TL_CRM(ParametersInterface,
                DifferentialEquationsInterface,
                CommunityPropertiesInterface):
    
    '''
    
    Consumer-resource model (CRM) class with self-limiting resource supply
    
    '''
    
    def __init__(self,
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
        
        self.trophic_levels = len(pool_sizes)
        
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
        # (death rates are per consumer trophic level, i.e. pool_sizes[1:] -
        #  pool_sizes[0] is the resource pool, used by 'b' and 'Aij' instead)
        dims_list = [(pool_size, ) for pool_size in self.pool_sizes[1:]] + \
                        [(self.pool_sizes[0], ), (self.pool_sizes[0], self.pool_sizes[0])]
        
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
                         t_eval = np.linspace(0, t_end, 200), events = unbounded_growth)

# %%

class ES_CRM(ParametersInterface, DifferentialEquationsInterface,
             CommunityPropertiesInterface):

    def __init__(self, pool_sizes : Union[list[int], tuple[int], dict[str, int],
                                          npt.NDArray]):

        if isinstance(pool_sizes, (list, tuple, np.ndarray)):

            self.no_resources, self.no_species = pool_sizes

        else:

            self.no_resources, self.no_species = \
                pool_sizes['no_resources'], pool_sizes['no_species']

        # make sure pool sizes are in the right order for generating initial
        # conditions and supplying them to your ODE
        self.pool_sizes = [self.no_resources, self.no_species]

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

        self.other_parameter_methods(influx_method,
                                     influx_args,
                                     'b',
                                     (self.no_resources, ))

        self.other_parameter_methods(outflux_method,
                                     outflux_args,
                                     'o',
                                     (self.no_resources, ))

        self.other_parameter_methods(death_method,
                                     death_args,
                                     'd',
                                     (self.no_species, ))

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
                resource pool size (used to separate y into resource and species
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

            # separate resource and species abundances
            resources, species = y[:M], y[M:]

            # change in consumer abundances over time
            dNdt = species * (np.sum(G * resources, axis = 1) - D)

            # change in resource abundances over time
            dRdt = (B - O * resources) - \
                (resources * np.sum(C * species, axis=1))

            return np.concatenate((dRdt, dNdt)) + 1e-8

        unbounded_growth.terminal = True

        # call the ODE solver with the unbounded growth event function
        # the ode solver stops when the event function is true (returns 0)           
        return solve_ivp(model, [0, t_end], initial_abundance,
                         args = (self.no_resources, self.growth, self.consumption,
                                 self.d, self.b, self.o),
                         method = 'LSODA', rtol = 1e-7, atol = 1e-9,
                         t_eval = np.linspace(0, t_end, 200), events = unbounded_growth)

# %%

class Hybrid_CRM(ParametersInterface, DifferentialEquationsInterface,
                 CommunityPropertiesInterface):

    def __init__(self, pool_sizes : Union[list[int], tuple[int], dict[str, int],
                                          npt.NDArray]):

        if isinstance(pool_sizes, (list, tuple, np.ndarray)):

            self.no_resources, self.no_species = pool_sizes

        else:

            self.no_resources, self.no_species = \
                pool_sizes['no_resources'], pool_sizes['no_species']

        # make sure pool sizes are in the right order for generating initial
        # conditions and supplying them to your ODE
        self.pool_sizes = [self.no_resources, self.no_species]


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
                                     'b',
                                     (self.no_resources, ))

        self.other_parameter_methods(outflux_method,
                                     outflux_args,
                                     'o',
                                     (self.no_resources, ))

        self.other_parameter_methods(resource_inhibition_method,
                                     resource_inhibition_args,
                                     'a',
                                     (self.no_resources, ))

        self.other_parameter_methods(death_method,
                                     death_args,
                                     'd',
                                     (self.no_species, ))

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
                resource pool size (used to separate y into resource and species
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

            # separate resource and species abundances
            resources, species = y[:M], y[M:]

            # change in consumer abundances over time
            dNdt = species * (np.sum(G * resources, axis = 1) - D)

            # change in resource abundances over time
            dRdt = (B + O * resources - A * resources**2) - \
                (resources * np.sum(C * species, axis=1))

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

# %%

class MP_CRM(ParametersInterface,
             DifferentialEquationsInterface,
             CommunityPropertiesInterface):

    '''

    Consumer-resource model (CRM) with structured metabolic pathways.

    Consumers grow by metabolising resources along a structured metabolic
    network, and release metabolic byproducts back into the resource pool.

    '''

    def __init__(self, pool_sizes : Union[list[int], tuple[int], dict[str, int],
                                          npt.NDArray]):

        '''

        Initiate model

        Parameters
        ----------
        pool_sizes : list, tuple, dict, or np.ndarray
            Size of each pool - 1st entry (or 'no_resources') is the resource
            pool size, 2nd entry (or 'no_species') is the species pool size.

        Returns
        -------
        None.

        '''

        if isinstance(pool_sizes, (list, tuple, np.ndarray)):

            self.no_resources, self.no_species = pool_sizes

        else:

            self.no_resources, self.no_species = \
                pool_sizes['no_resources'], pool_sizes['no_species']

        # make sure pool sizes are in the right order for generating initial
        # conditions and supplying them to your ODE
        self.pool_sizes = [self.no_resources, self.no_species]

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
                             resource_growth_method :
                                 Literal['normal', 'constant', 'user-supplied']
                                 = 'constant',
                             resource_growth_args :
                                 Union[TypedDict('normal', {'mu' : float, 'sigma' : float}),
                                       TypedDict('constant', {'b' : float}),
                                       TypedDict('user-supplied', {'b' : npt.NDArray})]
                                 = {'b' : 1},
                             resource_inhibition_method:
                                 Literal['normal', 'constant', 'user-supplied']
                                 = 'constant',
                             resource_inhibition_args : Union[TypedDict('normal', {'mu' : float, 'sigma' : float}),
                                                              TypedDict('constant', {'A' : float}),
                                                              TypedDict('user-supplied', {'A' : npt.NDArray})]
                             = {'A' : 1}):

        '''

        Generate parameters specific to the metabolic pathway CRM - consumer
        death rates, and the resource supply, self-decay and self-inhibition
        rates (o_alpha, b_alpha, A_{alpha alpha}).

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
            Method used to generate intrinsic resource supply rates, o_alpha.
            Options are the same as death_method, but named 'o' rather than 'd'.
        influx_args : dict
            Arguments for influx_method. Options are the same as
            death_method args.
        resource_growth_method : str
            Method used to generate resource self-decay rates, b_alpha.
            Options are the same as death_method, but named 'b' rather than 'd'.
        resource_growth_args : dict
            Arguments for resource_growth_method. Options are the same as
            death_method args.
        resource_inhibition_method : str
            Method used to generate resource self-inhibition rates, A_{alpha alpha}.
            Options are the same as death_method, but named 'A' rather than 'd'.
        resource_inhibition_args : dict
            Arguments for resource_inhibition_method. Options are the same as
            death_method args.

        Returns
        -------
        None.

        '''

        self.other_parameter_methods(influx_method,
                                     influx_args,
                                     'o',
                                     (self.no_resources, ))

        self.other_parameter_methods(resource_growth_method,
                                     resource_growth_args,
                                     'b',
                                     (self.no_resources, ))

        self.other_parameter_methods(resource_inhibition_method,
                                     resource_inhibition_args,
                                     'A',
                                     (self.no_resources, ))

        self.other_parameter_methods(death_method,
                                     death_args,
                                     'd',
                                     (self.no_species, ))

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

        M = self.no_resources
        C = self.consumption
        D, O, B, A = self.d, self.o, self.b, self.A

        unbounded_growth.terminal = True

        if not self.growth_saturation:

            # network-derived quantities (G_effective, C_gated,
            # production_gate_weighted*) are precomputed once in
            # metabolic_network() and stored as attributes, since they depend
            # only on the fixed metabolic network and rate parameters, never on
            # the dynamical state (N, R) or time
            G_effective = self.G_effective
            C_gated = self.C_gated
            production_gate_weighted = self.production_gate_weighted
            production_gate_weighted_flat = self.production_gate_weighted_flat
            production_gate_weighted_T = self.production_gate_weighted_T

            def model(t, y):

                '''

                ODE for the metabolic pathway CRM

                Parameters
                ----------
                t : float
                    time
                y : np.ndarray
                    consumer and resource abundances at time t

                Returns
                -------
                np.ndarray
                    Rate of change in species and resource abundances over time
                    (dNdt and dRdt)

                '''

                # separate resource and species abundances
                resources, species = y[:M], y[M:]

                # change in consumer abundances over time
                dNdt = species * (np.sum(G_effective * resources, axis=1) - D)

                # resources consumed, summed over consumers
                consumed = resources * np.sum(species[:, np.newaxis] * C_gated, axis=0)

                # resources produced as metabolic byproducts, summed over
                # consumers and source resources
                production_weight = species[:, np.newaxis] * C.T * resources[np.newaxis, :]
                produced = production_weight.reshape(-1) @ production_gate_weighted_flat

                # change in resource abundances over time - O is a constant supply
                # (influx), B*resources is intrinsic (logistic-style) resource
                # growth, and A*resources**2 is the self-limiting/carrying-capacity
                # term
                dRdt = (O + B*resources - A*resources**2) - consumed + produced

                return np.concatenate((dRdt, dNdt)) + 1e-8

            def jacobian(t, y):

                '''

                Analytic Jacobian of the metabolic pathway CRM's ODE, d(dy/dt)/dy.

                Supplying this to solve_ivp avoids LSODA falling back to a
                finite-difference Jacobian, which needs ~2*(M + S) extra calls
                to model() per Jacobian update - one analytic evaluation here
                costs about the same as a single call to model().

                '''

                resources, species = y[:M], y[M:]

                # d(dNdt)/d(species) - diagonal (no direct species-species term)
                dNdN_diag = np.sum(G_effective * resources, axis=1) - D

                # d(dNdt)/d(resources)
                dNdR = species[:, np.newaxis] * G_effective

                # d(dRdt)/d(resources) - diagonal self-decay/inhibition/consumption
                # terms, plus a dense block from the production term
                dRdR_diag = B - 2*A*resources - species @ C_gated
                production_weight_by_species = species[:, np.newaxis] * C.T
                dense_RR = np.matmul(production_weight_by_species.T[:, np.newaxis, :],
                                     production_gate_weighted_T)[:, 0, :]
                dRdR = np.diag(dRdR_diag) + dense_RR.T

                # d(dRdt)/d(species) - consumption term + production term
                dRdN_consumption = -resources[:, np.newaxis] * C_gated.T
                production_weight_by_resource = C.T * resources[np.newaxis, :]
                dense_RN = np.matmul(production_weight_by_resource[:, np.newaxis, :],
                                     production_gate_weighted)[:, 0, :]
                dRdN = dRdN_consumption + dense_RN.T

                J = np.zeros((M + self.no_species, M + self.no_species))
                J[:M, :M] = dRdR
                J[:M, M:] = dRdN
                J[M:, :M] = dNdR
                J[M:, M:] = np.diag(dNdN_diag)

                return J

            # call the ODE solver with the unbounded growth event function
            # the ode solver stops when the event function is true (returns 0)
            return solve_ivp(model, [0, t_end], initial_abundance,
                             method = 'LSODA', jac = jacobian,
                             rtol = 1e-7, atol = 1e-9,
                             t_eval = np.linspace(0, t_end, 200),
                             events = unbounded_growth)

        else:

            # log-corrected growth mode: growth on the edge alpha -> beta
            # uses R_alpha*[w_alpha - w_beta - log(R_beta/R_alpha)] instead
            # of the original R_alpha*(w_alpha - w_beta). Consumption and
            # production are unchanged (still the precomputed C_gated /
            # production_gate_weighted* from metabolic_network()), so only
            # growth needs a state-dependent correction, computed fresh at
            # every ODE evaluation.
            Q = self.q
            out_degree = self.out_degree
            metabolic_gain = self.metabolic_gain
            consumption_gate = self.consumption_gate
            production_gate = self.production_gate
            G = self.growth
            C_gated = self.C_gated
            production_gate_weighted = self.production_gate_weighted
            production_gate_weighted_flat = self.production_gate_weighted_flat
            production_gate_weighted_T = self.production_gate_weighted_T

            def model(t, y):

                '''

                ODE for the metabolic pathway CRM with a log-corrected
                growth term. Consumption/production are the original form.

                Parameters
                ----------
                t : float
                    time
                y : np.ndarray
                    consumer and resource abundances at time t

                Returns
                -------
                np.ndarray
                    Rate of change in species and resource abundances over time
                    (dNdt and dRdt)

                '''

                resources, species = y[:M], y[M:]

                # metabolic_gain_mod[i,alpha] = (1/out_degree[i,alpha]) *
                # sum_beta q_{i,alpha,beta} * [w_alpha - w_beta - log(R_beta/R_alpha)]
                # = metabolic_gain[i,alpha] + log(R_alpha)*consumption_gate[i,alpha]
                #   - (1/out_degree[i,alpha]) * sum_beta q_{i,alpha,beta}*log(R_beta)
                logR = np.log(resources)
                log_R_gate = np.matmul(Q, logR) / out_degree
                metabolic_gain_mod = metabolic_gain + logR[np.newaxis, :]*consumption_gate \
                    - log_R_gate

                dNdt = species * (np.sum(G * metabolic_gain_mod * resources, axis=1) - D)

                # consumption/production unchanged from the original model
                consumed = resources * np.sum(species[:, np.newaxis] * C_gated, axis=0)
                production_weight = species[:, np.newaxis] * C.T * resources[np.newaxis, :]
                produced = production_weight.reshape(-1) @ production_gate_weighted_flat

                dRdt = (O + B*resources - A*resources**2) - consumed + produced

                return np.concatenate((dRdt, dNdt)) + 1e-8

            def jacobian(t, y):

                '''

                Analytic Jacobian for the log-corrected growth variant.

                d(dRdt)/d(resources), d(dRdt)/d(species) are identical to the
                original (growth_saturation=False) Jacobian, since
                consumption/production are unchanged.

                For growth, write dN_i/dt = N_i*(f_i(R) - d_i), with
                f_i(R) = sum_a G[i,a]*R_a*MGM[i,a](R), where MGM is
                metabolic_gain_mod above. Then
                d(MGM[i,a])/dR_gamma = +consumption_gate[i,a]*delta(a,gamma)/R_a
                                       - q_{i,a,gamma}/(out_degree[i,a]*R_gamma)
                (the first term from d(log R_a)/dR_gamma, the second from
                d(log R_beta)/dR_gamma inside the sum over beta - both signs
                flipped relative to the log(R_alpha/R_beta) version, since
                log(R_beta/R_alpha) = -log(R_alpha/R_beta)). Substituting
                into d(f_i)/dR_gamma = sum_a G[i,a]*[delta(a,gamma)*MGM[i,a] +
                R_a*d(MGM[i,a])/dR_gamma] and simplifying (R_a and 1/R_a
                cancel in the first piece) gives:
                d(f_i)/dR_gamma = G[i,gamma]*(MGM[i,gamma] + consumption_gate[i,gamma])
                                  - (1/R_gamma) * sum_a G[i,a]*R_a*q_{i,a,gamma}/out_degree[i,a]

                '''

                resources, species = y[:M], y[M:]

                logR = np.log(resources)
                log_R_gate = np.matmul(Q, logR) / out_degree
                metabolic_gain_mod = metabolic_gain + logR[np.newaxis, :]*consumption_gate \
                    - log_R_gate

                # --- d(dNdt)/d(species), d(dNdt)/d(resources) ---

                dNdN_diag = np.sum(G * metabolic_gain_mod * resources, axis=1) - D

                GR = G * resources[np.newaxis, :]
                growth_flux_contracted = np.matmul(GR[:, np.newaxis, :], production_gate)[:, 0, :]
                dNdR = species[:, np.newaxis] * \
                    (G*(metabolic_gain_mod + consumption_gate) -
                     growth_flux_contracted / resources[np.newaxis, :])

                # --- d(dRdt)/d(resources), d(dRdt)/d(species) - identical to
                # the original (growth_saturation=False) Jacobian ---

                dRdR_diag = B - 2*A*resources - species @ C_gated
                production_weight_by_species = species[:, np.newaxis] * C.T
                dense_RR = np.matmul(production_weight_by_species.T[:, np.newaxis, :],
                                     production_gate_weighted_T)[:, 0, :]
                dRdR = np.diag(dRdR_diag) + dense_RR.T

                dRdN_consumption = -resources[:, np.newaxis] * C_gated.T
                production_weight_by_resource = C.T * resources[np.newaxis, :]
                dense_RN = np.matmul(production_weight_by_resource[:, np.newaxis, :],
                                     production_gate_weighted)[:, 0, :]
                dRdN = dRdN_consumption + dense_RN.T

                J = np.zeros((M + self.no_species, M + self.no_species))
                J[:M, :M] = dRdR
                J[:M, M:] = dRdN
                J[M:, :M] = dNdR
                J[M:, M:] = np.diag(dNdN_diag)

                return J

            return solve_ivp(model, [0, t_end], initial_abundance,
                             method = 'LSODA', jac = jacobian,
                             rtol = 1e-7, atol = 1e-9,
                             t_eval = np.linspace(0, t_end, 200),
                             events = unbounded_growth)

