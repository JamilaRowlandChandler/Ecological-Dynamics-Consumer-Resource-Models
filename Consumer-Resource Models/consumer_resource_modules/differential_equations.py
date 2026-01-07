# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 15:29:00 2024

@author: jamil

=======================================================================
    This is my code for solving ordinary differential equations 
    (for the consumer-resource model).
=======================================================================
"""

# %%

import numpy as np

from initial_abundances import InitialConditionsInterface

#%%

class DifferentialEquationsInterface(InitialConditionsInterface):
    
    def simulate_community(self,
                           t_end : float,
                           no_init_cond : int,
                           init_cond_func : str ='Mallmin',
                           assign : bool = True,
                           **kwargs : any):
        
        '''
        
        Call methods that generate initial abundances and run simulations. 

        Parameters
        ----------
        t_end : float
            Simulation end time.
        no_init_cond : int
            number of sets of initial abundances to simulate from (no. repeats).
        init_cond_func : str, optional
            What function is used to generate the initial conditions. 
            The options: 
                "Mallmin" - Uniform(dispersal, 2/M)
                "user supplied" - supply your own intial abundances
            The default is 'Mallmin'.
        assign : Bool, optional
            Whether or not to assign simulations as object attributes. True for
            simulations, false for calculating lyapunov exponents. 
            The default is True.
        **kwargs : any
            Optional argument for initial condition function. Of the form
            {'user_supplied_init_cond' : {'species' : [initial species abundances],
                                          'resources' : [initial resource abundances]}}

        Returns
        -------
        ODE_sols : list
            list of simulations from n = no_init_cond initial abundances.

        '''
        
        self.t_end = t_end
        
        # generate initial species and resource abundances
        initial_abundances = \
            self.generate_initial_conditions(no_init_cond, init_cond_func, **kwargs)
            
        # simulate community dynamics for each set of initial conditions
        # - this is called from the object-specific methods
        ODE_sols = [self.simulation(t_end, initial_abundances[:, i]) 
                    for i in range(initial_abundances.shape[1])]
        
        # should simulations be assigned to a class attribute?
        if assign is True:
            
            self.ODE_sols = ODE_sols
        
        else:
            
            return ODE_sols
    

def unbounded_growth(t, var, *args):
    
    '''

    This function identifies whether unbounded growth (to infinity) is occuring 
    during simulations. If growth is unbounded, the function stops the ODE solver
    early.

    Parameters
    ----------
    t : float
        time point.
    var : np.array() of floats
        Species and resource dynamics at timte t.
    *args : TYPE
        DESCRIPTION.

    Returns
    -------
    int
        If 0 is returned, the ODE solver (solve_ivp) terminates.
        If a non-zero value is returned, the ODE solver continues to run

    '''
    
    
    # if any species or resource abundances are greater than some threshold
    # or if any species abundances are less than or equal to 0  
    if np.any(np.log10(np.abs(var) + 1e-20) > 6) or np.isnan(np.log10(np.abs(var) + 1e-20)).any():
        
        return 0 # when the returned value of an event function is 0, the ode 
                    #solver terminates.
    
    else: 
        
        return 1 # the ode solver continues because the returned value is non-zero.
