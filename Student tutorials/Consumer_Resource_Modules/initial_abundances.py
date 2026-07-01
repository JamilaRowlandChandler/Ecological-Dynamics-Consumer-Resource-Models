# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 18:50:53 2024

@author: jamil
"""

################# Initial conditions module #############

import numpy as np

####### Functions #######

class Base_InitialConditions:
       
    def initial_variable_conditions(self,
                                    no_init_cond,
                                    dims,
                                    init_cond_func,
                                    initial_condition = None):
        
        '''
        
        Generate initial abundances for one variable.
        
        '''
        
        match init_cond_func:
            
            case 'Mallmin':
                
                initial_abundance = self.initial_abundances_mallmin(dims, no_init_cond)
            
            case 'user-supplied':
                
                initial_abundance = np.array(initial_condition).reshape((len(initial_condition[0]),
                                                                         len(initial_condition)))
                
        return initial_abundance
        
    ########## Functions for generating initial conditions ############
      
    def initial_abundances_mallmin(self, dims, no_init_cond):
        
        '''
        
        Sample multiple sets of initial abundances from Uniform(dispersal, 2/M)

        Parameters
        ----------
        no_init_cond : int
            Number of sets of initial abundances to sample.
        dims : int
            Number of abundances to sample per set (e.g. species or resource pool size).

        Returns
        -------
        np.ndarray with dimensions (dims, no_init_cond)
            Array of sets of initial abundances.

        '''
    
        return np.random.uniform(1e-8, 2/dims,
                                 dims * no_init_cond).reshape((dims,
                                                               no_init_cond))
    
class InitialConditionsInterface(Base_InitialConditions):
    
    def generate_initial_conditions(self,
                                    no_init_cond : int, 
                                    init_cond_func : str,
                                    **kwargs : any):
        '''
        
        Generate and assign initial abundances for species and resources 
        from multiple options/functions.

        Parameters
        ----------
        For details, see the simulate_community method in differential_equations.py


        '''

        # user-supplied initial conditions
        if init_cond_func == "user-supplied":
            
            initial_conditions = kwargs.get("initial_conditions")
            
            initial_abundances = self.initial_variable_conditions(None,
                                                                  None,
                                                                  init_cond_func,
                                                                  initial_conditions)
        # automatically generate initial conditions
        else:
            
            initial_abundances = np.vstack([self.initial_variable_conditions(no_init_cond,
                                                                             pool_size,
                                                                             init_cond_func)
                                            for pool_size in self.pool_sizes])
                    
        return initial_abundances
    
class InitialConditionsInterface_ELV(Base_InitialConditions):
    
    def generate_initial_conditions(self,
                                    no_init_cond : int, 
                                    init_cond_func : str,
                                    **kwargs : any):
        '''
        
        Generate and assign initial abundances for species and resources 
        from multiple options/functions.

        Parameters
        ----------
        For details, see the simulate_community method in differential_equations.py


        '''
        
        species_abundances = self.initial_variable_conditions(no_init_cond,
                                                              self.no_species,
                                                              init_cond_func,
                                                              **kwargs)
        
        return species_abundances