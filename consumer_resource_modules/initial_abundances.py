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
                
                initial_abundance = initial_condition.reshape((dims, no_init_cond))
                
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

        if hasattr(self, "trophic_levels"):
            
            if init_cond_func == "user-supplied":
                
                initial_abundances = np.vstack([self.initial_variable_conditions(no_init_cond,
                                                                                 pool_size,
                                                                                 init_cond_func,
                                                                                 var_initcond)
                                                for pool_size, var_initcond in 
                                                zip(self.pool_sizes,
                                                    kwargs.get("initial_conditions"))])
                
            else:
                
                initial_abundances = np.vstack([self.initial_variable_conditions(no_init_cond,
                                                                                 pool_size,
                                                                                 init_cond_func)
                                                for pool_size in self.pool_sizes])
                    
        elif hasattr(self, "no_producers"):
            
            if init_cond_func == "user-supplied":

                
                initial_abundances = \
                    np.vstack([self.initial_variable_conditions(no_init_cond,
                                                                pool_size,
                                                                init_cond_func,
                                                                var_initcond)
                               for pool_size, var_initcond in 
                               zip([self.no_species,
                                    self.no_resources,
                                    self.no_producers], 
                                   kwargs.get("initial_conditions"))])
        
            else:
                
                initial_abundances = \
                    np.vstack([self.initial_variable_conditions(no_init_cond,
                                                                pool_size,
                                                                init_cond_func)
                               for pool_size in [self.no_species,
                                                 self.no_resources,
                                                 self.no_producers]])
                
        else:
            
            if init_cond_func == "user-supplied":

                
                initial_abundances = \
                    np.vstack([self.initial_variable_conditions(no_init_cond,
                                                                pool_size,
                                                                init_cond_func,
                                                                var_initcond)
                               for pool_size, var_initcond in 
                               zip([self.no_species, self.no_resources], 
                                   kwargs.get("initial_conditions"))])
        
            else:
                
                initial_abundances = \
                    np.vstack([self.initial_variable_conditions(no_init_cond,
                                                                pool_size,
                                                                init_cond_func)
                               for pool_size in [self.no_species, self.no_resources]])
                    
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
        
        if init_cond_func == "user-supplied":
            
            initial_abundances = self.initial_variable_conditions(no_init_cond,
                                                                  self.no_species,
                                                                  init_cond_func,
                                                                  kwargs.get("initial_conditions"))
            
        else :
            
            initial_abundances = self.initial_variable_conditions(no_init_cond,
                                                                  self.no_species,
                                                                  init_cond_func)
        
        
        return initial_abundances