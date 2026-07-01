# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 18:10:06 2024

@author: jamil
"""

# %%
import numpy as np
import numpy.typing as npt
from typing import Literal, Union

# %%

class ParametersInterface:
    
    # Public methods
    
    def growth_consumption_rates(self,
                                 method : Literal['coupled by rho',
                                                  'growth function of consumption',
                                                  'consumption function of growth',
                                                  'user-supplied'],
                                 mu_c : Union[float, None],
                                 sigma_c : Union[float, None],
                                 mu_g : Union[float, None],
                                 sigma_g : Union[float, None],
                                 rho : Union[float, None] = None,
                                 user_consumption : Union[npt.NDArray, None] = None,
                                 user_growth : Union[npt.NDArray, None] = None,
                                 trophic_level : Union[int, None] = None):
        
        '''
        
        Parameters
        ----------
        method : str
            Type of method used to generate growth and consumption rates.
            Options are:
                'coupled by rho' - growth and consumption linear functions,
                coupled by a parameter rho that controls their reciprocity
                See Blumenthal et al., 2024 for details.
                
                'growth function of consumption' - growth and consumtion are coupled
                by a yield conversion factor. consumption rates = c, growth = g*c,
                therefore mu_c and sigma_c are the mean and std. dev. in consumption,
                and mu_g and sigma_g are the mean and std. dev. in yield conversion.
                
                'consumption function of growth' - growth and consumtion are coupled
                by a yield conversion factor. consumption rates = gc, growth = g,
                therefore mu_c and sigma_c are the mean and std. dev. in yield conversion,
                and mu_g and sigma_g are the mean and std. dev. in growth.
                
                'user supplied' - supply your own growth and consumption rates.
                If used, mu_c, sigma_c, mu_g sigma_g can be set to some arbitary value or None,
                or if you know the means and std. devs. of your rates, you can
                supply them instead.
        mu_c : float
            mean of parameter that determines consumption rates.
        mu_g : float
            mean of parameter that determines growth rates.
        sigma_c : float
            standard deviation of parameter that determines consumption rates.
        sigma_g : float
            standard deviation of parameter that determines growth rates.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''

        # generate random variables for growth and consumption rates
        X_c, X_g = self.__random_variable_component(trophic_level) 
        
        match method:
            
            case 'coupled by rho':
                
                if rho:
                    
                    consumption, growth = self.__gc_rho_coupled(X_c, X_g,
                                                                mu_c, sigma_c,
                                                                mu_g, sigma_g,
                                                                rho)
                
                else:
                    
                    raise Exception('Please supply a value for rho.\n' + \
                                    '(In growth_consumption_method(), add a rho = x argument.)')
                
            case 'growth function of consumption':
                
                consumption, rue, growth = self.__gc_rue_coupled(X_c, X_g,
                                                                 mu_c, sigma_c,
                                                                 mu_g, sigma_g)
            
            case 'consumption function of growth':
            
                growth, rue, consumption = self.__gc_rue_coupled(X_g, X_c,
                                                                 mu_g, sigma_g,
                                                                 mu_c, sigma_c)
                
            case 'user-supplied':
                
                if user_consumption is not None and user_growth is not None:
                    
                    consumption = user_consumption
                    growth = user_growth 
                    
                else: 
                        
                    raise Exception('Please supply your growth or consumption rates.\n'
                                    '(In growth_consumption_method(), add the arguments ' 
                                    'consumption = <some np.array>, growth = <some np.array>')
                    
            case _:
                
                raise Exception('You have not selected an exisiting method.\n' + \
                      'Please chose from either "coupled by rho", ' + \
                          '"growth function of consumption", ' + \
                              '"consumption function of growth", or ' + \
                                  '"user-supplied".')
                    
        # assign statistical properties of growth and consumption rates to object
        for name, statistic in zip(['mu_c', 'mu_g', 'sigma_c', 'sigma_g', 'rho',
                                    'growth', 'consumption'],
                                   [mu_c, mu_g, sigma_c, sigma_g, rho,
                                    growth, consumption]):
            
            if hasattr(self, "trophic_levels"):
                
                setattr(self, name + "_" + str(trophic_level), statistic)
                
            else:
            
                setattr(self, name, statistic)    
    
    def other_parameter_methods(self,
                                parameter_method : str,
                                parameter_args : dict,
                                p_label : str,
                                dims : tuple):
        
        '''
        
        Generate other model parameters (e.g. consumer death rates)
        
    
        Parameters
        ----------
        parameter_method : str
            Options are:
                'normal' - parameters are normally distributed
                'constant' - parameters are fixed
        parameter_args : dict
            parameter method arguments.
        p_label : str
            name of attribute to assign parameter values to.
        dims : tuple
            Dimensions of the parameter set (e.g., array, matrix).
    
        Returns
        -------
        None.
    
        '''
        
        match parameter_method:
            
            case 'normal':
                
                try:
                
                    mu, sigma = parameter_args['mu'], parameter_args['sigma']
                    
                    # assign statistical properties to object
                    setattr(self, 'mu_' + p_label[0], mu)
                    setattr(self, 'sigma_' + p_label[0], sigma)
                    
                    # generate parameters
                    parameters = self.__normal_parameters(mu, sigma, dims)
                    
                    # assign parameters to class attributes
                    setattr(self, p_label, parameters)
                    
                except KeyError:
                    
                    print("You need to supply a value for 'mu' and 'sigma' in your dictionary argument.")
                
            case 'constant':
                
                try:
                    
                    # assign fixed value of parameter to object
                    setattr(self, p_label + '_val', parameter_args[p_label])
                    
                    # generate parameters and assign to object
                    setattr(self, p_label, parameter_args[p_label] * np.ones(dims))
                
                except KeyError:
                    
                    print("You need to supply a value for " + p_label + " in your dictionary argument.")
                    
            case 'user-supplied':
                
                try:
                    
                    # assign user-supplied rates
                    setattr(self, p_label, parameter_args[p_label])
                
                except KeyError:
                    
                    print("You need to supply parameters for " + p_label + " in your dictionary argument.")
    
                    
# %%

# private associated methods

    def __random_variable_component(self, 
                                    trophic_level):
        
        if not trophic_level:
            
            X_c, X_g = np.random.randn(self.no_resources, self.no_consumers),\
                        np.random.randn(self.no_consumers, self.no_resources)
                        
        else: 
               
            no_lowlvl = self.pool_sizes[trophic_level - 2]
            
            no_upplvl = self.pool_sizes[trophic_level - 1]
            
            X_c, X_g = np.random.randn(no_lowlvl, no_upplvl),\
                        np.random.randn(no_upplvl, no_lowlvl)
                
        return X_c, X_g
    
    def __gc_rho_coupled(self, X_c, X_g,
                         mu_c, sigma_c, mu_g, sigma_g, rho):
        
        consumption = mu_c + sigma_c*X_c
        growth = mu_g + sigma_g*(rho*X_c.T + np.sqrt(1 - rho**2)*X_g)
        
        return consumption, growth
    
    def __gc_rue_coupled(self, X_base, X_rue,
                                mu_base, sigma_base, mu_rue, sigma_rue):
        
        base = mu_base + sigma_base*X_base
        rue = mu_rue + sigma_rue*X_rue
        
        base_coupled = rue * base.T
        
        return base, rue, base_coupled
        
    def __normal_parameters(self, mu, sigma, dims):
        
        '''
        
        Generate normally distributed parameters

        Parameters
        ----------
        mu : float
            mean.
        sigma : float
            standard deviation.
        dims : tuple
            dimensions of the parameter set (e.g. could be wanting to generate 
                                             an array of matrix of parameters).

        Returns
        -------
        np.ndarray
            normally distributed parameters.

        '''
        return mu + sigma*np.random.randn(*dims)
    