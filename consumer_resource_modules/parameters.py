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
                                 consumption : Union[npt.NDArray, None] = None,
                                 growth : Union[npt.NDArray, None] = None,
                                 no_negative : bool = False):
        
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
        
        # assign statistical properties of growth and consumption rates to object
        for name, statistic in zip(['mu_c', 'mu_g', 'sigma_c', 'sigma_g', 'rho'],
                                   [mu_c, mu_g, sigma_c, sigma_g, rho]): 
            
                setattr(self, name, statistic)

        # generate random variables for growth and consumption rates
        X_c, X_g = np.random.randn(self.no_resources, self.no_species),\
                    np.random.randn(self.no_species, self.no_resources)
                    
        if no_negative is True:
            
            X_c, X_g = np.abs(X_c), np.abs(X_g)
        
        match method:
            
            case 'coupled by rho':
                
                if hasattr(self, "rho") is True:
                    
                    self.consumption = self.mu_c + self.sigma_c*X_c
                    self.growth = self.mu_g + self.sigma_g*(self.rho*X_c.T + np.sqrt(1 - self.rho**2)*X_g)
                
                else:
                    
                    raise Exception('Please supply a value for rho.\n' + \
                                    '(In growth_consumption_method(), add a rho = x argument.)')
                
            case 'growth function of consumption':
                
                self.consumption = self.mu_c + self.sigma_c*X_c
                self.rue = self.mu_g + self.sigma_g*X_g
                
                self.growth = self.rue * self.consumption.T
            
            case 'consumption function of growth':
            
                self.growth = self.mu_g + self.sigma_g*X_g
                self.rue = self.mu_c + self.sigma_c*X_c
                    
                self.consumption = self.rue * self.growth.T
                
            case 'user-supplied':
                
                if consumption is not None and growth is not None:
                    
                    self.consumption = consumption
                    self.growth = growth 
                    
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
                    
    def growth_consumption_rates_noneg(self,
                                 method : Literal['coupled by rho',
                                                  'growth function of consumption',
                                                  'consumption function of growth',
                                                  'user supplied'],
                                 mu_c : float,
                                 sigma_c : float,
                                 mu_g : float,
                                 sigma_g : float,
                                 conserve_mass : bool = False):
        
        # assign statistical properties of growth and consumption rates to object
        for name, statistic in zip(['mu_c', 'mu_g', 'sigma_c', 'sigma_g'],
                                   [mu_c, mu_g, sigma_c, sigma_g]): 
            
            setattr(self, name, statistic)
            
        # generate random variables for growth and consumption rates
        X_c = np.random.gamma(shape = (self.mu_c/self.sigma_c)**2,
                              scale = (self.sigma_c)**2/self.mu_c, 
                              size = (self.no_resources, self.no_species))
        
        X_g = np.random.gamma(shape = (self.mu_g/self.sigma_g)**2,
                              scale = (self.sigma_g/self.mu_g)**2, 
                              size = (self.no_species, self.no_resources))
            
        match method:
            
            case 'growth function of consumption':
                 
                 self.consumption = X_c
                 self.rue = X_g
                 
                 self.growth = self.rue * self.consumption.T
    
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
                    
                except KeyError as e:
                    
                    print("You need to supply a value for 'mu' and 'sigma' in your dictionary argument.")
                
            case 'constant':
                
                try:
                    
                    # assign fixed value of parameter to object
                    setattr(self, p_label + '_val', parameter_args[p_label])
                    
                    # generate parameters and assign to object
                    setattr(self, p_label, parameter_args[p_label] * np.ones(dims))
                
                except KeyError as e:
                    
                    print("You need to supply a value for " + p_label + " in your dictionary argument.")
                    
            case 'user-supplied':
                
                try:
                    
                    # assign user-supplied rates
                    setattr(self, p_label, parameter_args[p_label])
                
                except KeyError as e:
                    
                    print("You need to supply parameters for " + p_label + " in your dictionary argument.")

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
    