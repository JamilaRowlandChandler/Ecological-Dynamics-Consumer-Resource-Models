# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 18:10:06 2024

@author: jamil
"""

# %%
import numpy as np
import numpy.typing as npt
from typing import Literal, Union, TypedDict
from scipy.stats import gamma

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
            
            if getattr(self, "trophic_levels", None):
                
                setattr(self, name + "_" + str(trophic_level), statistic)
                
            else:
            
                setattr(self, name, statistic)    
                
# %%

# private associated methods

    def __random_variable_component(self, trophic_level):
        
        if not trophic_level:
            
            X_c, X_g = np.random.randn(self.no_resources, self.no_species),\
                        np.random.randn(self.no_species, self.no_resources)
                        
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
        
        
                 
# %%
            
    def growth_consumption_rates_old(self,
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

    def metabolic_network(self,
                          energies : Union[None, npt.NDArray] = None,
                          network_method : Literal['gamma', 'step'] = 'gamma',
                          resource_conversions : Union[TypedDict('gamma',
                                                                 {'mean' : float,
                                                                  'variance' : float}),
                                                       TypedDict('step',
                                                                 {'p_s' : float})]
                              = {'mean' : 1, 'variance' : 1},
                          production_method :
                              Literal['normal', 'constant', 'user-supplied']
                              = 'constant',
                          production_args :
                              Union[TypedDict('normal', {'mu' : float, 'sigma' : float}),
                                    TypedDict('constant', {'p' : float}),
                                    TypedDict('user-supplied', {'p' : npt.NDArray})]
                              = {'p' : 1}):

        '''

        Generate a structured metabolic network and resource production rates
        for the metabolic pathway consumer-resource model.

        Parameters
        ----------
        energies : np.ndarray or None
            Resource 'energies' w_alpha. If None, sampled from Uniform(0, 1)
            with sample size = no_resources. If supplied, used directly.
        network_method : str
            Method used to generate the probability that a metabolic link
            exists between a pair of resources alpha, beta (then used as the
            Bernoulli probability for sampling q_{i, alpha, beta}). Options are:
                'gamma' : probability given by a gamma distribution used as a
                likelihood function f(x), evaluated at x = w_alpha - w_beta
                and normalised by its value at the distribution's mode.
                'step' : probability is p_s if w_alpha - w_beta > 0, and 0
                otherwise.
        resource_conversions : dict
            Arguments for network_method.
            If 'gamma', mean and variance of the gamma distribution,
            e.g. {'mean' : mean, 'variance' : variance}
            If 'step', the link probability for w_alpha - w_beta > 0,
            e.g. {'p_s' : p_s}
        production_method : str
            Method used to generate resource production rates, p_{i, alpha},
            where alpha is the byproduct/target resource being produced
            (not the resource being consumed).
            Options are:
                'normal' : normally distributed parameters
                'constant' : production rates are fixed
                'user-supplied' : supply your own production rates
        production_args : dict
            Arguments for production_method. Options are the same as
            other_parameter_methods args, but named 'p' rather than 'd'.

        Returns
        -------
        None.

        '''

        # resource energies, w_alpha - either user-supplied or Uniform(0, 1)
        if energies is None:

            self.w = np.random.uniform(0, 1, self.no_resources)

        else:

            self.w = energies

        # pairwise energy differences, w_alpha - w_beta
        energy_differences = self.w[:, np.newaxis] - self.w[np.newaxis, :]

        match network_method:

            case 'gamma':

                # gamma distribution used as a likelihood function f(x), giving
                # the probability of a metabolic link existing between
                # resources alpha and beta
                mean, variance = resource_conversions['mean'], resource_conversions['variance']

                self.mean_q, self.variance_q = mean, variance

                gamma_shape, gamma_scale = mean**2/variance, variance/mean

                if gamma_shape >= 1:

                    mode = (gamma_shape - 1) * gamma_scale

                else:

                    mode = 0

                link_probability = gamma.pdf(energy_differences, a = gamma_shape, scale = gamma_scale) / \
                                gamma.pdf(mode, a=gamma_shape, scale=gamma_scale)

            case 'step':

                # a metabolic link exists with probability p_s if resource
                # alpha has higher energy than resource beta, and never
                # otherwise
                p_s = resource_conversions['p_s']

                self.p_s = p_s

                link_probability = np.where(energy_differences > 0, p_s, 0)

        # sample the metabolic network - an independent Bernoulli trial for
        # each consumer, for every ordered pair of resources (alpha, beta)
        self.q = np.random.binomial(1, link_probability,
                                    size = (self.no_species, self.no_resources,
                                            self.no_resources))

        # generate resource production rates, p_{i, alpha} (alpha = the
        # byproduct/target resource being produced)
        self.other_parameter_methods(production_method, production_args, 'p',
                                     (self.no_species, self.no_resources))

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
    