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
                'uniform' - parameters are uniformly distributed (use for
                anything that must stay strictly positive, e.g. K_m/V_max -
                unlike 'normal', this can't sample a negative value)
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

            case 'uniform':

                try:

                    low, high = parameter_args['low'], parameter_args['high']

                    # assign statistical properties to object
                    setattr(self, 'low_' + p_label[0], low)
                    setattr(self, 'high_' + p_label[0], high)

                    # generate parameters
                    parameters = np.random.uniform(low, high, dims)

                    # assign parameters to class attributes
                    setattr(self, p_label, parameters)

                except KeyError as e:

                    print("You need to supply a value for 'low' and 'high' in your dictionary argument.")

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
                          adjacency : Union[None, npt.NDArray] = None,
                          network_method : Literal['gamma', 'step', 'connected_gamma'] = 'gamma',
                          resource_conversions : Union[TypedDict('gamma',
                                                                 {'mean' : float,
                                                                  'variance' : float}),
                                                       TypedDict('step',
                                                                 {'p_s' : float}),
                                                       TypedDict('connected_gamma',
                                                                 {'mean' : float,
                                                                  'variance' : float,
                                                                  'extra_edge_scale' : float})]
                              = {'mean' : 1, 'variance' : 1},
                          gated : bool = True,
                          shared_network : bool = False,
                          growth_saturation : bool = False,
                          saturation_kinetics : Literal['flux', 'thermodynamic', 'reversible', 'boltzmann'] = 'flux',
                          K_m : float = 1e-8,
                          K_m_method : Literal['constant', 'normal', 'uniform', 'user-supplied'] = 'constant',
                          K_m_args : Union[TypedDict('normal', {'mu' : float, 'sigma' : float}),
                                           TypedDict('uniform', {'low' : float, 'high' : float}),
                                           TypedDict('constant', {'K_m_tensor' : float}),
                                           TypedDict('user-supplied', {'K_m_tensor' : npt.NDArray}),
                                           None]
                              = None,
                          v_max_method : Literal['constant', 'normal', 'uniform', 'user-supplied'] = 'constant',
                          v_max_args : Union[TypedDict('normal', {'mu' : float, 'sigma' : float}),
                                             TypedDict('uniform', {'low' : float, 'high' : float}),
                                             TypedDict('constant', {'v_max_tensor' : float}),
                                             TypedDict('user-supplied', {'v_max_tensor' : npt.NDArray})]
                              = {'v_max_tensor' : 1},
                          log_eps : float = 1e-4,
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
        for the metabolic pathway consumer-resource model, and precompute the
        network-derived quantities (effective growth, gated consumption,
        weighted production gating) used by MP_CRM.simulation(). These only
        depend on the fixed metabolic network and rate parameters, never on
        the dynamical state, so computing them once here (rather than on
        every ODE function evaluation) is what makes simulation() fast.

        Requires growth_consumption_rates() to have already been called,
        since self.growth/self.consumption must exist to compute the
        effective growth/consumption attributes below.

        Parameters
        ----------
        energies : np.ndarray or None
            Resource 'energies' w_alpha. If None, sampled from Uniform(0, 1)
            with sample size = no_resources. If supplied, used directly.
        adjacency : np.ndarray or None
            A pre-sampled (no_resources, no_resources) 0/1 metabolic network,
            q_{alpha, beta}, shared by every consumer (tiled along the
            species axis, like shared_network = True). If supplied, this is
            used directly and no Bernoulli sampling is performed - link_probability
            is not computed/used, so network_method/resource_conversions/gated
            are ignored for the network itself (though 'step' still needs
            resource_conversions['p_s'] set for self.p_s bookkeeping if you
            rely on it elsewhere). Lets a network be sampled once (e.g.
            outside a loop over communities) and reused identically across
            many communities, while growth/consumption rates are still
            resampled per community.
        network_method : str
            Method used to generate the probability that a metabolic link
            exists between a pair of resources alpha, beta (then used as the
            Bernoulli probability for sampling q_{i, alpha, beta}). Options are:
                'gamma' : probability given by a gamma distribution used as a
                likelihood function f(x), evaluated at x = w_alpha - w_beta
                and normalised by its value at the distribution's mode.
                Requires mean**2/variance >= 1 (see resource_conversions
                below) - raises ValueError otherwise.
                'step' : probability is p_s if w_alpha - w_beta > 0, and 0
                otherwise (or unconditionally p_s if gated = False).
                'connected_gamma' : like 'gamma', but GUARANTEES every
                resource is reachable from the dominant (highest-energy)
                resource via directed edges, instead of independently
                Bernoulli-sampling every edge and risking a disconnected
                network (found empirically: with 'gamma' on a sparse,
                low-mean setting, ~10/10 tested seeds left most resources
                unreachable from the dominant one - see
                resource_species_diversity/influx_diversity/
                network_diagnostics.py). Builds a random spanning tree
                rooted at the dominant resource first (every other resource
                picks exactly one higher-energy parent, weighted toward
                small energy gaps via the same gamma-density preference as
                'gamma'), then layers a few extra edges on top using the
                'gamma' method's own Bernoulli sampling (scaled down by
                resource_conversions['extra_edge_scale']) for branching -
                the extra-edge step can only ever ADD edges on top of the
                guaranteed-connected tree, never remove one, so
                connectivity survives regardless of how many extra edges
                land. Always gated (like 'gamma') since it only ever
                creates w_alpha > w_beta edges.
        resource_conversions : dict
            Arguments for network_method.
            If 'gamma', mean and variance of the gamma distribution,
            e.g. {'mean' : mean, 'variance' : variance}. The gamma shape
            parameter mean**2/variance must be >= 1 (i.e. variance <=
            mean**2) - below 1 the gamma density has no finite maximum (it
            diverges as x -> 0), so the mode-normalisation used here would
            silently divide by infinity and zero every link probability;
            metabolic_network() raises ValueError instead of allowing that.
            A shape just above 1 (variance just below mean**2) still puts
            the mode near zero, i.e. links between close-in-energy resources
            strongly preferred, without hitting the degenerate case.
            If 'step', the link probability, e.g. {'p_s' : p_s}
            If 'connected_gamma', mean and variance (same constraints and
            meaning as 'gamma') plus 'extra_edge_scale' (float in [0, 1],
            default 0.3 if omitted) - the Bernoulli probability multiplier
            for the extra (non-tree) edges layered on top of the guaranteed
            spanning tree. 0 gives the bare spanning tree (sparsest possible
            connected network, no_resources - 1 edges); larger values add
            progressively more redundant edges.
        gated : bool
            Only affects network_method = 'step'. If True (default), a link
            between alpha and beta exists with probability p_s only when
            w_alpha - w_beta > 0 (energy-descending links only), and never
            otherwise. If False, q_{i, alpha, beta} is instead sampled from
            a Bernoulli(p_s) independently of the sign of w_alpha - w_beta -
            i.e. an unstructured random network with a fixed link probability.
        shared_network : bool
            If True, a single (no_resources, no_resources) network is
            sampled and shared by every consumer (self.q still has shape
            (no_species, no_resources, no_resources), but is identical along
            the species axis). If False (default), each consumer gets an
            independently sampled network.
        growth_saturation : bool
            If False (default), growth/consumption/production on the edge
            alpha -> beta use the bare R_alpha factor (original
            (w_alpha - w_beta) energy term for growth, linear-in-R for
            consumption/production). If True, R_alpha is instead replaced
            everywhere (growth, consumption, and production alike) by the
            saturating per-edge flux R_alpha**2 / (R_alpha + R_beta + K_m) -
            growth/depletion from consuming alpha, and the resulting
            byproduct production into beta, are all suppressed as the
            byproduct beta accumulates relative to the source alpha. Since
            this flux is state-dependent, it can't be precomputed here -
            MP_CRM.simulation() recomputes it at every ODE evaluation
            instead, using the structural (state-independent) network
            attributes stored below (self.q, self.energy_differences,
            self.out_degree).
        saturation_kinetics : str
            Only used if growth_saturation = True - selects which saturating
            flux formula replaces the bare R_alpha factor. Options are:
                'flux' (default) - the R_alpha**2/(R_alpha+R_beta+K_m) flux
                described under growth_saturation above.
                'thermodynamic' - a reversible-Michaelis-Menten-style flux
                that also depends on the consuming species' own abundance
                N_i. Per edge alpha -> beta: writing the energy difference
                Delta = w_alpha - w_beta and the log resource ratio
                L = log(R_beta/R_alpha), the consumption/production flux is
                R_alpha / (K_m + N_i / (1 - exp(-(Delta - L)))), and the
                growth flux (replacing (w_alpha-w_beta)*R_alpha) is
                (Delta + L) times that same saturating term. (Delta - L) is
                a chemical-potential-like driving force for the alpha->beta
                reaction (since Delta - L = (w_alpha + log R_alpha) -
                (w_beta + log R_beta)); the 1-exp(-.) factor drives the flux
                to zero as the reaction approaches equilibrium, and can go
                negative (reversing the net flux) if the byproduct beta
                has accumulated enough to overcome the energy gap. Because
                the denominator depends on N_i / (1-exp(-.)) linearly, it can
                cross zero at (Delta-L) = -log(1+N_i/K_m) - a genuine
                singularity in this formula (not just a numerical
                regularisation artefact), so this mode is more prone to
                stiffness/blow-up than 'flux' and is worth testing carefully
                at small scale before large sweeps.
                'reversible' - a Haldane-style reversible saturating flux
                that, unlike 'thermodynamic', depends only on R_alpha and
                R_beta (not on N_i), so doesn't share that variant's
                species-count-driven stiffness. Per edge alpha -> beta:
                f = (R_alpha - R_beta*exp(-(w_alpha-w_beta))) /
                (K_m + R_alpha + R_beta). The denominator is always
                >= K_m > 0 (no poles). f -> R_alpha/(K_m+R_alpha) (forward
                only) as R_beta -> 0, and can go negative (net flux
                reverses, from beta to alpha) if the byproduct beta has
                built up enough relative to alpha to overcome the
                exp(-(w_alpha-w_beta)) equilibrium factor. Growth uses
                g_{i,alpha}*f directly (f already encodes the energy
                asymmetry via its exp(.) term, so - unlike 'flux' - no
                separate (w_alpha-w_beta) weighting is added on top);
                consumption/production also use f directly in place of the
                bare R_alpha factor, so growth and consumption share the
                same per-edge aggregate quantity in this variant.
                'boltzmann' - a Boltzmann-weighted LINEAR flux (not a
                saturating ratio - no denominator, so unlike every other
                growth_saturation=True mode this is unbounded in R_alpha,
                R_beta). Per edge alpha -> beta: f = exp(w_alpha/K_m)*R_alpha
                - exp(w_beta/K_m)*R_beta. K_m here is a thermal/temperature-
                like scale in a Boltzmann factor, NOT a Michaelis-Menten
                half-saturation constant - it must NOT be set small (see the
                K_m argument's docstring below for why a small K_m is
                actively dangerous for this variant specifically). f is
                positive (net forward flux, alpha -> beta) whenever
                exp(w_alpha/K_m)*R_alpha > exp(w_beta/K_m)*R_beta, and can go
                negative (net flux reverses) otherwise - like 'reversible',
                but via a linear rather than saturating comparison. Growth
                uses g_{i,alpha}*f directly (f already encodes the full
                energy weighting, so no separate (w_alpha-w_beta) factor is
                added on top, exactly as in 'reversible'); consumption/
                production also use f directly, so growth, consumption, and
                production all share the same per-edge aggregate quantity.
        K_m_method, K_m_args : str, dict
            Only used if saturation_kinetics = 'reversible'. Controls
            self.K_m_tensor, a (no_species, no_resources, no_resources)
            array giving each CONSUMER its own K_m for each metabolic
            REACTION (edge alpha -> beta) - i.e. K_m_{i,alpha,beta}, sampled
            independently per (species, resource-pair) rather than a single
            shared scalar. Options for K_m_method are the same as
            growth_consumption_rates' rate-generating methods:
                'constant' (default) - every entry equals K_m_args['K_m_tensor'];
                if K_m_args is left as None, this defaults to the scalar K_m
                argument above, reproducing the original uniform-K_m
                behaviour exactly.
                'normal' - K_m_{i,alpha,beta} ~ Normal(K_m_args['mu'], K_m_args['sigma']),
                independently per (species, resource-pair) - can sample
                negative values if sigma is large relative to mu (see the
                floor note below).
                'uniform' - K_m_{i,alpha,beta} ~ Uniform(K_m_args['low'],
                K_m_args['high']), independently per (species,
                resource-pair) - the recommended choice when K_m must stay
                strictly positive, since (unlike 'normal') it can't sample a
                negative value as long as low > 0.
                'user-supplied' - supply your own (no_species, no_resources,
                no_resources) array as K_m_args['K_m_tensor'].
            Sampled/supplied values are floored at 1e-8 regardless of method
            (a sampled K_m crossing zero or negative would flip the
            reversible flux's denominator sign unpredictably, or divide by
            zero at exactly 0) - with 'uniform' and low > 0 this floor is
            never actually triggered.
        v_max_method, v_max_args : str, dict
            Only used if saturation_kinetics = 'reversible'. Same idea as
            K_m_method/K_m_args, but for self.v_max_tensor, a multiplicative
            ceiling scaling the reversible flux per (species, resource-pair):
            f_{i,alpha,beta} = v_max_tensor_{i,alpha,beta} * (R_alpha -
            R_beta*exp(-(w_alpha-w_beta))) / (K_m_tensor_{i,alpha,beta} +
            R_alpha + R_beta). Defaults to 1 everywhere (no scaling, matching
            the original 'reversible' formula). Sampling K_m and/or v_max
            per (species, resource-pair) - rather than sharing one global
            scalar - introduces the same kind of fixed, structural
            heterogeneity across consumers that 'flux' gets "for free" from
            its (w_alpha-w_beta) energy weighting, which 'reversible'
            otherwise lacks (its sign/magnitude is driven by the current
            resource state, not by any fixed per-consumer quantity).
        log_eps : float
            Only used if growth_saturation = True and saturation_kinetics =
            'thermodynamic' - a floor applied to R before taking log(R) (used
            to form L = log(R_beta/R_alpha) and its 1/R derivatives). Unlike
            K_m, this isn't primarily a literal-zero guard (R = 0 exactly is
            rare mid-trajectory) - its main job is capping how negative L can
            get as a resource approaches extinction relative to others, which
            otherwise drives Th = 1-exp(-(Delta-L)) to very large-magnitude
            values and makes the ODE stiff. Larger log_eps (e.g. 1e-2 to
            1e-1) smooths this at the cost of changing behaviour for
            genuinely near-extinct resources; smaller log_eps (e.g. 1e-8) is
            closer to "no regularisation" and more likely to reproduce the
            stiffness seen with tiny K_m in the 'flux' variant.
        K_m : float
            Only used if growth_saturation = True - a Michaelis-Menten-style
            half-saturation constant added to the denominator of the
            saturating flux (see growth_saturation above). Only meaningful
            relative to the typical scale of resource abundances: a value
            much smaller than that scale (e.g. the default 1e-8, intended
            purely as protection against a literal 0/0 when both R_alpha and
            R_beta hit numerical zero) leaves the flux extremely steep near
            R_alpha=R_beta=0, which can make the ODE very stiff there and
            slow to integrate; a value comparable to or larger than the
            typical resource scale (e.g. 1e-4 to 1) smooths the flux over a
            wider range and is much better-conditioned for the solver, while
            also changing the model's behaviour (not just a numerical
            regularisation) by setting the concentration scale at which
            saturation kicks in.
            For saturation_kinetics = 'boltzmann', K_m plays a completely
            different role - it is NOT a half-saturation constant but a
            thermal/temperature-like scale inside exp(w_alpha/K_m), and
            MUST NOT be set small: since w is typically in [0, 1], a small
            K_m (e.g. the 1e-8 default, or the 1e-2 typically used for
            'flux') makes exp(w/K_m) astronomically large (e.g.
            exp(1/0.01) = exp(100) ~ 2.7e43), causing immediate numerical
            overflow. Choose K_m on the order of the typical energy range
            (e.g. K_m ~ 0.1-10) for 'boltzmann' instead.
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

        if not hasattr(self, 'growth') or not hasattr(self, 'consumption'):

            raise Exception('metabolic_network() requires growth_consumption_rates() '
                            'to be called first (self.growth/self.consumption are '
                            'not yet set).')

        # resource energies, w_alpha - either user-supplied or Uniform(0, 1)
        if energies is None:

            self.w = np.random.uniform(0, 1, self.no_resources)

        else:

            self.w = energies

        # pairwise energy differences, w_alpha - w_beta
        energy_differences = self.w[:, np.newaxis] - self.w[np.newaxis, :]

        if adjacency is not None:

            # network supplied directly (e.g. sampled once and reused across
            # many communities) - skip link-probability computation and
            # Bernoulli sampling entirely, just tile it across consumers
            if network_method == 'step' and 'p_s' in resource_conversions:

                self.p_s = resource_conversions['p_s']

            self.q = np.tile(adjacency, (self.no_species, 1, 1))

        else:

            match network_method:

                case 'gamma':

                    # gamma distribution used as a likelihood function f(x), giving
                    # the probability of a metabolic link existing between
                    # resources alpha and beta
                    mean, variance = resource_conversions['mean'], resource_conversions['variance']

                    self.mean_q, self.variance_q = mean, variance

                    gamma_shape, gamma_scale = mean**2/variance, variance/mean

                    if gamma_shape < 1:

                        raise ValueError(
                            "metabolic_network(network_method='gamma') requires "
                            f"mean**2/variance >= 1 (got shape={gamma_shape:.4g} "
                            f"from mean={mean}, variance={variance}). For "
                            "shape < 1 the gamma density has no finite maximum "
                            "(it diverges as x -> 0), so normalising "
                            "link_probability by its value at the mode divides "
                            "by infinity and silently zeroes every link "
                            "probability instead of raising an error. Choose "
                            "mean/variance so that variance <= mean**2 (e.g. "
                            "variance just below mean**2) to keep the mode near "
                            "zero - i.e. links between close-in-energy resources "
                            "strongly preferred - without hitting this degenerate "
                            "case.")

                    mode = (gamma_shape - 1) * gamma_scale

                    link_probability = gamma.pdf(energy_differences, a = gamma_shape, scale = gamma_scale) / \
                                    gamma.pdf(mode, a=gamma_shape, scale=gamma_scale)

                case 'step':

                    # a metabolic link exists with probability p_s if resource
                    # alpha has higher energy than resource beta, and never
                    # otherwise - unless gated = False, in which case the link
                    # probability is p_s regardless of energy ordering
                    p_s = resource_conversions['p_s']

                    self.p_s = p_s

                    if gated:

                        link_probability = np.where(energy_differences > 0, p_s, 0)

                    else:

                        link_probability = np.full_like(energy_differences, p_s)

                case 'connected_gamma':

                    # spanning tree rooted at the dominant (highest-energy)
                    # resource, guaranteeing every resource is reachable from
                    # it, plus a few extra edges on top - see this method's
                    # docstring and network_diagnostics.py for the full
                    # rationale/derivation. Sets self.q directly (both
                    # shared_network cases) since the construction doesn't
                    # fit the shared "sample q from a precomputed
                    # link_probability" path used by 'gamma'/'step' below.
                    mean, variance = resource_conversions['mean'], resource_conversions['variance']
                    extra_edge_scale = resource_conversions.get('extra_edge_scale', 0.3)

                    self.mean_q, self.variance_q = mean, variance

                    if not (0 <= extra_edge_scale <= 1):
                        raise ValueError(
                            f"extra_edge_scale must be in [0, 1], got {extra_edge_scale}")

                    gamma_shape, gamma_scale = mean**2/variance, variance/mean

                    if gamma_shape < 1:

                        raise ValueError(
                            "metabolic_network(network_method='connected_gamma') "
                            f"requires mean**2/variance >= 1 (got shape="
                            f"{gamma_shape:.4g} from mean={mean}, variance={variance}) "
                            "- see the 'gamma' ValueError message above for why "
                            "shape < 1 is degenerate.")

                    mode = (gamma_shape - 1) * gamma_scale
                    pdf_mode = gamma.pdf(mode, a=gamma_shape, scale=gamma_scale)

                    order = np.argsort(-self.w)
                    n_networks = 1 if shared_network else self.no_species

                    tree_adjacency = np.zeros(
                        (n_networks, self.no_resources, self.no_resources), dtype=int)

                    # every non-root resource picks exactly one higher-energy
                    # parent, weighted toward small energy gaps - vectorised
                    # across n_networks per rank (all networks share the same
                    # w, hence the same per-rank candidate/weight set, but
                    # each draws its own independent parent)
                    for k in range(1, self.no_resources):

                        node = order[k]
                        parent_candidates = order[:k]
                        gaps = self.w[parent_candidates] - self.w[node]
                        weights = gamma.pdf(gaps, a=gamma_shape, scale=gamma_scale) / pdf_mode
                        weights = weights / weights.sum()
                        parents = np.random.choice(parent_candidates, size=n_networks, p=weights)
                        tree_adjacency[np.arange(n_networks), parents, node] = 1

                    # extra edges on top, same 'gamma' Bernoulli approach,
                    # scaled down - can only add to the tree (np.maximum),
                    # never remove from it, so connectivity is preserved
                    link_probability = gamma.pdf(energy_differences, a=gamma_shape, scale=gamma_scale) / \
                        pdf_mode
                    link_probability = link_probability * extra_edge_scale
                    extra = np.random.binomial(1, link_probability,
                                               size=(n_networks, self.no_resources, self.no_resources))
                    network_adjacency = np.maximum(tree_adjacency, extra)

                    if shared_network:

                        self.q = np.tile(network_adjacency[0], (self.no_species, 1, 1))

                    else:

                        self.q = network_adjacency

            # sample the metabolic network - an independent Bernoulli trial for
            # each consumer, for every ordered pair of resources (alpha, beta) -
            # or a single network shared by every consumer if shared_network = True.
            # 'connected_gamma' already set self.q directly above (its
            # construction doesn't fit this precomputed-link_probability path).
            if network_method != 'connected_gamma':

                if shared_network:

                    shared_q = np.random.binomial(1, link_probability,
                                                  size = (self.no_resources, self.no_resources))

                    self.q = np.tile(shared_q, (self.no_species, 1, 1))

                else:

                    self.q = np.random.binomial(1, link_probability,
                                                size = (self.no_species, self.no_resources,
                                                        self.no_resources))

        # generate resource production rates, p_{i, alpha} (alpha = the
        # byproduct/target resource being produced)
        self.other_parameter_methods(production_method, production_args, 'p',
                                     (self.no_species, self.no_resources))

        self.growth_saturation = growth_saturation
        self.saturation_kinetics = saturation_kinetics
        self.K_m = K_m
        self.log_eps = log_eps

        # consumer- and reaction-specific K_m/V_max, only used by
        # saturation_kinetics = 'reversible' (self.K_m above stays a scalar,
        # unused by 'reversible', so 'flux'/'thermodynamic' are unaffected).
        # Defaults reproduce a uniform scalar K_m (from the K_m argument
        # above) and V_max = 1 (no scaling) everywhere, matching the
        # previous behaviour when K_m_method/v_max_method are left at
        # 'constant' - so this is backward compatible unless you explicitly
        # ask for 'normal' (or 'user-supplied') sampling.
        if K_m_args is None:

            K_m_args = {'K_m_tensor': K_m}

        self.other_parameter_methods(K_m_method, K_m_args, 'K_m_tensor',
                                     (self.no_species, self.no_resources, self.no_resources))
        self.other_parameter_methods(v_max_method, v_max_args, 'v_max_tensor',
                                     (self.no_species, self.no_resources, self.no_resources))

        # safety floor - a sampled K_m or v_max that goes negative (or hits
        # exactly 0) would either flip signs unpredictably or risk a
        # division-by-zero in the 'reversible' flux; clip both to stay
        # strictly positive rather than silently letting that happen
        floor = 1e-8
        self.K_m_tensor = np.maximum(self.K_m_tensor, floor)
        self.v_max_tensor = np.maximum(self.v_max_tensor, floor)

        # --- structural (state-independent) network quantities, needed by
        # MP_CRM.simulation() in both growth_saturation modes ---

        eps = 1e-8

        self.energy_differences = energy_differences

        # normalisation over outgoing metabolic links from each resource,
        # for each consumer - sum_gamma q_{i, alpha, gamma} (+ eps)
        self.out_degree_raw = np.sum(self.q, axis=2)
        self.out_degree = self.out_degree_raw + eps

        if growth_saturation:

            # growth/consumption/production now all depend on the dynamical
            # state (via the saturating R_alpha**2/(R_alpha+R_beta) term), so
            # the network-derived gating terms below can't be precomputed
            # here - MP_CRM.simulation() recomputes them at every ODE
            # evaluation instead, using self.q, self.energy_differences,
            # self.out_degree, self.growth, self.consumption and self.p
            # (all state-independent) plus the current resource abundances
            return

        # --- precompute network-derived quantities used by
        # MP_CRM.simulation() when growth_saturation = False - these depend
        # only on the fixed metabolic network (self.q, self.w) and rate
        # parameters (self.growth, self.consumption, self.p), never on the
        # dynamical state (N, R) or time, so computing them once here avoids
        # redoing several O(S x M x M) array operations on every single ODE
        # function evaluation ---

        # metabolic energy gain per unit resource, for each consumer -
        # sum_beta q_{i, alpha, beta}(w_alpha - w_beta) / out_degree
        self.metabolic_gain = np.sum(self.q * energy_differences[np.newaxis, :, :],
                                     axis=2) / self.out_degree

        # fraction of consumption of resource alpha channelled through the
        # metabolic network, for each consumer -
        # sum_beta q_{i, alpha, beta} / out_degree
        self.consumption_gate = self.out_degree_raw / self.out_degree

        # fraction of consumption of resource beta channelled into
        # byproduct resource alpha, for each consumer -
        # q_{i, beta, alpha} / out_degree_beta
        self.production_gate = self.q / self.out_degree[:, :, np.newaxis]

        # P (production rate) is indexed by the TARGET/produced resource
        # alpha, not the source resource beta being consumed - so it weights
        # production_gate's last axis, not the consumption rate matrix
        self.production_gate_weighted = self.p[:, np.newaxis, :] * self.production_gate

        # fold the (now precomputed) network-dependent gating terms directly
        # into the growth/consumption rate matrices
        self.G_effective = self.growth * self.metabolic_gain
        self.C_gated = self.consumption.T * self.consumption_gate

        # flatten production_gate_weighted's (species, source resource) axes
        # together so the production term becomes a single BLAS
        # matrix-vector product per step, rather than an np.einsum
        # contraction (which doesn't reliably dispatch to BLAS for this
        # index pattern)
        self.production_gate_weighted_flat = \
            self.production_gate_weighted.reshape(self.no_species * self.no_resources,
                                                  self.no_resources)

        # production_gate_weighted transposed to (source resource, species,
        # target resource) - used by the analytic Jacobian's dR/dR block
        self.production_gate_weighted_T = self.production_gate_weighted.transpose(1, 0, 2)

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
    