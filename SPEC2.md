# Create a branch in Ecological-Dynamics-Consumer-Resource-Models called gamma-branch.

## Generating only positive growth and consumption rates

# In consumer_resouce_models/parameters.py, expand growth_consumption_rates() to include the argument all_positive, which is a Boolean. Its default value is False.
# Make __gc_rho_coupled take this argument. When all_positive is False, keep the code the same as the current version of the method. When all_positive is True, add the following code:

Z_c = X_c
Z_g = self.rho*X_c.T + np.sqrt(1 - self.rho**2)*X_g

U_c = stats.norm.cdf(Z_c)
U_g = stats.norm.cdf(Z_g)

shape_c, scale_c = mu_c**2/sigma_c**2, sigma_c**2/mu_c
shape_g, scale_g = mu_g**2/sigma_g**2, sigma_g**2/mu_g

self.consumption = stats.gamma.ppf(U_c, a=shape_c, scale=scale_c)
self.growth = stats.gamma.ppf(U_g, a=shape_g, scale=scale_g)

# Verify that the mean and standard deviation of self.consumption approximately equal mu_c and sigma_c, same for self.growth with mu_g and sigma_g. 
# Verify that the correlation between self.consumption and self.growth approximately equals rho.

## Running simulations

# Now, you will be working in the external_resource_stability subdirectory.
# Update simulation_functions_new.py so that the growth_consumption_rates() and rho_coupled() functions can take all_positive as an argument if it is present in parameter_sets.

# Go into the stability_transitions folder, and make a copy of rho_vs_sigma_es_vs_el.py. Name it rho_vs_sigma_es_vs_el_allplus.py.
# In rho_vs_sigma_es_vs_el_allplus.py, Remove all the code after line 111.
# In rho_vs_sigma_es_vs_el_allplus.py, update the dicts in line 93 and 104 to include all_positive = True.
# Change the strings on lines 96 and 107 to "external_resource_stability/simulations/rho_sigma_mu50_es_allplus" and "external_resource_stability/simulations/rho_sigma_mu50_sl_allplus" respectively.

## Generating figures

# Go into the figures folder, and make a copy of abiotic_vs_biotic_resources.py. Name it abiotic_vs_biotic_resources_allplus.py.
# In abiotic_vs_biotic_resources_allplus.py, edit the code so that only stability_plot() is plotted in the figure, removing unnecessary code.
# Change lines 266 and 267 so that load_in_simulations takes the strings "rho_sigma_mu50_es_allplus" and "rho_sigma_mu50_sl_allplus" respectively.


