# -*- coding: utf-8 -*-
"""
Created on Mon May  5 18:08:45 2025

@author: jamil

"""

import os
import sys
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from copy import deepcopy
from scipy.stats import pearsonr, norm
import seaborn as sns

os.chdir("C:/Users/jamil/Documents/PhD/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models/consumer_resource_modules")

from models import Consumer_Resource_Model
from effective_LV_models import gLV
from community_level_properties import max_le, eigenspectrum

sys.path.insert(0, 'C:/Users/jamil/Documents/PhD/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models/resource_diversity_stability(sl)')
#from simulation_functions import simulation_df_from_communities

# %%

M = 150
mu = 50
sigma = 50
d = 1
b = 1
#rho = 0.001
rhos = [1.0, 0.8, 0.6, 0.4, 0.2, 0.001] #[1.0, 0.9, 0.8, 0.7, 0.6]#[1.0, 0.8, 0.6, 0.4, 0.2, 0.001]
cvs = []
stability = []

for rho in rhos:
    
    rho = 0.89

    community = Consumer_Resource_Model("Self-limiting resource supply",
                                        M,
                                        M)
    
    community.growth_consumption_rates('coupled by rho',
                                       mu_c = mu/M,
                                       sigma_c = sigma/np.sqrt(M),
                                       mu_g = mu/M,
                                       sigma_g = sigma/np.sqrt(M) + 0.01,
                                       rho = rho)
    community.model_specific_rates(death_method = "constant",
                                   death_args = {'d' : d},
                                   resource_growth_method = "constant",
                                   resource_growth_args = {'b' : b})
    
    #print(len(np.where((community.consumption.flatten() < 0) & \
    #                   (community.growth.T.flatten() > 0))[0]))
    
    #community.consumption = np.abs(community.consumption)
    #community.growth = np.abs(community.growth)
        
    simulation_rawparam = community.simulate_community(4000, 1, assign=False)[0]
    
    plt.plot(simulation_rawparam.t,
             simulation_rawparam.y[:M, :].T)
    plt.show()
    
    #community.calculate_community_properties() 
    
    community.lyapunov_exponent = max_le(community,
                                         simulation_rawparam.y[:, -1],
                                         T = 1000,
                                         perturbation = 1e-6)
    stability.append(community.lyapunov_exponent)
    
    
    #community.consumption = np.abs(community.consumption)
    #community.growth = np.abs(community.growth)
    
    #simulation_absparam = community.simulate_community(4000, 1, assign=False)[0]
    
    #fig, (ax1, ax2) = plt.subplots(1, 2, sharex = True)
           
    #ax1.plot(simulation_rawparam.t,
    #         simulation_rawparam.y[:M, :].T)
    #ax2.plot(simulation_absparam.t,
    #         simulation_absparam.y[:M, :].T)
    #plt.show()s
    
    end_survivors = np.apply_along_axis(lambda x : np.any(x > 1e-4),
                                        1,
                                        simulation_rawparam.y[:, -50:]) 
    
    survivor_y = simulation_rawparam.y[end_survivors, -50:]
    
    cvs.append(np.mean(np.std(survivor_y)/ np.mean(survivor_y)))
    
print(np.array(cvs),
      "\n",
      np.array(stability))

# %%

resource_pool_sizes = [50, 250]
mu_c = 130
sigma_c = 1.6
mu_y = 1
sigma_y= 0.13
d = 1
b = 1

for M in resource_pool_sizes:


    community = Consumer_Resource_Model("Self-limiting resource supply",
                                        M,
                                        M)
    
    community.growth_consumption_rates('growth function of consumption',
                                       mu_c = mu_c/M,
                                       sigma_c = sigma_c/np.sqrt(M),
                                       mu_g = mu_y,
                                       sigma_g = sigma_y)
    community.model_specific_rates(death_method = "constant",
                                   death_args = {'d' : 1},
                                   resource_growth_method = "constant",
                                   resource_growth_args = {'b' : 1})
    
    community.simulate_community(3000, 1)

    #community.calculate_community_properties() 

    community.lyapunov_exponent = max_le(community,
                                         community.ODE_sols[0].y[:, -1],
                                         T = 1000,
                                         perturbation = 1e-6)
    print(community.lyapunov_exponent)

    plt.plot(community.ODE_sols[0].t,
             community.ODE_sols[0].y[:M, :].T)
    plt.show()

# %%

M = 150
mu = 100
sigma = 3

for _ in range(10):

    community = Consumer_Resource_Model("Self-limiting resource supply",
                                        M,
                                        M)
    
    community.growth_consumption_rates('coupled by rho',
                                       mu/M, sigma/np.sqrt(M),
                                       mu/M, sigma/np.sqrt(M),
                                       rho = 0.9)
    community.model_specific_rates(death_method = "normal",
                                   death_args = {'mu' : 1.0, 'sigma' : 0.0},
                                   resource_growth_method = "normal",
                                   resource_growth_args = {'mu' : 1.0,
                                                           'sigma' : 0.0})
    
    community.simulate_community(2000, 1)
    
    #community.calculate_community_properties() 
    
    #community.lyapunov_exponent = max_le(community, community.ODE_sols[0].y[:, -1],
    #                                         T = 1000, perturbation = 1e-6)
    #print(community.lyapunov_exponent)
    
    plt.plot(community.ODE_sols[0].t,
             community.ODE_sols[0].y[:M, :].T)
    plt.show()
    
    output = eigenspectrum(community,
                           community.ODE_sols[0].y[:, -1])
    
    print(output['magnitude_ratios'])

# %%

M = 180
mu_c = 200

community = Consumer_Resource_Model("Self-limiting resource supply, multi-trophic level",
                                    trophic_levels = 3,
                                    pool_sizes = [M, M, M])

community.growth_consumption_rates('growth function of consumption',
                                   mu_c/M,
                                   1.6/np.sqrt(M),
                                   1,
                                   0.13069,
                                   trophic_level = 2)
community.growth_consumption_rates('growth function of consumption',
                                   mu_c/M,
                                   1.6/np.sqrt(M),
                                   1.0,
                                   0.0,
                                   trophic_level = 3)

community.model_specific_rates(death_methods = ['constant', 'constant'],
                               death_args = [{'d_2' : 1}, {'d_3' : 0.93}],
                               resource_interaction_method = 'normal',
                               resource_interaction_args = {'mu' : 10/M, 
                                                            'sigma' : 0.5/np.sqrt(M)}) 

community.simulate_community(9000, 1)

fig, axs = plt.subplots(1, 3, figsize = (8, 2.5), layout = "constrained")
    
axs[0].plot(community.ODE_sols[0].t, community.ODE_sols[0].y[:M, :].T)
axs[1].plot(community.ODE_sols[0].t, community.ODE_sols[0].y[M:-M, :].T)
axs[2].plot(community.ODE_sols[0].t, community.ODE_sols[0].y[-M:, :].T)

plt.show()


community.calculate_community_properties() 

community.lyapunov_exponent = max_le(community, community.ODE_sols[0].y[:, -1],
                                         T = 1000, perturbation = 1e-6)

print(community.lyapunov_exponent)

# %%

community_df = simulation_df_from_communities([community, community],
                                              "Self-limiting resource supply, multi-trophic level",
                                              "growth function of consumption")

# %% MiCRM - single community

S = 100
M = 100
mu = 50
sigma = 3
rho = 0.01
d = 1
b = 1
o = 0.01

def correlated_uniforms(S, M,
                        rho,
                        mu,
                        sigma):

    # Spearman rho -> underlying Gaussian correlation
    r = 2 * np.sin(np.pi * rho / 6)

    # Correlated standard normals
    z1 = np.random.randn(S, M)
    z2 = r * z1 + np.sqrt(1 - r**2) * np.random.randn(S, M)

    # Normal CDF gives Uniform(0, 1)
    u1, u2 = norm.cdf(z1), norm.cdf(z2)

    # Rescale to the requested mean and variance
    width = np.sqrt(12 * (sigma**2)/M)
    low = np.max([0, mu/M - width / 2])
    
    return low + width * u1, (low + width * u2).T

def correlated_lognormals(S, M,
                          rho,
                          mu,
                          sigma):

    m = mu / M
    v = sigma**2 / M

    # Underlying normal parameters
    s2 = np.log(1 + v / m**2)
    mu_ln = np.log(m) - s2 / 2

    arg = 1 + rho * (np.exp(s2) - 1)
    r = np.log(arg) / s2

    # Correlated standard normals
    z1 = np.random.randn(S, M)
    z2 = r * z1 + np.sqrt(1 - r**2) * np.random.randn(S, M)

    # Exponentiate to get lognormals
    s = np.sqrt(s2)
    
    return np.exp(mu_ln + s * z1), np.exp(mu_ln + s * z2)
    

# Example
growth, consumption = correlated_uniforms(S, M,
                                          rho,
                                          mu,
                                          sigma)

community = Consumer_Resource_Model("MiCRM", S, M)

community.growth_consumption_rates('coupled by rho',
                                   mu_c = mu/M,
                                   sigma_c = sigma/np.sqrt(M),
                                   mu_g = mu/M,
                                   sigma_g = sigma/np.sqrt(M),
                                   rho = rho,
                                   all_positive = True)
                                   #consumption = consumption,
                                   #growth = growth)

community.metabolic_rates(energy_method = 'user-supplied',
                          energy_args = {'w' : np.random.uniform(0.1, 1, 
                                                                 (M, ))},
                          metabolic_method = 'dirichlet',
                          metabolic_args = {'s' : 0.05},
                          leakage_method = 'constant',
                          leakage_args = {'l' : 0.7})

community.model_specific_rates(death_method = 'constant',
                               death_args = {'d' : d},
                               influx_method = 'one-in',
                               influx_args = {'b' : b},
                               outflux_method = 'constant',
                               outflux_args = {'o' : o})

#community.b[np.random.randint(0,
#                              M,
#                              int(0.1*M))] = b

# metabolic matrix columns should sum to 1 (mass conservation)
print("MiCRM: D column sums in [", community.D.sum(axis = 0).min(), ",",
      community.D.sum(axis = 0).max(), "]")

community.simulate_community(1000, 1)

sol = community.ODE_sols[0]
print("MiCRM: simulation ended at t =", sol.t[-1],
      "| survivors =", np.sum(sol.y[:S, -1] > 1e-4), "/", S)

fig, axs = plt.subplots(1, 2, figsize = (8, 2.5), layout = "constrained")
axs[0].plot(sol.t, sol.y[:S, :].T)
axs[0].set_title("MiCRM consumers")
axs[1].plot(sol.t, sol.y[S:, :].T)
axs[1].set_title("MiCRM resources")
plt.show()

#community.lyapunov_exponent = max_le(community, sol.y[:, -1],
#                                     T = 1000, perturbation = 1e-6)
#print("MiCRM: max. Lyapunov exponent =", community.lyapunov_exponent)

# %% LB_CRM - single community

S = 90
M = 100
mu = 50
sigma = 5.0
rho = 0.01

community = Consumer_Resource_Model("Leached biomolecules", S, M)

community.growth_consumption_rates('coupled by rho',
                                   mu_c = mu/M,
                                   sigma_c = sigma/np.sqrt(M),
                                   mu_g = mu/M,
                                   sigma_g = sigma/np.sqrt(M),
                                   rho = rho)
community.model_specific_rates(death_method = 'constant',
                               death_args = {'d' : 0.1},
                               outflux_method = 'constant',
                               outflux_args = {'o' : 0},
                               leach_method = 'constant',
                               leach_args = {'l' : 1},
                               production_method = 'bernoulli',
                               production_args = {'c' : 1})

# each consumer should produce c = 10 resources on average
print("LB_CRM: mean no. produced resources per consumer =",
      community.P.sum(axis = 1).mean())

community.simulate_community(50, 1)

sol = community.ODE_sols[0]
print("LB_CRM: simulation ended at t =", sol.t[-1],
      "| survivors =", np.sum(sol.y[:S, -1] > 1e-4), "/", S)

fig, axs = plt.subplots(1, 2, figsize = (8, 2.5), layout = "constrained")
axs[0].plot(sol.t, sol.y[:S, :].T)
axs[0].set_title("LB_CRM consumers")
axs[1].plot(sol.t, sol.y[S:, :].T)
axs[1].set_title("LB_CRM resources")
plt.show()

community.lyapunov_exponent = max_le(community, sol.y[:, -1],
                                     T = 1000, perturbation = 1e-6)
print("LB_CRM: max. Lyapunov exponent =", community.lyapunov_exponent)

# %%

resource_pool_sizes = [50, 250]
mu_c = 130
sigma_c = 3
mu_y = 1
sigma_y= 0.13
d = 1
b = 1

Ms = []
les = []

for M in resource_pool_sizes:
    
    for _ in range(5):

        community = Consumer_Resource_Model("Self-limiting resource supply",
                                            M,
                                            M)
        
        community.growth_consumption_rates('growth function of consumption',
                                           mu_c = mu_c/M,
                                           sigma_c = sigma_c/M,
                                           mu_g = mu_y,
                                           sigma_g = sigma_y)
        community.model_specific_rates(death_method = "constant",
                                       death_args = {'d' : 1},
                                       resource_growth_method = "constant",
                                       resource_growth_args = {'b' : 1})
        
        community.simulate_community(7000, 1)
    
        #community.calculate_community_properties() 
    
        community.lyapunov_exponent = max_le(community,
                                             community.ODE_sols[0].y[:, -1],
                                             T = 1000,
                                             perturbation = 1e-6)
        Ms.append(M)
        les.append(deepcopy(community.lyapunov_exponent))
    
        #plt.plot(community.ODE_sols[0].t,
        #         community.ODE_sols[0].y[:M, :].T)
        #plt.show()
        
# %%

print(np.array(les)[np.array(Ms) == 250])

# %%

M = 50
mu_c = 130
sigma_c = 3
mu_y = 1
sigma_y= 0.0
d = 1
b = 1

community = Consumer_Resource_Model("Self-limiting resource supply",
                                    M,
                                    M)

community.growth_consumption_rates('growth function of consumption',
                                   mu_c = mu_c/M,
                                   sigma_c = sigma_c/M,
                                   mu_g = mu_y,
                                   sigma_g = sigma_y)
community.model_specific_rates(death_method = "constant",
                               death_args = {'d' : 1},
                               resource_growth_method = "constant",
                               resource_growth_args = {'b' : 1})

community.simulate_community(5000, 1)


plt.plot(community.ODE_sols[0].t,
         community.ODE_sols[0].y[:M, :].T)
plt.show()

# %%

M = 150
mu_a = 0.4
sigma_a = 0.1

gLV_community = gLV(M)
gLV_community.model_specific_rates(growth_method='constant',
                                   growth_args=dict(r=1.0),
                                   interaction_method='normal',
                                   interaction_args=dict(mu=mu_a,
                                                         sigma=sigma_a),
                                   self_inhibition_method='constant',
                                   self_inhibition_args=dict(Aii=1.0))

gLV_community.calculate_interaction_stats()

# run simulations from randomly generated initial abundances
gLV_community.simulate_community(t_end = 7000,
                                 no_init_cond = 1)

fig, ax = plt.subplots(1, 1)

t = gLV_community.ODE_sols[0].t
y = gLV_community.ODE_sols[0].y
        
ax.stackplot(t[10:], y[:, 10:])

ax.tick_params(axis="both", which="major", labelsize=7)
ax.tick_params(axis="both", which="minor", labelsize=7)

fig.supxlabel('time', weight="bold", fontsize=7)
fig.supylabel('abundance', weight="bold", fontsize=7)

#ax.plot(gLV_community.ODE_sols[0].t,
#         gLV_community.ODE_sols[0].y.T)

plt.savefig("C:/Users/jamil/Documents/PhD/Figures/resource_diversity_stability/gLV_edge_of_chaos.png",
            bbox_inches='tight')
plt.show()

# %%


def sample(S, M,
           mu_c, sigma_c,
           mu_y, sigma_y):
    
    target = mu_y * mu_c
    y = mu_y + sigma_y*np.random.randn(S, M)
    c_uncon = mu_c/M + (sigma_c/np.sqrt(M))*np.random.randn(S, M)
    
    g_uncon = y * c_uncon
    
    c_con1 = c_uncon + \
            (g_uncon/c_uncon) * (target - g_uncon.sum(1, keepdims=True)) /  \
                                ((g_uncon/c_uncon)**2).sum(1, keepdims=True)
                                
    c_con2 = c_uncon + y * (target - g_uncon.sum(1, keepdims=True)) / (y**2).sum(1, keepdims=True)
    
    g_con = y * c_con1
    
    return dict(c_uncon = c_uncon,
                c_con_est1 = c_con1,
                c_con_est2 = c_con2,
                g_uncon = g_uncon,
                g_con = g_con)

for M in [50, 250]:
    
    sample_M = sample(M, M,
                      130.0, 3.0,
                      1.0, 0.13)

    rho_con1 = np.corrcoef(sample_M["g_con"].ravel(),
                          sample_M["c_con_est1"].ravel())[0, 1]
    
    rho_con2 = np.corrcoef(sample_M["g_con"].ravel(),
                          sample_M["c_con_est2"].ravel())[0, 1]
    
    rho_unc = np.corrcoef(sample_M["g_uncon"].ravel(),
                          sample_M["c_uncon"].ravel())[0, 1]
    
    print("constrained 1 = ", rho_con1, "\n",
          "constrained 2 = ", rho_con2, "\n",
          "unconstrained = ", rho_unc)

# %%

M = 50
mu_c = 130
sigma_c = 3
mu_y = 1
sigma_y = 0.13
d = 1
b = 1

community = Consumer_Resource_Model("Self-limiting resource supply",
                                    M-1,
                                    M)

community.growth_consumption_rates('growth function of consumption',
                                   mu_c = mu_c/M,
                                   sigma_c = sigma_c/np.sqrt(M),
                                   mu_g = mu_y,
                                   sigma_g = sigma_y,
                                   fixed_budget = mu_y*mu_c)
community.model_specific_rates(death_method = "constant",
                               death_args = {'d' : 1},
                               resource_growth_method = "constant",
                               resource_growth_args = {'b' : 1})

community.simulate_community(5000, 1)


plt.plot(community.ODE_sols[0].t,
         community.ODE_sols[0].y[:M, :].T)
plt.show()

# %%

M = 50
mu_c = 130
sigma_c = 3
mu_y = 1
sigma_y = 0.0
d = 1
b = 1

community = Consumer_Resource_Model("Self-limiting resource supply",
                                    M-1,
                                    M)

community.growth_consumption_rates('growth function of consumption',
                                   mu_c = mu_c/M,
                                   sigma_c = sigma_c/np.sqrt(M),
                                   mu_g = mu_y,
                                   sigma_g = sigma_y,
                                   all_positive = True)
community.model_specific_rates(death_method = "constant",
                               death_args = {'d' : 1},
                               resource_growth_method = "constant",
                               resource_growth_args = {'b' : 1})

community.simulate_community(5000, 1)


plt.plot(community.ODE_sols[0].t,
         community.ODE_sols[0].y[:M, :].T)
plt.show()

# %%

M = 350
mu_c = 50
sigma_c = 1.2
mu_y = 1
sigma_y = 0.13
d = 1
b = 1

community = Consumer_Resource_Model("Self-limiting resource supply, generic resource type",
                                    M-1,
                                    M)

community.growth_consumption_rates('growth function of consumption',
                                   mu_c = mu_c/M,
                                   sigma_c = sigma_c/np.sqrt(M),
                                   mu_g = mu_y,
                                   sigma_g = sigma_y)
community.model_specific_rates(death_method = "constant",
                               death_args = {'d' : 1},
                               resource_growth_method = "constant",
                               resource_growth_args = {'b' : 1},
                               resource_exponent = 0.7)
community.simulate_community(7000, 1)

community.lyapunov_exponent = max_le(community,
                                     community.ODE_sols[0].y[:, -1],
                                     T = 1000,
                                     perturbation = 1e-6)

plt.plot(community.ODE_sols[0].t[50:],
         community.ODE_sols[0].y[:M, 50:].T)
plt.show()

# %%

M = 10
mu_c = 20
sigma_c = 5
mu_y = 0.8
sigma_y = 0.2
d = 1
b = 1

consumption = np.random.uniform((mu_c/M) - np.sqrt(3)*(sigma_c/np.sqrt(M)),
                                (mu_c/M) + np.sqrt(3)*(sigma_c/np.sqrt(M)),
                                (M, M))
consumption[consumption < 0] = 0

yield_conversion = np.random.uniform(mu_y - np.sqrt(3)*sigma_y,
                                     mu_y + np.sqrt(3)*sigma_y,
                                     (M, M))

growth = yield_conversion * consumption.T

print(pearsonr(growth.flatten(), consumption.T.flatten()))

community = Consumer_Resource_Model("Self-limiting resource supply",
                                    M,
                                    M)

community.growth_consumption_rates('growth function of consumption',
                                   mu_c = mu_c/M,
                                   sigma_c = sigma_c/np.sqrt(M),
                                   mu_g = mu_y,
                                   sigma_g = sigma_y,
                                   consumption = consumption,
                                   growth = growth)
community.model_specific_rates(death_method = "constant",
                               death_args = {'d' : 1},
                               resource_growth_method = "constant",
                               resource_growth_args = {'b' : 1})

community.simulate_community(10000, 1)


plt.plot(community.ODE_sols[0].t,
         community.ODE_sols[0].y[:M, :].T)
plt.show()

# %%

def uniform_from_unit(u, mean, sd):
    
    half_width = np.sqrt(3.0) * sd
    
    dist = mean + half_width * (2.0 * u - 1.0)
    dist[dist < 0.0] = 0.0
    
    return dist
 
 
def correlated_uniform_rates(S, M, mu_c, sigma_c, mu_g, sigma_g, rho,
                             rng=None, require_positive=True):
    """
    Return (c, g), each an S x M array, with uniform marginals and
    corr(c_ia, g_ia) = rho for every (i, a). Different (i, a) pairs
    are independent.
    """
    if not -1.0 <= rho <= 1.0:
        raise ValueError("rho must lie in [-1, 1]")
    rng = np.random.default_rng(rng)
 
    # Gaussian correlation that yields Pearson correlation rho for the uniforms
    r = 2.0 * np.sin(np.pi * rho / 6.0)
 
    z1 = rng.standard_normal((S, M))
    z2 = r * z1 + np.sqrt(1.0 - r**2) * rng.standard_normal((S, M))
    u1, u2 = norm.cdf(z1), norm.cdf(z2)
 
    c = uniform_from_unit(u1, mu_c / M, sigma_c / np.sqrt(M))
    g = uniform_from_unit(u2, mu_g / M, sigma_g / np.sqrt(M))
    
    return c.T, g

M = 150
mu = 20
sigma = 5
rho = 0.7
d = 1
b = 1

consumption, growth = correlated_uniform_rates(M, M,
                                               mu,
                                               sigma,
                                               mu,
                                               sigma,
                                               rho)

print(pearsonr(growth.flatten(), consumption.T.flatten()))

community = Consumer_Resource_Model("Self-limiting resource supply",
                                    M,
                                    M)

community.growth_consumption_rates('coupled by rho',
                                   mu_c = mu/M,
                                   sigma_c = sigma/np.sqrt(M),
                                   mu_g = mu/M,
                                   sigma_g = sigma/np.sqrt(M),
                                   consumption = consumption,
                                   growth = growth)
community.model_specific_rates(death_method = "constant",
                               death_args = {'d' : 1},
                               resource_growth_method = "constant",
                               resource_growth_args = {'b' : 1})

community.simulate_community(5000, 1)


plt.plot(community.ODE_sols[0].t,
         community.ODE_sols[0].y[:M, :].T)
plt.show()

# %%

#M = 250
#mu_c = 6.575
#sigma_c = 0.5788
#mu_y = 0.8132
#sigma_y = 1.317
M = 50
mu_c = 50
sigma_c = 30
mu_y = 1
sigma_y = 0.6
#mu_y = 1
#sigma_y = 0.3

community = Consumer_Resource_Model("Externally-supplied resources",
                                    M,
                                    M)

community.growth_consumption_rates('growth function of consumption',
                                   mu_c = mu_c/M,
                                   sigma_c = sigma_c/np.sqrt(M),
                                   mu_g = mu_y,
                                   sigma_g = sigma_y)

rho = pearsonr(community.growth.flatten(),
               community.consumption.T.flatten())[0]
print(rho**2, "\n",
      (1/np.sqrt(1 + ((sigma_y/mu_y)**2 * (1 + (mu_c**2 / (sigma_c**2 * M))))))**2)

community.model_specific_rates(influx_method = "constant",
                               influx_args = {'b' : 10**(-3.5)},
                               outflux_method = "constant",
                               outflux_args = {'o' : 10**(-3.5)})

community.simulate_community(7000, 1)
community.calculate_community_properties()
surviving_consumers, _ = np.histogram(community.ODE_sols[0].y[:M, -1],
                                      M)
print(rho**2 - (M - surviving_consumers[0])/M)
print(rho**2 - community.species_survival_fraction[0]/community.resource_survival_fraction[0])

lyapunov_exponent = max_le(community,
                           community.ODE_sols[0].y[:, -1],
                           T = 1000,
                           perturbation = 1e-6)

# %%
fig, ax = plt.subplots(1, 1)

plt.plot(community.ODE_sols[0].t,
         community.ODE_sols[0].y[:M, :].T)
#plt.ylim([0, 0.04])
plt.show()
    

# %%

community.simulate_community(100000, 1)
plt.plot(community.ODE_sols[0].t,
         community.ODE_sols[0].y[:M, :].T)
plt.show()

recalc_lyapunov_exponent = max_le(community,
                                  community.ODE_sols[0].y[:, -1],
                                  T = 1000,
                                  perturbation = 1e-6)
print(recalc_lyapunov_exponent)