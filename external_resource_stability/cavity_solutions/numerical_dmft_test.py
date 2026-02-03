# -*- coding: utf-8 -*-
"""
Created on Fri Jan 30 11:27:13 2026

@author: jamil
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import truncnorm
import pandas as pd
from matplotlib import pyplot as plt

# %%

def condition(dist_simulation, 
              dist_estimated):
    
    return dist_simulation - dist_estimated

# %%

def generate_abundances(S,
                        mu_r, sigma_r,
                        mu_A, sigma_A, rho_A, 
                        N_mean, N_fluct, v):
    
    mu_gN = mu_r - mu_A * N_mean
    sigma_gN = np.sqrt(sigma_r**2 + sigma_A**2 * N_fluct)
    feedback = 1 - sigma_A**2 * rho_A * v
    
    Z_N = np.random.randn(S)
    
    N_i = (mu_gN + sigma_gN * Z_N)/feedback
    
    N_i[N_i < 0] = 0
    
    return Z_N, N_i
    
# %% 

def distribution_properties(Z_N, N_i):
    
    phi_N = np.count_nonzero(N_i)
    
    N_mean = np.mean(N_i)
    
    N_fluct = np.mean(N_i**2)
    
    v_N = suseptibility(N_fluct, Z_N, N_i)
    
    return phi_N, N_mean, N_fluct, v_N

# %%

def update_sce(X_old, X_new, a = 0.3):
    
    return a * X_old + (1 - a) * X_new

# %%

def suseptibility(sigma_r, sigma_A,
                  N_fluct, Z_N, N_i):
    
    v = (1/np.sqrt(sigma_r**2 + sigma_A**2 * N_fluct)) * np.mean(N_i * Z_N)
    
    return v

# %%

def gLV_simulations(reps, 
                    S, 
                    mu_r, sigma_r,
                    mu_A, sigma_A, rho):

    def gLV_community(S, 
                      mu_r, sigma_r,
                      mu_A, sigma_A, rho):
        
        r = gaussian_parameters(mu_r, sigma_r, (S, ))
        
        A = correlated_interactions(S, mu_A, sigma_A, rho)
        
        N0 = initial_species_abundances(S)
        
        sol = solve_ivp(gLV_dynamics,
                        [0, 1000],
                        N0,
                        method = 'LSODA',
                        args = (r, A))
        
        N_ss = sol.y[:, -1]
        phi_N = np.count_nonzero(N_ss)
        N_mean = np.mean(N_ss)
        N_fluct = np.mean(N_ss)
        
        return sol, dict(S = S,
                         mu_r = mu_r,
                         sigma_r = sigma_r,
                         mu_A = mu_A,
                         sigma_A = sigma_A,
                         phi_N = phi_N, 
                         N_mean = N_mean,
                         N_fluct = N_fluct)
    
    multi_communities = [gLV_community(S, mu_r, sigma_r, mu_A, sigma_A)
                         for _ in range(reps)]
    
    simulations = [comm[0] for comm in multi_communities]
    summary_df = pd.DataFrame([comm[1] for comm in multi_communities])
    
    return simulations, summary_df

# %%

def gaussian_parameters(mu, sigma, dims):
    
    return mu + sigma * np.random.randn(*dims)

# %% 

def correlated_interactions(S, mu, sigma, rho):
    
    A_uncorr = np.random.randn(S, S)
    
    A_ij = np.tril(A_uncorr)
    A_ji_uncorr = np.triu(A_uncorr)
    
    A_ji_corr = rho*A_ij.T + np.sqrt(1 - rho)*A_ji_uncorr
    
    A_corr = mu + sigma*(A_ij + A_ji_corr)
    
    np.fill_diagonal(A_corr, 1)
    
    return A_corr
    
# %%

def initial_species_abundances(S):
    
    return np.random.uniform(1e-8, 2/S, S)

# %%

def gLV_dynamics(t, N,
                 r, A):
    
    dNdt = N * (r - A @ N)
    
    return dNdt

# %%

simulations, df_simulations = gLV_simulations(5, 200, 1, 0, 20, 1)


