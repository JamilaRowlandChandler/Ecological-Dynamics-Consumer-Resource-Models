# -*- coding: utf-8 -*-
"""
Created on Fri Jan 30 11:27:13 2026

@author: jamil
"""

import numpy as np
from scipy.integrate import solve_ivp
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import least_squares

# %%

def solve_least_squares(S,
                        mu_r, sigma_r,
                        mu_A, sigma_A, rho_A,):
    
    x0 = [0.5, 0.01, 0.5, 2]
    bounds = [[0, 0, 0, 0], [1, 1e15, 1e15, 1e15]]
    x_scale = [1, 0.1, 0.1, 1]
    
    sol = least_squares(fun_ls, x0,
                        bounds = bounds,
                        x_scale = x_scale,
                        args = (S,
                                mu_r, sigma_r,
                                mu_A, sigma_A, rho_A,),
                        xtol = 1e-15, ftol = 1e-15)
    
    return sol

# %%

def fun_ls(sces,
           S,
           mu_r, sigma_r,
           mu_A, sigma_A, rho_A):
    
    Z_N, N_i  = generate_abundances(S,
                                    mu_r, sigma_r,
                                    mu_A, sigma_A, rho_A,
                                    sces[1],
                                    sces[2],
                                    sces[3])

    eq_sces = distribution_properties(S, sigma_r, sigma_A, 
                                      Z_N, N_i)
    
    return sces - eq_sces

# %%

def solve_dmft(S,
               mu_r, sigma_r,
               mu_A, sigma_A, rho_A,
               threshold = -30,
               maxiter = 10000):
    
    sces_iminus1 = initialise_sces()
    
    fit, sces_i = make_dist_check_fit(S,
                                      mu_r, sigma_r,
                                      mu_A, sigma_A, rho_A,
                                      sces_iminus1)
    
    for _ in range(maxiter):
        
        sces_upd = update_sce(sces_iminus1, sces_i, a = 0.1)
        sces_iminus1 = sces_upd
        
        fit, sces_i = make_dist_check_fit(S,
                                          mu_r, sigma_r,
                                          mu_A, sigma_A, rho_A,
                                          sces_iminus1)
        
        if fit <= threshold:
            
            print("Solver converged")
            
            return sces_i
        
    print("Solver did not converge.")
    
    return {"log10(loss)" : fit, "sces" : sces_i}

# %%

def initialise_sces():
    
    phi_N = np.random.uniform(0, 1)
    N_mean = np.random.uniform(0, 0.1)
    N_fluct = np.random.uniform(0, 0.1)
    v_N = np.random.uniform(0, 0.5)
    
    return np.array([phi_N, N_mean, N_fluct, v_N])

# %%

def make_dist_check_fit(S,
                        mu_r, sigma_r,
                        mu_A, sigma_A, rho_A,
                        sces_iminus1):
    
    Z_N, N_i  = generate_abundances(S,
                                    mu_r, sigma_r,
                                    mu_A, sigma_A, rho_A,
                                    sces_iminus1[1],
                                    sces_iminus1[2],
                                    sces_iminus1[3])

    sces_i = distribution_properties(S, sigma_r, sigma_A, 
                                     Z_N, N_i)
    
    fit = condition(sces_iminus1, sces_i)
    
    return fit, sces_i

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
    
    N_i[N_i < 1e-4] = 0
    
    return Z_N, N_i
    
# %% 

def distribution_properties(S, sigma_r, sigma_A,
                            Z_N, N_i):
    
    phi_N = np.count_nonzero(N_i)/S
    
    N_mean = np.mean(N_i)
    
    N_fluct = np.mean(N_i**2)
    
    v_N = suseptibility(sigma_r, sigma_A,
                        N_fluct, Z_N, N_i)
    
    return np.array([phi_N, N_mean, N_fluct, v_N])

# %%

def suseptibility(sigma_r, sigma_A,
                  N_fluct, Z_N, N_i):
    
    v = (1/np.sqrt(sigma_r**2 + sigma_A**2 * N_fluct)) * np.mean(N_i * Z_N)
    
    return v

# %%

def condition(dist_simulation, 
              dist_estimated):
    
    return np.log10(np.sum(dist_simulation - dist_estimated)**2)

# %%

def update_sce(X_old, X_new, a = 0.3):
    
    return a * X_old + (1 - a) * X_new

# %%

def gLV_simulations(reps, 
                    S, 
                    mu_r, sigma_r,
                    mu_A, sigma_A, rho_A):

    def gLV_community(S, 
                      mu_r, sigma_r,
                      mu_A, sigma_A, rho_A):
        
        r = gaussian_parameters(mu_r, sigma_r, (S, ))
        
        A = correlated_interactions(S, mu_A, sigma_A, rho_A)
        
        N0 = initial_species_abundances(S)
        
        sol = solve_ivp(gLV_dynamics,
                        [0, 4000],
                        N0,
                        method = 'LSODA',
                        args = (r, A))
        
        N_ss = sol.y[:, -1]
        phi_N = np.sum(N_ss > 1e-4)/S
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
    
    multi_communities = [gLV_community(S, mu_r, sigma_r, mu_A, sigma_A, rho_A)
                         for _ in range(reps)]
    
    simulations = [comm[0] for comm in multi_communities]
    summary_df = pd.DataFrame([comm[1] for comm in multi_communities])
    
    return simulations, summary_df

# %%

def gaussian_parameters(mu, sigma, dims):
    
    return mu + sigma * np.random.randn(*dims)

# %% 

def correlated_interactions(S, mu, sigma, rho_A):
    
    A_uncorr = np.random.randn(S, S)
    
    A_ij = np.tril(A_uncorr)
    A_ji_uncorr = np.triu(A_uncorr)
    
    A_ji_corr = rho_A*A_ij.T + np.sqrt(1 - rho_A**2)*A_ji_uncorr
    
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

def main():
    
    S = 500
    mu_r = 1.
    sigma_r = 0.
    mu_A = 40.
    sigma_A = 0.5
    rho_A = 1

    simulations, df_simulations = gLV_simulations(5,
                                                  S,
                                                  mu_r, sigma_r,
                                                  mu_A/S,
                                                  sigma_A/np.sqrt(S),
                                                  rho_A)

    #sces = solve_least_squares(S, mu_r, sigma_r, mu_A, sigma_A, rho_A)

    sces = solve_dmft(S, mu_r, sigma_r, mu_A, sigma_A, rho_A,
                      maxiter = 100000)
    
    print(df_simulations[['phi_N', 'N_mean', 'N_fluct']], "\n",
          #sces.x, "\n",
          #np.log10(sces.cost), "\n",
          #sces.success)
          sces)
    
if __name__ == '__main__':
    
    main()
    
