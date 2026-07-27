# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 2026

@author: jamil

Individual (unaveraged) species/resource trajectory comparison, on the same
M=25 sparse connected network used throughout this investigation (gamma
network_method, seed=900201, mean=0.04/variance=0.0014, dominant resource
index 12), 'reversible' saturation kinetics, d=0.5, o in {0.3, 1.0} -
comparing the unlagged MP_CRM against MP_CRM_Lagged (mass-conserving
growth/production pool formulation - see models.py's MP_CRM_Lagged
docstring for why the pool formulation replaced an earlier rate-lag design
that went unstable at any tau large enough to show a visible delay).

tau_growth = tau_production = 10 (the same tau that caused the abandoned
rate-lag version to blow up outright) - the pool formulation is expected to
remain bounded (finite pools can't compound without limit the way a lagged
RATE could) while still showing a visible lag/oscillation relative to the
unlagged model.
"""

import numpy as np
import sys
import os
import matplotlib.pyplot as plt

abspath = os.path.abspath(__file__)
file_directory_name = os.path.dirname(abspath)
os.chdir(file_directory_name)

sys.path.insert(0, file_directory_name)
sys.path.insert(0, 'C:/Users/jamil/Documents/PhD/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models/consumer_resource_modules')

from network_diagnostics import sample_connected_gamma_network, check_connectivity
from models import Consumer_Resource_Model

# %%

M, S = 25, 50
mu_C, sigma_C = 40, 1.6
mu_c, sigma_c = mu_C / M, sigma_C / np.sqrt(M)
K_m = 1e-2
K_m_method, K_m_args = 'uniform', {'low': 0.001, 'high': 0.1}
b, p = -0.001, 1

GAMMA_SEED = 900201
GAMMA_MEAN, GAMMA_VARIANCE = 0.04, 0.0014

d = 0.5
t_end = 100
o_values = [0.3, 1.0]
tau_growth = tau_production = 10
rate_seed = 950000

w, adjacency = sample_connected_gamma_network(M, GAMMA_MEAN, GAMMA_VARIANCE, GAMMA_SEED)
dominant = int(np.argmax(w))
check_connectivity(w, adjacency, verbose=True)


def _build_community(model_str, o_val):

    o = np.zeros(M)
    o[dominant] = o_val
    np.random.seed(rate_seed)

    community = Consumer_Resource_Model(model_str, pool_sizes=(M, S))
    community.growth_consumption_rates('growth function of consumption',
                                       mu_c=mu_c, sigma_c=sigma_c, mu_g=1, sigma_g=0)
    community.model_specific_rates(death_args={'d': d},
                                   influx_method='user-supplied', influx_args={'o': o},
                                   resource_growth_args={'b': b}, resource_inhibition_args={'A': 0})
    community.metabolic_network(energies=w, adjacency=adjacency, network_method='step',
                                resource_conversions={'p_s': 1}, growth_saturation=True,
                                saturation_kinetics='reversible', K_m=K_m,
                                K_m_method=K_m_method, K_m_args=K_m_args,
                                production_method='constant', production_args={'p': p})
    if model_str == 'Metabolic pathways, lagged':
        community.set_lag(tau_growth=tau_growth, tau_production=tau_production)

    return community


results = {}

for o_val in o_values:
    for model_str, label in [('Metabolic pathways', 'unlagged'),
                              ('Metabolic pathways, lagged', 'lagged')]:
        community = _build_community(model_str, o_val)
        community.simulate_community(t_end=t_end, no_init_cond=1)
        sol = community.ODE_sols[0]
        print(f"o={o_val}, {label}: status={sol.status}, message={sol.message!r}, "
             f"max|y|={np.max(np.abs(sol.y)):.3e}", flush=True)
        results[(o_val, label)] = sol

# %%

fig, axes = plt.subplots(4, 2, figsize=(11, 14), sharex='col')

for col, o_val in enumerate(o_values):
    for row, label in enumerate(['unlagged', 'lagged']):
        sol = results[(o_val, label)]
        ax_R = axes[2 * row, col]
        ax_N = axes[2 * row + 1, col]

        for a in range(M):
            ax_R.plot(sol.t, sol.y[a, :], lw=0.7, alpha=0.7)
        for i in range(S):
            ax_N.plot(sol.t, sol.y[M + i, :], lw=0.7, alpha=0.7)

        ax_R.set_title(f"o={o_val}, {label} - resources")
        ax_N.set_title(f"o={o_val}, {label} - species")
        ax_R.set_ylabel('R')
        ax_N.set_ylabel('N')

axes[-1, 0].set_xlabel('t')
axes[-1, 1].set_xlabel('t')

fig.suptitle(f"MP_CRM vs MP_CRM_Lagged (pool, tau_g=tau_p={tau_growth}), "
            f"M=25 connected network seed={GAMMA_SEED}, d={d}")
fig.tight_layout()
out_path = os.path.join(file_directory_name, 'M25_sparse_lagged_pool_trajectories.png')
fig.savefig(out_path, dpi=150)
print(f"Saved figure to {out_path}")
