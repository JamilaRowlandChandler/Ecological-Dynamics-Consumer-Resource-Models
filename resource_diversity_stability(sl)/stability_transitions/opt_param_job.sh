#!/bin/bash
#SBATCH --job-name=opt_param
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=12:00:00
#SBATCH --output=opt_param_%j.out

# Runs the DIRECT then Nelder-Mead search in M_vs_mu_c_external_supply_opt_param.py.
# Submit from this directory with:  sbatch opt_param_job.sh
# Results are saved to opt_param_results.pkl in this directory.

# load your Python environment here (needs numpy, scipy, matplotlib), e.g.
# module load anaconda3
# source activate my_env

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLBACKEND=Agg

python -u M_vs_mu_c_external_supply_opt_param.py
