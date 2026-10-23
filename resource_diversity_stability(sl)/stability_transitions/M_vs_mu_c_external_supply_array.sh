#!/bin/bash
#SBATCH --job-name=M_vs_mu_c_external
#SBATCH --array=0-9
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=24:00:00
#SBATCH --output=M_vs_mu_c_external_%A_%a.out

# Runs M_vs_mu_c_external_supply.py as an array job: the (M, mu_c) parameter sets
# are split between the array tasks. There are 10 parameter sets (M = 25, 50, ..., 250),
# so --array=0-9 gives one parameter set per task. With fewer tasks (e.g. --array=0-4),
# each task runs several parameter sets.
# Submit from this directory with:  sbatch M_vs_mu_c_external_supply_array.sh

# load your Python environment here (needs numpy, scipy, pandas, matplotlib,
# seaborn, tqdm), e.g.
# module load anaconda3
# source activate my_env

# where to save the results (the script adds
# resource_diversity_stability/simulations/<subdirectory> to this)
export DATA_DIRECTORY="$HOME/PhD/Data"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLBACKEND=Agg

python -u M_vs_mu_c_external_supply.py
