#!/bin/bash
# Grid Engine (SGE) job: runs the DIRECT then Nelder-Mead search in
# M_vs_mu_c_external_supply_opt_param.py (a single, serial job).
#
# Submit from this directory with:  qsub opt_param_job.sh
# Output goes to opt_param.o<job ID>, and the results to opt_param_results.pkl,
# both in this directory.

#$ -S /bin/bash
#$ -l h_rt=12:00:00              # time limit (hours:minutes:seconds)
#$ -l h_vmem=2G,tmem=2G          # memory (both are needed, and must match)
#$ -cwd                          # run in the directory the job was submitted from
#$ -j y                          # write errors into the same output file
#$ -N opt_param                  # job name

hostname
date

# EDIT: the Python version to use (3.11 or newer). To see what's available, run
#   ls /share/apps/source_files/python/
PYTHON_SOURCE=/share/apps/source_files/python/python-3.X.Y.source
source "$PYTHON_SOURCE" || { echo "Can't find $PYTHON_SOURCE - edit PYTHON_SOURCE in this script"; exit 1; }

# one CPU core, and no screen for matplotlib
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLBACKEND=Agg

python3 -u M_vs_mu_c_external_supply_opt_param.py
