#!/bin/bash
# Grid Engine (SGE) array job: runs M_vs_mu_c_external_supply.py, with the (M, mu_c)
# parameter sets split between the array tasks. There are 10 parameter sets
# (M = 25, 50, ..., 250), so "-t 1-10" gives one parameter set per task. With fewer
# tasks (e.g. "-t 1-5"), each task runs several parameter sets.
#
# Submit from this directory with:  qsub M_vs_mu_c_external_supply_array.sh
# Each task writes its output to M_vs_mu_c_external.o<job ID>.<task ID> here.

#$ -S /bin/bash
#$ -l h_rt=2:00:00               # time limit per task (hours:minutes:seconds)
#$ -l h_vmem=4G,tmem=4G          # memory per task (both are needed, and must match)
#$ -cwd                          # run in the directory the job was submitted from
#$ -j y                          # write errors into the same output file
#$ -N M_vs_mu_c_external         # job name
#$ -t 1-10                       # array tasks 1, ..., 10

hostname
date

# EDIT: the Python version to use (3.11 or newer). To see what's available, run
#   ls /share/apps/source_files/python/
PYTHON_SOURCE=/share/apps/source_files/python/python-3.X.Y.source
source "$PYTHON_SOURCE" || { echo "Can't find $PYTHON_SOURCE - edit PYTHON_SOURCE in this script"; exit 1; }

# where to save the results (the script adds
#   resource_diversity_stability/simulations/<subdirectory> to this)
export DATA_DIRECTORY="$HOME/PhD/Data"

# one CPU core per task, and no screen for matplotlib
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLBACKEND=Agg

python3 -u M_vs_mu_c_external_supply.py
