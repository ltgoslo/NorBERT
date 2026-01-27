#!/bin/bash

#SBATCH --job-name=SANITY
#SBATCH --account=nn10029k
#SBATCH --time=0:15:00
#SBATCH --mem-per-cpu=7G
#SBATCH --cpus-per-task=7
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=normal
#SBATCH --output=sanity-%j.out

set -o errexit  # Exit the script on any error
set -o nounset  # Treat any unset variables as an error

SIF="/cluster/projects/nn9851k/containers/pytorch2.7_cu2.9_py3.12_amd_nlpl.sif"

srun apptainer exec -B /cluster/projects/:/cluster/projects/,/cluster/work/projects/:/cluster/work/projects/ $SIF python3 sanity_check.py

