#!/bin/bash
#SBATCH --account=project_465001386
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=48:00:00

set -o errexit  # Exit the script on any error
set -o nounset  # Treat any unset variables as an error


python3 shard.py --subcorpus $1
