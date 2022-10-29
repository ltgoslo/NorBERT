#!/bin/bash
#SBATCH --job-name=deduplication
#SBATCH --account=project_465000144
#SBATCH --time=23:00:00
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --mem=192G
#SBATCH --partition=small

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

set -o errexit  # Recommended for easier debugging

module purge
module load LUMI/22.08
module load cray-python


export PYTHONUSERBASE='/projappl/project_465000157/.local'
export PATH=$PYTHONUSERBASE/bin:$PATH
export PYTHONPATH=$PYTHONUSERBASE/lib/python3.9/site-packages:$PYTHONPATH

REFCORPUS=${1} # Reference corpus. Can be a file or a directory with *.gz files
CORPUS=${2}  # Corpus to de-dupe
LOGFILE=${3}

srun python3 cross_dedup.py -c1 ${REFCORPUS} -c2 ${CORPUS} -l ${LOGFILE}

