#!/bin/bash
#SBATCH --job-name=deduplication
#SBATCH --account=project_465000144
#SBATCH --time=23:00:00
#SBATCH --cpus-per-task=6
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --partition=small

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

set -o errexit  # Recommended for easier debugging

module purge
module load LUMI/22.08
module load cray-python


export PYTHONUSERBASE='/projappl/project_465000157/.local'
export PATH=$PYTHONUSERBASE/bin:$PATH
export PYTHONPATH=$PYTHONUSERBASE/lib/python3.9/site-packages:$PYTHONPATH

INCORPUS=${1} # Can be a file or a directory with *.gz files
OUTCORPUS=${2}  # Better place it in a separate directory to avoid loops
LOGFILE=${3}

srun python3 deduplication.py -c ${INCORPUS} -l ${LOGFILE} | gzip > ${OUTCORPUS}