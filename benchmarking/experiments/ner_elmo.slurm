#!/bin/bash
#SBATCH --job-name=bert_ner
#SBATCH --mail-type=FAIL
#SBATCH --account=nn9851k
#SBATCH --partition=accel    # To use the accelerator nodes
#SBATCH --gres=gpu:1         # To specify how many GPUs to use (on one node)
#SBATCH --time=1:00:00      # Max walltime is 14 days.
#SBATCH --mem-per-cpu=4G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6

set -o errexit  # Recommended for easier debugging

## Load your modules
module use -a /cluster/projects/nn9851k/software/easybuild/install/modules/all/
module purge   # Recommended for reproducibility
module load NLPL-nlptools/2021.01-gomkl-2019b-Python-3.7.4
module load NLPL-simple_elmo/0.6.0-gomkl-2019b-Python-3.7.4

python3 ner_elmo.py --train ${1} --dev ${2} --test ${3} --elmo ${4} --name ${5}
