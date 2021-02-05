#!/bin/bash
#
#SBATCH --job-name=norbert_ner
#SBATCH --account=nn9851k
#SBATCH --partition=accel
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=3:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --ntasks=8

module purge
module use -a /cluster/projects/nn9851k/software/easybuild/install/modules/all/
module load NLPL-transformers/4.2.2-gomkl-2019b-Python-3.7.4

MODEL_NAME=${1} # ltgoslo/norbert is valid
DATASET=${2}  # path to dataset

echo $MODEL_NAME
echo $DATASET

PYTHONHASHSEED=0 python3 ner.py --model_name ${MODEL_NAME} --dataset ${DATASET}
