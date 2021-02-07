#!/bin/bash
#
#SBATCH --job-name=norbert_pos
#SBATCH --account=nn9851k
#SBATCH --partition=accel
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=5:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --ntasks=8

module purge
module use -a /cluster/projects/nn9851k/software/easybuild/install/modules/all/
module load NLPL-transformers/4.2.2-gomkl-2019b-Python-3.7.4

MODEL_NAME=${1}
SHORT_MODEL_NAME=${2} # ltgoslo/norbert is valid
LANG=${3}

echo $MODEL_NAME
echo $SHORT_MODEL_NAME
echo $LANG

PYTHONHASHSEED=0 python3 pos_finetuning.py --model_name "$MODEL_NAME" --short_model_name "$SHORT_MODEL_NAME" --training_language $LANG --epochs 20
