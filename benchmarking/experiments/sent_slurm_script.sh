#!/bin/bash

#SBATCH --job-name=bert_sentiment
#SBATCH --account=nn9851k
#SBATCH --partition=accel
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=5:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --cpus-per-task=4


module purge
module use -a /cluster/shared/nlpl/software/eb/etc/all
module load nlpl-scipy-ecosystem/2021.01-gomkl-2019b-Python-3.7.4
module load nlpl-transformers/4.5.1-gomkl-2019b-Python-3.7.4

MODEL_NAME=${1}
SHORT_MODEL_NAME=${2} # ltgoslo/norbert is valid

echo $MODEL_NAME
echo $SHORT_MODEL_NAME

PYTHONHASHSEED=0 python3 sentiment_finetuning.py --model_name "$MODEL_NAME" --short_model_name "$SHORT_MODEL_NAME" --epochs 20 --use_class_weights
