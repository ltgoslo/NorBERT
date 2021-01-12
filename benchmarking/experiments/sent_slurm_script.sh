#!/bin/bash
#
#SBATCH --job-name=negscope --account=nn9851k
#SBATCH --output=norbert_sent.out
#SBATCH --partition=accel --gres=gpu:1
#SBATCH -n 1
#SBATCH -t 10:00:00
#SBATCH --mem 8GB
#XSBATCH --mail-type=ALL
#SBATCH --mail-user=jeremycb@ifi.uio.no
#

module purge
source deactivate

module use -a /cluster/projects/nn9851k/software/easybuild/install/modules/all/
module load NLPL-PyTorch/1.6.0-gomkl-2019b-Python-3.7.4
module load NLPL-transformers/4.1.1-gomkl-2019b-Python-3.7.4

MODEL_NAME=$1;shift
SHORT_MODEL_NAME=$1;shift

echo $MODEL_NAME
echo $SHORT_MODEL_NAME

python sentiment_finetuning.py --model_name "$MODEL_NAME" --short_model_name "$SHORT_MODEL_NAME" --use_class_weights
