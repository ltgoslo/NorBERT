#!/bin/bash
#
#SBATCH --job-name=norelmo_sentiment
#SBATCH --account=nn9851k
#SBATCH --partition=accel
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=2:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --ntasks=8

module purge
module use -a /cluster/projects/nn9851k/software/easybuild/install/modules/all/
module load NLPL-simple_elmo/0.6.0-gomkl-2019b-Python-3.7.4

DATA=${1}  # ../data/sentiment/no/
ELMO=${2} # /cluster/projects/nn9851k/andreku/norlm/norelmo30
METHOD=${3} # bow or lstm

echo $DATA
echo $ELMO
echo $METHOD

PYTHONHASHSEED=0 python3 elmo_binary_sentiment.py --input ${DATA} --elmo ${ELMO} --method ${METHOD} --elmo_layers top