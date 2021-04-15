#!/bin/bash
#SBATCH --job-name=sentencepiece
#SBATCH --mail-type=FAIL
#SBATCH --account=nn9851k
#SBATCH --time=2:10:00
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=8G
#SBATCH --ntasks=16

source ${HOME}/.bashrc

set -o errexit  # Recommended for easier debugging

## Load your modules
module purge   # Recommended for reproducibility
module use -a /cluster/projects/nn9851k/software/easybuild/install/modules/all/
module load SentencePiece/0.1.94-gomkl-2019b-Python-3.7.4

echo "Corpus: ${1}"
echo "Output file: ${2}"

spm_train --input=${1} --model_prefix=${2} --vocab_size=30000 --model_type=bpe --character_coverage=0.999 --split_by_number=false

