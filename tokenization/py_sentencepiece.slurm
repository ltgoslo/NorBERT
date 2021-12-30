#!/bin/bash
#SBATCH --job-name=sentencepiece
#SBATCH --mail-type=FAIL
#SBATCH --account=nn9851k
#SBATCH --partition=bigmem
#SBATCH --time=40:00:00
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=32G
#SBATCH --cpus-per-task=16

source ${HOME}/.bashrc

set -o errexit  # Recommended for easier debugging

## Load your modules
module purge   # Recommended for reproducibility
module use -a /cluster/shared/nlpl/software/eb/etc/all
module load nlpl-gensim/3.8.3-gomkl-2019b-Python-3.7.4
module load sentencepiece/0.1.96-gomkl-2019b-Python-3.7.4

echo "Corpus: ${1}"
echo "Output model prefix: ${2}"
echo "Vocab size: ${3}"

python3 spiece_tokenizer2.py ${1} ${2} ${3}
