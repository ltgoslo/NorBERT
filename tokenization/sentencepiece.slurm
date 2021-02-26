#!/bin/bash
#SBATCH --job-name=sentencepiece
#SBATCH --mail-type=FAIL
#SBATCH --account=nn9851k
#SBATCH --time=2:10:00
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=8G
#SBATCH --ntasks=8

source ${HOME}/.bashrc

set -o errexit  # Recommended for easier debugging

## Load your modules
module purge   # Recommended for reproducibility
module use -a /cluster/projects/nn9851k/software/easybuild/install/modules/all/
module load NLPL-tokenizers/0.10.1-gomkl-2019b-Python-3.7.4
module load NLPL-nlptools/2021.01-gomkl-2019b-Python-3.7.4

echo "Corpus: ${1}"
echo "Output file: ${2}"

python3 spiece_tokenizer.py ${1} ${2}
python3 sent2wordpiece.py ${2}.json -o ${2}.txt
