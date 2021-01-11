#!/bin/bash
#SBATCH --job-name=sentencepiece
#SBATCH --mail-type=FAIL
#SBATCH --account=YOUR_PROJECT_NUMBER
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=4G
#SBATCH --ntasks=8

source ${HOME}/.bashrc

set -o errexit  # Recommended for easier debugging

## Load your modules
module purge   # Recommended for reproducibility
module use -a /cluster/projects/nn9851k/software/easybuild/install/modules/all/
module load NLPL-tokenizers/0.9.4-gomkl-2019b-Python-3.7.4

echo "Corpus: ${1}"
echo "Output file: ${2}"

python3 spiece_tokenizer.py ${1} ${2}
python3 sent2wordpiece.py ${2}.json -o ${2}.txt
