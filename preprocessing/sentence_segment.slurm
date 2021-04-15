#!/bin/bash
#SBATCH --job-name=segmenting
#SBATCH --nodes=1
#SBATCH --time=20:00:00
#SBATCH --partition=accel
#SBATCH --gres=gpu:1
#SBATCH --account=nn9851k
#SBATCH --ntasks=8
#SBATCH --mem-per-cpu=8G

set -o errexit # Make bash exit on any error
set -o nounset # Treat unset variables as errors

module use -a /cluster/projects/nn9851k/software/easybuild/install/modules/all/
module purge

module load NLPL-stanza/1.1.1-gomkl-2019b-Python-3.7.4

echo "Input file: ${1}"
echo "Language: ${2}"
echo "Output file: ${3}"

zcat ${1} | python3 segmenter.py ${2} | gzip > ${3}
