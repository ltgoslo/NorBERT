#!/bin/bash
#SBATCH --job-name=segmenting
#SBATCH --nodes=1
#SBATCH --time=30:00:00
#SBATCH --partition=accel
#SBATCH --gres=gpu:1
#SBATCH --account=nn9851k
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=6G

set -o errexit # Make bash exit on any error
set -o nounset # Treat unset variables as errors

module use -a /cluster/shared/nlpl/software/eb/etc/all
module purge

module load nlpl-stanza/1.1.1-gomkl-2019b-Python-3.7.4

echo "Input file: ${1}"
echo "Language: ${2}"

zcat ${1} | python3 segmenter2.py ${2} | gzip > ${1}_segmented.gz
