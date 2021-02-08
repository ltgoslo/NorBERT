#!/bin/bash
#
#SBATCH --job-name=norelmo_pos
#SBATCH --account=nn9851k
#SBATCH --partition=accel
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=5:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --ntasks=8

module purge
module use -a /cluster/projects/nn9851k/software/easybuild/install/modules/all/
module load NLPL-simple_elmo/0.6.0-gomkl-2019b-Python-3.7.4

UD=${1}  # ../data/ud/nob
ELMO=${2} # /cluster/projects/nn9851k/andreku/norlm/norelmo30

echo $UD
echo $ELMO

PYTHONHASHSEED=0 python3 elmo_pos.py --input ${UD} --elmo ${ELMO}
