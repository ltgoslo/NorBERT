#!/bin/bash

#SBATCH --job-name=CONVERT
#SBATCH --account=nn10029k
#SBATCH --time=00:15:00
#SBATCH --mem-per-cpu=1750
#SBATCH --cpus-per-task=7
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=normal
#SBATCH --output=/cluster/work/projects/nn9851k/mariiaf/hplt/logs/convert-%j.out

set -o errexit  # Exit the script on any error
set -o nounset  # Treat any unset variables as an error

SIF="/cluster/projects/nn9851k/containers/pytorch2.7_cu2.9_py3.12_amd_nlpl.sif"

INPUT_PATH=${1}
OUTPUT_DIR=${2}
ALL_CHECKPOINTS=${3} # 1 if convert all checkpoints
LANGS=${@:4}
for LANG in $LANGS
  do
    if [ $ALL_CHECKPOINTS != "1" ]; then
      srun apptainer exec -B /cluster/projects/:/cluster/projects/,/cluster/work/projects/:/cluster/work/projects/ $SIF python3 convert_to_hf.py --input_model_directory $INPUT_PATH --output_model_directory $OUTPUT_DIR --language $LANG
    else
      srun apptainer exec -B /cluster/projects/:/cluster/projects/,/cluster/work/projects/:/cluster/work/projects/ $SIF python3 convert_to_hf.py --input_model_directory $INPUT_PATH --output_model_directory $OUTPUT_DIR --language $LANG --all_checkpoints
    fi
  done
