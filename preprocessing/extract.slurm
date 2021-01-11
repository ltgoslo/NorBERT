#!/bin/bash
#SBATCH --job-name=preprocessing
#SBATCH --mail-type=FAIL
#SBATCH --account=YOUR_PROJECT_NUMBER
#SBATCH --time=23:59:00      # Max walltime is 14 days.
#SBATCH --mem-per-cpu=8G

# Definining resource we want to allocate.
#SBATCH --nodes=1
#SBATCH --ntasks=4

# This is used to make checkpoints and logs to readable and writable by other members in the project.
umask 0007

# Clear all modules and load Tensorflow.
module purge   # Recommended for reproducibility
module load Python/3.7.4-GCCcore-8.3.0

./extract_text_from_xml.sh
