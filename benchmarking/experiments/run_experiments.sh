#!/bin/bash

sbatch pos_slurm_script.sh norbert ../checkpoints/norbert3/
sbatch pos_slurm_script.sh mbert ../checkpoints/mbert/
sbatch sent_slurm_script.sh norbert ../checkpoints/norbert3/
sbatch sent_slurm_script.sh mbert ../checkpoints/mbert/
