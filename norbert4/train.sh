#!/bin/bash -e
#SBATCH --job-name=GPT-BERT
#SBATCH --nodes=32
#SBATCH --gpus-per-node=8
#SBATCH --tasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --output="output_%x_%j.txt"
#SBATCH --partition=standard-g
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH --account=project_465001386


export EBU_USER_PREFIX=/project/project_465001925/charpent2.6
module purge
module load LUMI
module load PyTorch/2.6.0-rocm-6.2.4-python-3.12-singularity-20250404


c=fe
MYMASKS="0x${c}000000000000,0x${c}00000000000000,0x${c}0000,0x${c}000000,0x${c},0x${c}00,0x${c}00000000,0x${c}0000000000"


CMD="train.py \
--train_path \
/scratch/project_465001384/dasamuel/large-gpt-bert/norsk_data/tokenized_shards/all_se \
/scratch/project_465001384/dasamuel/large-gpt-bert/norsk_data/tokenized_shards/fineweb_nb \
/scratch/project_465001384/dasamuel/large-gpt-bert/norsk_data/tokenized_shards/fineweb_nn \
/scratch/project_465001384/dasamuel/large-gpt-bert/norsk_data/tokenized_shards/hplt_nb \
/scratch/project_465001384/dasamuel/large-gpt-bert/norsk_data/tokenized_shards/hplt_nn \
/scratch/project_465001384/dasamuel/large-gpt-bert/norsk_data/tokenized_shards/mimir_nb \
/scratch/project_465001384/dasamuel/large-gpt-bert/norsk_data/tokenized_shards/mimir_nn \
/scratch/project_465001384/dasamuel/large-gpt-bert/norsk_data/tokenized_shards/wiki_nb \
/scratch/project_465001384/dasamuel/large-gpt-bert/norsk_data/tokenized_shards/wiki_nn \
/scratch/project_465001384/dasamuel/large-gpt-bert/norsk_data/tokenized_shards/wiki_se \
--dataset_weights 1.0 358.0 21.7 319.0 54.0 39.0 11.3 3.0 2.4 0.05 \
--name NorBERT4_base_7:8_weighted_v5 \
--number_of_tokens 600000000000 \
--max_steps 150000 \
--config_file ./config_base_fix_2.json \
--cooldown_proportion 0.2 \
--weight_decay 0.1 \
--learning_rate 0.002 \
--embed_lr 0.002 \
--scalar_lr 0.002 \
--head_lr 0.002 \
--warmup_proportion 0.0 \
--z_loss_weight 0.000 \
--local_batch_size 1 \
--optimizer muon \
--experiment full_run \
--global_batch_size 256 \
--momentum 0.95 \
--hybrid_numerator 7 \
--hybrid_denominator 8 \
--wd_scales"

echo $CMD


srun --label --cpu-bind=mask_cpu:$MYMASKS \
  singularity exec $SIFPYTORCH \
    ./conda-python-distributed -u $CMD
