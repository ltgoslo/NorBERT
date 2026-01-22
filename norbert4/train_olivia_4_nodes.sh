#!/bin/bash -e
#SBATCH --job-name=norbert4hplt
#SBATCH --account=nn10029k
#SBATCH --nodes=4
#SBATCH --gpus-per-node=4
#SBATCH --tasks-per-node=4
#SBATCH --cpus-per-task=7
#SBATCH --partition=accel
#SBATCH --mem=0
#SBATCH --time=48:00:00

echo "SLURM_TASKS_PER_NODE: $SLURM_TASKS_PER_NODE"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_NODELIST: $SLURM_NODELIST"
echo "SLURM_NTASKS: $SLURM_NTASKS"

SIF="/cluster/projects/nn9851k/containers/pytorch2.7_cu2.9_py3.12_arm_nlpl.sif"

export WORLD_SIZE=$SLURM_NTASKS
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR
BATCH_SIZE=1
CMD="python3 train.py \
--train_path \
/cluster/work/projects/nn9851k/mariiaf/hplt/deu_Latn/tokenized_shards_16/train \
--tokenizer_path /cluster/work/projects/nn9851k/mariiaf/hplt/deu_Latn/tokenizer.json \
--output_dir /cluster/work/projects/nn9851k/mariiaf/hplt/deu_Latn/norbert_4_nodes \
--dataset_weights 1.0 \
--name NorBERT4_base_deu_Latn_4_nodes \
--max_steps 31250 \
--config_file configs/base.json \
--cooldown_proportion 0.2 \
--weight_decay 0.1 \
--learning_rate 0.002 \
--embed_lr 0.002 \
--scalar_lr 0.002 \
--head_lr 0.002 \
--warmup_proportion 0.0 \
--z_loss_weight 0.000 \
--local_batch_size $BATCH_SIZE \
--max_seq_length $((8192*2)) \
--optimizer muon \
--experiment NorBERT4_base_deu_Latn_4_nodes \
--global_batch_size 256 \
--momentum 0.95 \
--hybrid_numerator 7 \
--hybrid_denominator 8 \
--wd_scales"

echo $CMD

srun apptainer exec --nv -B /cluster/projects/:/cluster/projects/,/cluster/work/projects/:/cluster/work/projects/ $SIF $CMD