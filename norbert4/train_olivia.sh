#!/bin/bash -e
#SBATCH --job-name=norbert4hplt
#SBATCH --account=nn10029k
#SBATCH --nodes=2
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

CMD="python3 train.py \
--train_path \
/cluster/work/projects/nn9851k/mariiaf/hplt/deu_Latn/tokenized_shards/ \
--tokenizer_path /cluster/work/projects/nn9851k/mariiaf/hplt/deu_Latn/tokenizer.json \
--output_dir /cluster/work/projects/nn9851k/mariiaf/hplt/deu_Latn/norbert \
--dataset_weights 1.0 \
--name NorBERT4_base_deu_Latn_2_nodes \
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
--local_batch_size 1 \
--optimizer muon \
--experiment NorBERT4_base_deu_Latn \
--global_batch_size 32 \
--momentum 0.95 \
--hybrid_numerator 7 \
--hybrid_denominator 8 \
--wd_scales"

echo $CMD

srun apptainer exec -B /cluster/projects/:/cluster/projects/,/cluster/work/projects/:/cluster/work/projects/ $SIF $CMD