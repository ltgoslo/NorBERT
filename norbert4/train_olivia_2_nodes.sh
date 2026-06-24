#!/bin/bash -e
#SBATCH --job-name=norbert4hplt
#SBATCH --account=nn10029k
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --tasks-per-node=4
#SBATCH --cpus-per-task=7
#SBATCH --partition=accel
#SBATCH --mem=0
#SBATCH --time=72:00:00
#SBATCH --output=/cluster/work/projects/nn9851k/mariiaf/hplt/logs/train-%j.out

echo "SLURM_TASKS_PER_NODE: $SLURM_TASKS_PER_NODE"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_NODELIST: $SLURM_NODELIST"
echo "SLURM_NTASKS: $SLURM_NTASKS"

ml reset
ml load NRIS/GPU
ml load NCCL/2.26.6-GCCcore-14.2.0-CUDA-12.8.0

LIBFABRIC_LIB_PATH="${EBROOTLIBFABRIC}/lib"
LIBFABRIC_INCLUDE_PATH="${EBROOTLIBFABRIC}/include"
NCCL_ROOT_PATH="${EBROOTNCCL}"
AWS_OFI_NCCL_LIB_PATH="${EBROOTAWSMINOFIMINNCCL}/lib"
CXI_LIB_PATH="/usr/lib64"

export SINGULARITYENV_TRITON_LIBCUDA_PATH="/usr/local/cuda/compat/lib.real"
export APPTAINERENV_TRITON_LIBCUDA_PATH="/usr/local/cuda/compat/lib.real"

SIF="/cluster/projects/nn9851k/containers/pytorch2.7_cu2.9_py3.12_arm_nlpl.sif"

LANGUAGE=${1}
echo $LANGUAGE
export WORLD_SIZE=$SLURM_NTASKS
head_node=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# same as in https://documentation.sigma2.no/code_development/guides/pytorch_olivia/PyTorchMultiNode.html#job-script-for-multi-node-training
export MASTER_ADDR=$head_node
echo "MASTER_ADDR=$MASTER_ADDR"
export MASTER_PORT=29500
echo "MASTER_PORT=$MASTER_PORT"
BATCH_SIZE=1

# change max_steps to 15625 for small
CMD="python3 train.py \
--train_path /cluster/work/projects/nn9851k/mariiaf/hplt/$LANGUAGE/tokenized_shards/train \
--tokenizer_path /cluster/work/projects/nn9851k/mariiaf/hplt/$LANGUAGE/tokenizer.json \
--output_dir /cluster/work/projects/nn9851k/mariiaf/hplt/$LANGUAGE/norbert_2_nodes \
--dataset_weights 1.0 \
--name NorBERT4_base_$LANGUAGE \
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
--experiment NorBERT4_base_$LANGUAGE \
--global_batch_size 256 \
--momentum 0.95 \
--hybrid_numerator 7 \
--hybrid_denominator 8 \
--wd_scales \
--checkpoint_every 3125 \
--save_every 0 \
--validation_steps 10 \
--validate_every 3125 \
--window_update 4,4,8,16,16 ${@:2}"

echo $CMD

srun apptainer exec --nv \
  -B "${LIBFABRIC_LIB_PATH}:/opt/libfabric/lib" \
  -B "${LIBFABRIC_INCLUDE_PATH}:/opt/libfabric/include" \
  -B "${NCCL_ROOT_PATH}:/opt/nccl" \
  -B "${AWS_OFI_NCCL_LIB_PATH}:/opt/aws-ofi-nccl/lib" \
  -B "${CXI_LIB_PATH}:${CXI_LIB_PATH}" \
  --env FI_PROVIDER="${FI_PROVIDER:-cxi}" \
  --env FI_CXI_RX_MATCH_MODE="${FI_CXI_RX_MATCH_MODE:-hybrid}" \
  --env NCCL_PROTO="${NCCL_PROTO:-^LL128}" \
  --env LIBFABRIC_HOME="/opt/libfabric" \
  --env NCCL_HOME="/opt/nccl" \
  --env AWS_OFI_NCCL_HOME="/opt/aws-ofi-nccl" \
  --env LD_LIBRARY_PATH="${LIBFABRIC_HOME}/lib:${NCCL_HOME}/lib:${AWS_OFI_NCCL_HOME}/lib:/usr/lib64:${LD_LIBRARY_PATH}" \
  --env CPATH="${LIBFABRIC_HOME}/include:${CPATH:-}" \
  -B /cluster/projects/:/cluster/projects/,/cluster/work/projects/:/cluster/work/projects/ $SIF \
  $CMD
