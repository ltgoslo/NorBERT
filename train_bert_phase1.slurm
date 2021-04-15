#!/bin/bash
#SBATCH --job-name=BERT
#SBATCH --mail-type=FAIL
#SBATCH --account=nn9447k
#SBATCH --partition=accel    # To use the accelerator nodes
#SBATCH --gres=gpu:4         # To specify how many GPUs to use (on one node)
#SBATCH --time=50:00:00      # Max walltime is 14 days.
#SBATCH --mem-per-cpu=8G

# Definining resource we want to allocate. We set 16 tasks over 4 nodes as we have 4 GPUs per node.
#SBATCH --nodes=4
#SBATCH --ntasks=16

# 6 CPU cores per task to keep the parallel data feeding going. A little overkill, but CPU time is very cheap compared to GPU time.
#SBATCH --cpus-per-task=6

# This is used to make checkpoints and logs to readable and writable by other members in the project.
umask 0007

module use -a /cluster/projects/nn9851k/software/easybuild/install/modules/all/
module purge   # Recommended for reproducibility
module load NLPL-nvidia_BERT/20.06.8-gomkl-2019b-TensorFlow-1.15.2-Python-3.7.4


# export BERT_ROOT=$EBROOTNLPLMINNVIDIAMINBERT  # change to this for recent releases of NLPL Laboratory
export BERT_ROOT=$EBROOTNLPLMINNVIDIA_BERT

export LOCAL_ROOT=`pwd`


export CORPUS=${1}  # path to the input TFR
export MODEL_DIR=${2}  # path to the trained model directory
export CONFIG=${3}  # path to the BERT config file (JSON)

export N_GPU=16  # number of GPUs to use
export N_BATCH=64  # train batch size

export MAX_PR=20 # max predictions per sequence (20 for the 1st phase, 77 for the 2nd phase)
export MAX_SEQ_LEN=128 # max sequence length (128 for the 1st phase, 512 for the 2nd phase)

echo "Training TFR: ${CORPUS}"
echo "BERT configuration file: ${CONFIG}"

mkdir -p $MODEL_DIR

echo "Directory for the trained model: ${MODEL_DIR}"

export NCCL_DEBUG=INFO
export TF_XLA_FLAGS="--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"

echo "Training BERT on the ${CORPUS}..."

# The actual command we want to run.
# Batch size is the max amount we can fit into VRAM, `--max_seq_length` is 128 for the first part of the training.
# `--max_predictions_per_seq` is the default and must be the same as set in the tfrecord generation.
# `--horovod` enables Horovod support, `--use_xla` enables TF's XLA JIT.
# `--manual_fp16` and `--noamp` enable using mixed-precision training on Tesla P100 GPUs
# For more modern GPUs (compute capability 7.0 or higher), these flags are not required, Automatic Mixed Precision (AMP) will be used by default.

# sentence_number = num_train_steps * global_batch_size
# global_batch_size = train_batch_size * GPUs number
# num_train_steps for 1 epoch = sentence_number / global_batch_size

mpiexec --bind-to socket -np ${N_GPU} python3 ${BERT_ROOT}/run_pretraining.py --input_files_dir=${CORPUS} --output_dir=${MODEL_DIR} --do_train=True --do_eval=False --bert_config_file=${CONFIG} --train_batch_size=${N_BATCH} --max_seq_length=${MAX_SEQ_LEN} --max_predictions_per_seq=${MAX_PR} --num_train_steps=50000 --num_warmup_steps=100 --learning_rate=1e-4 --horovod --noamp --manual_fp16 --dllog_path=${LOCAL_ROOT}/bert_phase1_log.json

echo "Phase 1 of training BERT finished."
