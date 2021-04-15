#!/bin/bash
#SBATCH --job-name=BERT_TFR
#SBATCH --mail-type=FAIL
#SBATCH --account=nn9851k
#SBATCH --time=15:00:00      # Max walltime is 14 days.
#SBATCH --mem-per-cpu=8G

# Definining resource we want to allocate.
#SBATCH --nodes=1
#SBATCH --ntasks=8

# This is used to make checkpoints and logs to readable and writable by other members in the project.
umask 0007

module use -a /cluster/projects/nn9851k/software/easybuild/install/modules/all/
module purge   # Recommended for reproducibility
module load NLPL-nvidia_BERT/20.06.8-gomkl-2019b-TensorFlow-1.15.2-Python-3.7.4

export MAX_PR=20 # max predictions per sequence
export MAX_SEQ_LEN=128 # max sequence length (128 for the 1st phase, 512 for the 2nd phase)

# Some handy variables, you'll need to change these.
export BERT_ROOT=$EBROOTNLPLMINNVIDIA_BERT
export LOCAL_ROOT=`pwd`
export OUTPUT_DIR=$LOCAL_ROOT/data/norbert${MAX_SEQ_LEN}/

mkdir -p $OUTPUT_DIR

echo ${1}  # input corpus
echo ${2}  # wordpiece vocabulary file
echo ${3}  # name(s) of the output TFR file(s), for example, "norbert.tfr"

# TODO: implement creating a list of TFR file names from the list of input file names?

python3 ${BERT_ROOT}/utils/create_pretraining_data.py --input_file=${1} --vocab_file=${2} --dupe_factor=10 --max_seq_length=${MAX_SEQ_LEN} --max_predictions_per_seq=${MAX_PR} --output_file=${OUTPUT_DIR}${3}

# This is for the Uncased variant:
# python3 create_pretraining_data.py --input_file=${1}  --vocab_file=${2} --dupe_factor=10 --max_seq_length=128 --max_predictions_per_seq=20 --do_lower_case --output_file=${OUTPUT_DIR}${3}
