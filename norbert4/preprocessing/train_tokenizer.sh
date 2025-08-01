#!/bin/bash
#SBATCH --account=project_465001384
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=48:00:00

set -o errexit  # Exit the script on any error
set -o nounset  # Treat any unset variables as an error

mkdir -p workdir
wd=$(realpath workdir)


# if run without sbatch, invoke here
if [ -z $SLURM_JOB_ID ]; then
    mkdir -p logs
    sbatch "$0"
    exit
fi

# singularity setup
CONTAINER="/scratch/project_465001384/pytorch-lumi_sles-rocm-5.5.1-python-3.10-pytorch-v2.0.1-apex-torchvision-torchdata-torchtext-torchaudio.sif"
SING_BIND="/scratch/project_465001384"

# hold separate logs for easier debugging
rm -rf separate-logs
mkdir -p separate-logs
set -euo pipefail

# symlink logs/latest.out and logs/latest.err
ln -f -s $SLURM_JOB_ID.out logs/latest.out
ln -f -s $SLURM_JOB_ID.err logs/latest.err

CMD="train_tokenizer.py --num_shards 128"

echo $CMD
echo "START $SLURM_JOBID: $(date)"

if [ ! -d $wd/cray-deps ] ; then
  rm -rf $wd/cray-deps
  mkdir $wd/cray-deps
  cp /usr/lib64/libcxi* $wd/cray-deps
fi

srun \
    --label \
    singularity exec \
    -B /opt/cray:/opt/cray \
    -B $wd/cray-deps:/opt/cray-deps \
    -B $wd:/workdir \
    -B "$SING_BIND" \
    "$CONTAINER" \
    ./launch.sh \
    $CMD
