#! /bin/bash

for i in {0..46}
do
echo ${i}
sbatch create_tfrec_phase1.slurm ready/norw_${i}.txt.gz vocabulary/norwegian_wordpiece_vocab_30k.txt ${i}.tfr
done




