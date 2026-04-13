for lang in fra_Latn nob_Latn swe_Latn
    do
        echo $lang
        sbatch convert_to_hf.sh /cluster/work/projects/nn9851k/mariiaf/hplt/ /cluster/work/projects/nn9851k/mariiaf/hplt/hplt_hf_models 1 $lang
    done
