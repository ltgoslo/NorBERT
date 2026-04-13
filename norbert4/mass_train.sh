for lang in fra_Latn swe_Latn
    do
        echo $lang
        sbatch train_olivia_2_nodes.sh $lang 
    done
