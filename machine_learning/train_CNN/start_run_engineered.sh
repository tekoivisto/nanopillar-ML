#!/bin/bash

gt_files=("moduluses_0.0015.npy" "yield_stress_0.005.npy")
#gt_files=("moduluses_0.0015.npy")
descriptors=("grain_boundary" "quaternion" "combined")
resolutions=(16 32)
#learning_rates=(0.05 0.015 0.005 0.0015 0.0005)
learning_rates=(0.0005)

for f in "${gt_files[@]}"
do
    echo $f
    mkdir -p "$f"
    for d in "${descriptors[@]}"
    do
        echo $d
        mkdir -p "$f/$d"
        for r in "${resolutions[@]}"
        do
	    
            echo $r
            mkdir -p "$f/$d/$r"
            for l in "${learning_rates[@]}"
            do
                echo $l
                mkdir -p "$f/$d/$r/$l"

                cd "$f/$d/$r/$l"

                sbatch ../../../../job_engineered.sh $f $r $d $l
                cd ../../../../
            done
        done
    done
done

