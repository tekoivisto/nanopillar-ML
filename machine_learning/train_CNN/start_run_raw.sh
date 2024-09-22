#!/bin/bash

gt_files=("moduluses_0.0015.npy" "yield_stress_0.005.npy")
descriptors=("raw")
resolutions=(64 128)

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
            cd "$f/$d/$r"

            sbatch ../../../job_raw.sh $f $r $d
            cd ../../../
        done
    done
done

    
