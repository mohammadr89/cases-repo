#!/bin/bash
base_dir="."  # This ensures directories are created inside `cases/`
scenarios=("grass" "forest" "hetero")  # Land-use types
wind=("3u" "10u")  
grids=("8grid_gz" "10grid" "20grid")
compensations=("bi_dir" "one_dir" "no_ps_bidir" "no_ps_onedir")  

for scenario in "${scenarios[@]}"; do
    for w in "${wind[@]}"; do
        for grid in "${grids[@]}"; do
            for comp in "${compensations[@]}"; do
                mkdir -p "${base_dir}/${scenario}/${w}/${grid}/${comp}"
            done
        done
    done
done
