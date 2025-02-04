#!/usr/bin/env bash

datasets=("ppi" "flickr" "reddit")
configs=("rw" "node")

for dataset in "${datasets[@]}"; do
    for config in "${configs[@]}"; do
        for (( i=0; i<3; i++ )); do
            python main.py -d "data/${dataset}" -c "conf/${dataset}_${config}.yaml"
        done
    done
done

datasets=("ppi")
configs=("rnd")

for dataset in "${datasets[@]}"; do
    for config in "${configs[@]}"; do
        for (( i=0; i<3; i++ )); do
            python main.py -d "data/${dataset}" -c "conf/${dataset}_${config}.yaml"
        done
    done
done

