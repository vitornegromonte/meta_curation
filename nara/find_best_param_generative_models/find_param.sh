#!/bin/bash

model_names=('ctgan' 'tvae' 'ddpm' 'nflow' 'bayesian_network' 'tabformer' 'great')
path2save='/home/vitor/Development/meta_curation/nara/models'

for model in "${model_names[@]}"; do
    python get_parameters.py --path2save "$path2save" --model_name "$model"
done

