#!/bin/bash

if [ "$#" -ne 1 ]; then
        echo "run_ffnn.sh <exp-dir>"
fi

exp_dir=$1
out_dir=$exp_dir/model
testset_dir=$exp_dir

which python3.9

python3.9 train_ffnn.py $exp_dir $out_dir \
        --lr 0.01 --epochs 1

python3.9 plot_ffnn.py $out_dir
