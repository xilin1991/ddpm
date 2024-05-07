#!/bin/bash

hostname=$(hostname)
if [ "$hostname" = "E701-amax" ]; then
    source ~/dev/anaconda3/bin/activate
elif [ "$hostname" = "amax" ]; then
    source ~/ProgramData/anaconda3/bin/activate
else
    echo 'error'
    exit 0
fi

conda activate torch_lts
export CUDA_VISIBLE_DEVICES=0
# export OMP_NUM_THREADS=4

python main.py --data_type 's'
python main.py --data_type 'o'
python main.py --data_type 'b'
python main.py --data_type 'm'
python main.py --data_type 'r'
