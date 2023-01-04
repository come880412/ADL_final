#!/bin/bash
python3 main.py --unbalanced_weight --data_dir $1 --dataset_type $3

python3 test.py --resume ./$3/model_last.pth --data_dir $1 --pred_dir $2 --dataset_type $3
              
