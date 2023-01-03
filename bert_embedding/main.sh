python preprocess.py --data_dir "${1}"

# python main.py --unbalanced_weight \
#                --feature_name gender occupation_titles interests recreation_names \
#                --fix_encoder \
#                --dataset_type courses \
#                --thres 0.35

CUDA_VISIBLE_DEVICES=1 python test.py --data_dir "${1}" \
               --feature_name gender occupation_titles interests recreation_names \
               --resume ./model.pth \
               --dataset_type courses \
               --thres 0.35