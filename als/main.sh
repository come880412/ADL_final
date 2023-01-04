# Preprocess
python preprocess.py --data_dir "${1}" --train_file_course_out './cache/train.json'

# Seen Topic (Pubic Test MAP@50: 0.29579)
# kaggle上最好的結果有用到此份result做ensemble
python3 run_seen_group_post_process.py --train_file './cache/train.json' --data_dir "${1}" --output_file "${2}"