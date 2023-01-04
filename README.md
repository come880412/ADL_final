# ADL_final
2022 Fall ADL final

Teammates: \
黃繼綸 (r09942171)
黃佳文 (r11942157)
林彥伯 (d10943030)
林詩敏 (r11922058)

Seen course prediction:
Please refer to the `bert_embedding` folder

## BM25
### Environment
```
cd BM25
pip install -r requirements.txt
```

### Preprocessing
```
usage: preprocess.py [-h] [--input_dir INPUT_DIR] [--output_dir OUTPUT_DIR]

optional arguments:
  -h, --help            show this help message and exit
  --input_dir INPUT_DIR
                        Path to the datasets.
  --output_dir OUTPUT_DIR
                        Directory to save the results.
```


### Methods
#### Relevant courses/subgroups with weights
```
usage: bm25_target_weight.py [-h] [--input_dir INPUT_DIR] [--cache_dir CACHE_DIR] [--output_dir OUTPUT_DIR] [--k_price K_PRICE] [--k_frequency K_FREQUENCY] [--k_mAP K_MAP]
                             [--evaluation_sets EVALUATION_SETS]

optional arguments:
  -h, --help            show this help message and exit
  --input_dir INPUT_DIR
                        Path to the datasets.
  --cache_dir CACHE_DIR
                        Path to the preprocessed datasets.
  --output_dir OUTPUT_DIR
                        Directory to save the results.
  --k_price K_PRICE     The constant for price weight.
  --k_frequency K_FREQUENCY
                        The constant for frequency weight.
  --k_mAP K_MAP         The constant for mean Average Precision (mAP).
  --evaluation_sets EVALUATION_SETS
                        The datasets for evaluation.
```

#### Relevant courses/subgroups with filters
```
usage: bm25_target_filter.py [-h] [--input_dir INPUT_DIR] [--cache_dir CACHE_DIR] [--output_dir OUTPUT_DIR] [--k_price K_PRICE] [--k_frequency K_FREQUENCY] [--k_mAP K_MAP]
                             [--evaluation_sets EVALUATION_SETS]

optional arguments:
  -h, --help            show this help message and exit
  --input_dir INPUT_DIR
                        Path to the datasets.
  --cache_dir CACHE_DIR
                        Path to the preprocessed datasets.
  --output_dir OUTPUT_DIR
                        Directory to save the results.
  --k_price K_PRICE     The constant for price weight.
  --k_frequency K_FREQUENCY
                        The constant for frequency weight.
  --k_mAP K_MAP         The constant for mean Average Precision (mAP).
  --evaluation_sets EVALUATION_SETS
                        The datasets for evaluation.
```

#### Relevant users
```
usage: bm25_user.py [-h] [--input_dir INPUT_DIR] [--cache_dir CACHE_DIR] [--output_dir OUTPUT_DIR] [--k_mAP K_MAP] [--n_users N_USERS] [--evaluation_sets EVALUATION_SETS]

optional arguments:
  -h, --help            show this help message and exit
  --input_dir INPUT_DIR
                        Path to the datasets.
  --cache_dir CACHE_DIR
                        Path to the preprocessed datasets.
  --output_dir OUTPUT_DIR
                        Directory to save the results.
  --k_mAP K_MAP         The constant for mean Average Precision (mAP).
  --n_users N_USERS     The number of relevant users. (-1 for all users)
  --evaluation_sets EVALUATION_SETS
                        The datasets for evaluation.
```

#### Relevant users with votes
```
usage: bm25_vote.py [-h] [--input_dir INPUT_DIR] [--cache_dir CACHE_DIR] [--output_dir OUTPUT_DIR] [--k_mAP K_MAP] [--factor FACTOR] [--n_users N_USERS]
                    [--evaluation_sets EVALUATION_SETS]

optional arguments:
  -h, --help            show this help message and exit
  --input_dir INPUT_DIR
                        Path to the datasets.
  --cache_dir CACHE_DIR
                        Path to the preprocessed datasets.
  --output_dir OUTPUT_DIR
                        Directory to save the results.
  --k_mAP K_MAP         The constant for mean Average Precision (mAP).
  --factor FACTOR       The factor for the number of considered target (factor * k_MAP).
  --n_users N_USERS     The number of relevant users. (-1 for all users)
  --evaluation_sets EVALUATION_SETS
                        The datasets for evaluation.
```

## Ensemble
```
usage: ensemble.py [-h] [--submission_input_files SUBMISSION_INPUT_FILES] [--submission_output_file SUBMISSION_OUTPUT_FILE] [--validation_file VALIDATION_FILE]
                   [--target TARGET] [--postprocess_ids POSTPROCESS_IDS] [--k_mAP K_MAP]

optional arguments:
  -h, --help            show this help message and exit
  --submission_input_files SUBMISSION_INPUT_FILES
                        Path to the input submission files to emsemble.
  --submission_output_file SUBMISSION_OUTPUT_FILE
                        Path to the output submission file.
  --validation_file VALIDATION_FILE
                        Path to the validation file for mAP evaluation.
  --target TARGET       Either course_id or subgroup.
  --postprocess_ids POSTPROCESS_IDS
                        The ids for postprocessing
  --k_mAP K_MAP         The constant for mean Average Precision (mAP).
```
