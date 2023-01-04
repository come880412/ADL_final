import pandas as pd
from argparse import ArgumentParser, Namespace
from collections import defaultdict

from average_precision import mapk

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--submission_input_files",
        type=str,
        help="Path to the input submission files to emsemble.",
        default="./submission/ensemble_unseen_group.csv \
            ./submission/BM25_ensemble_unseen_group.csv",
    )
    parser.add_argument(
        "--submission_output_file",
        type=str,
        help="Path to the output submission file.",
        default="./submission/ensemble_unseen_group_final.csv",
    )
    parser.add_argument(
        "--validation_file",
        type=str,
        help="Path to the validation file for mAP evaluation.",
        default=None,
    )
    parser.add_argument(
        "--target",
        type=str,
        help="Either course_id or subgroup.",
        default="subgroup",
    )
    parser.add_argument(
        "--postprocess_ids",
        type=str,
        help="The ids for postprocessing",
        # default="6156a77fdf426a0007cc5fe1",
        default="",
    )
    parser.add_argument(
        "--k_mAP", 
        type=int, 
        default=50, 
        help="The constant for mean Average Precision (mAP)."
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    
    result_files = args.submission_input_files.split()
    n = len(result_files)
    results = [pd.read_csv(file, index_col=0) for file in result_files]
    predicted = []
    for k, user in enumerate(results[0].index):
        rank = defaultdict(lambda: args.k_mAP * n)
        for i in range(n):
            target_ids = str(results[i].loc[user][args.target]).split()
            for j, target_id in enumerate(target_ids):
                if j >= args.k_mAP:
                    break
                rank[target_id] = rank[target_id] - args.k_mAP + j
        sorted_rank = sorted(rank.items(), key=lambda x:x[1])
        postprocess_ids = args.postprocess_ids.split()
        for target, _ in sorted_rank:
            if target not in postprocess_ids:
                postprocess_ids.append(target)
            if len(postprocess_ids) >= args.k_mAP:
                break
        predicted.append(" ".join(postprocess_ids))
    df = pd.DataFrame({args.target: predicted}, index = results[0].index)
    df.to_csv(args.submission_output_file)
    if args.validation_file:
        df = pd.read_csv(args.validation_file, index_col=0)
        actual = [str(target_ids).split() for target_ids in df[args.target]]
        predicted = [str(target_ids).split() for target_ids in predicted]
        print("map:", mapk(actual, predicted, args.k_mAP))