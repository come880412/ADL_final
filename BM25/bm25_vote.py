import numpy as np
import json
import pandas as pd
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from rank_bm25 import BM25Okapi
from tqdm.auto import tqdm
from collections import defaultdict

from average_precision import mapk

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=Path,
        help="Path to the datasets.",
        default="./data",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Path to the preprocessed datasets.",
        default="./cache_bm25",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Directory to save the results.",
        default="./output_bm25_vote",
    )
    parser.add_argument(
        "--k_mAP", 
        type=int, 
        default=50, 
        help="The constant for mean Average Precision (mAP)."
    )
    parser.add_argument(
        "--factor",
        type=float,
        default=2,
        help="The factor for the number of considered target (factor * k_MAP)."
    )
    parser.add_argument(
        "--n_users", 
        type=int, 
        default=100, 
        help="The number of relevant users. (-1 for all users)"
    )
    parser.add_argument(
        "--evaluation_sets", 
        type=str, 
        default="val_seen val_unseen", 
        help="The datasets for evaluation."
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    for tag in ["", "_group"]:
        target = "subgroup" if tag else "course_id"
        
        data_files = {}
        data_files["train"] = args.input_dir / f"train{tag}.csv"
        data_files["val_seen"] = args.input_dir / f"val_seen{tag}.csv"
        data_files["val_unseen"] = args.input_dir / f"val_unseen{tag}.csv"
        data_files["test_seen"] = args.input_dir / f"test_seen{tag}.csv"
        data_files["test_unseen"] = args.input_dir / f"test_unseen{tag}.csv"
        datasets = {split: pd.read_csv(file, index_col=0) for split, file in data_files.items()}
        results = {split: -1 for split, file in data_files.items()}
        
        # Train
        with open(args.cache_dir / f"tokenized_train{tag}.pkl", "rb") as f:
            tokenized_corpus = pickle.load(f)
        n = len(tokenized_corpus)
        if args.n_users == -1:
            bm25 = BM25Okapi(tokenized_corpus)
        
        # Evaluation
        np.random.seed(1)
        eval_sets = args.evaluation_sets.split()
        for split in eval_sets:
            with open(args.cache_dir / f"tokenized_{split}{tag}.pkl", "rb") as f:
                tokenized_query = pickle.load(f)
            actual = []
            predicted = []
            for k, user in enumerate(tqdm(datasets[split].index, desc=f"{split}{tag}")):
                if args.n_users != -1:
                    subcorpus_idx = np.random.choice(n, args.n_users, replace=False)
                    tokenized_subcorpus = [tokenized_corpus[i] for i in subcorpus_idx]
                    bm25 = BM25Okapi(tokenized_subcorpus)
                doc_scores = bm25.get_scores(tokenized_query[k])
                subuser_rank = np.argsort(doc_scores)[::-1]
                user_rank = subuser_rank if args.n_users == -1 else [subcorpus_idx[i] for i in subuser_rank]
                history = []
                if not tag and not "unseen" in split:
                    history = str(datasets["train"].loc[user][target]).split()
                c = args.factor * args.k_mAP
                target_rank = defaultdict(lambda: c * args.n_users)
                relevant = []
                for i, r in enumerate(user_rank):
                    target_ids = str(datasets["train"].iloc[r][target]).split()
                    for j, target_id in enumerate(target_ids):
                        if j >= args.k_mAP:
                            break
                        if not tag and target_id in history:
                            continue
                        target_rank[target_id] = target_rank[target_id] - c + i
                        if target_id not in relevant:
                            relevant.append(target_id)
                    if len(relevant) > args.factor * args.k_mAP:
                        break
                sorted_target_rank = sorted(target_rank.items(), key=lambda x:x[1])
                # print(sorted_target_rank[:50])
                actual.append(str(datasets[split].iloc[k][target]).split())
                predicted.append([target_id for target_id, _ in sorted_target_rank][:50])
            df = pd.DataFrame({target: [" ".join(pred) for pred in predicted]},
                              index=datasets[split].index)
            df.to_csv(args.output_dir / f"submission_{split}{tag}.csv")
            if "val" in split:
                results[split] = mapk(actual, predicted, args.k_mAP)
                print(f"{split}{tag} result: {results[split]}")
        result_path = args.output_dir / f"results{tag}.json"
        result_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
