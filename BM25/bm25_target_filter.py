import numpy as np
import json
import pandas as pd
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from rank_bm25 import BM25Okapi
from tqdm.auto import tqdm

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
        default="./output_bm25_target_filter",
    )
    parser.add_argument(
        "--k_price",
        type=float,
        help="The constant for price weight.",
        default=1.0,
    )
    parser.add_argument(
        "--k_frequency",
        type=float,
        help="The constant for frequency weight.",
        default=0.4,
    )
    parser.add_argument(
        "--k_mAP", 
        type=int, 
        default=50, 
        help="The constant for mean Average Precision (mAP)."
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
        corpus = "subgroup" if tag else "course"
        idx2target_path = args.cache_dir / f"idx2{corpus}.json"
        idx2target = json.loads(idx2target_path.read_text())
        with open(args.cache_dir / f"tokenized_{corpus}_description.pkl", "rb") as f:
            tokenized_corpus = pickle.load(f)
        bm25 = BM25Okapi(tokenized_corpus)
        with open(args.cache_dir / f"{corpus}_price_weight.pkl", "rb") as f:
            price_weight = np.array(pickle.load(f))
        with open(args.cache_dir / f"{corpus}_frequency_weight.pkl", "rb") as f:
            frequency_weight = np.array(pickle.load(f))
        frequency_filter = np.argsort(frequency_weight)[::-1][:int(args.k_frequency * len(tokenized_corpus))]
        filters = []
        for i in frequency_filter:
            if tag or price_weight[i] >= args.k_price:
                filters.append(i)
        
        # Evaluation
        eval_sets = args.evaluation_sets.split()
        for split in eval_sets:
            with open(args.cache_dir / f"tokenized_{split}{tag}.pkl", "rb") as f:
                tokenized_query = pickle.load(f)
            actual = []
            predicted = []
            for k, user in enumerate(tqdm(datasets[split].index, desc=f"{split}{tag}")):
                doc_scores = bm25.get_scores(tokenized_query[k])
                rank = np.argsort(doc_scores)[::-1]
                actual.append(str(datasets[split].iloc[k][target]).split())
                predicted.append([])
                history = []
                if not tag and not "unseen" in split:
                    history = str(datasets["train"].loc[user][target]).split()
                for r in rank:
                    if r not in filters:
                        continue
                    target_id = idx2target[str(r)]
                    if not tag and target_id in history:
                        continue
                    if target_id not in predicted[k]:
                        predicted[k].append(str(target_id))
                    if len(predicted[k]) >= args.k_mAP:
                        break
                for r in rank:
                    target_id = idx2target[str(r)]
                    if not tag and target_id in history:
                        continue
                    if target_id not in predicted[k]:
                        predicted[k].append(str(target_id))
                    if len(predicted[k]) >= args.k_mAP:
                        break
            df = pd.DataFrame({target: [" ".join(pred) for pred in predicted]},
                              index=datasets[split].index)
            df.to_csv(args.output_dir / f"submission_{split}{tag}.csv")
            if "val" in split:
                results[split] = mapk(actual, predicted, args.k_mAP)
                print(f"{split}{tag} result: {results[split]}")
        result_path = args.output_dir / f"results{tag}.json"
        result_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
