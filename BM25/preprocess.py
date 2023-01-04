import numpy as np
import pandas as pd
import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from ckip_transformers.nlp import CkipWordSegmenter

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=Path,
        help="Path to the datasets.",
        default="./data",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Directory to save the results.",
        default="./cache_bm25",
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    users = pd.read_csv(args.input_dir / "users.csv", index_col=0)
    courses = pd.read_csv(args.input_dir / "courses.csv", index_col=0)
    subgroups = pd.read_csv(args.input_dir / "subgroups.csv", index_col=0)
    field2tag = {"course_name": "課程名稱", "groups": "類別", "sub_groups": "子類別", 
                 "topics": "主題", "course_published_at_local": "課程發佈時間",
                 "gender": "性別", "female": "女性", "male": "男性", "other": "其他",
                 "occupation_titles": "職業", "interests": "興趣", "recreation_names": "休閒"}
    
    course_description = []
    target_fields = {'course_name', 'groups', 'sub_groups', 'topics', 'course_published_at_local'}
    for idx, _ in enumerate(courses.index):
        course = courses.iloc[idx]
        course_description.append("")
        fields = course.dropna().index
        for field in fields:
            if not field in target_fields:
                continue
            course_description[idx] += f"{field2tag[field]}：{course[field]}。"
    course_description_df = pd.DataFrame(
        course_description, columns=["course_description"], index=courses.index
    )
    course_description_path = args.output_dir / "course_description.csv"
    course_description_df.to_csv(course_description_path)
    
    ws_driver = CkipWordSegmenter(model="bert-base", device=0)
    tokenized_course = ws_driver(course_description)
    with open(args.output_dir / "tokenized_course_description.pkl", "wb") as f:
        pickle.dump(tokenized_course, f)
    tokenized_subgroup = ws_driver(subgroups['subgroup_name'].tolist())
    with open(args.output_dir / "tokenized_subgroup_description.pkl", "wb") as f:
        pickle.dump(tokenized_subgroup, f)
    
    course2idx = {course: idx for idx, course in enumerate(courses.index)}
    idx2course = {idx: course for idx, course in enumerate(courses.index)}
    idx2course_path = args.output_dir / "idx2course.json"
    idx2course_path.write_text(json.dumps(idx2course, indent=2))
    subgroup2idx = {str(subgroup): idx for idx, subgroup in enumerate(subgroups.index)}
    idx2subgroup = {idx: subgroup for idx, subgroup in enumerate(subgroups.index)}
    idx2subgroup_path = args.output_dir / "idx2subgroup.json"
    idx2subgroup_path.write_text(json.dumps(idx2subgroup, indent=2))
    
    courses['weight'] = 1.0 - courses['course_price']/courses['course_price'].max()
    with open(args.output_dir / "course_price_weight.pkl", "wb") as f:
        pickle.dump(courses['weight'].tolist(), f)
    sub_group2idx = {subgroups.iloc[idx]['subgroup_name']: idx 
                    for idx, subgroup_id in enumerate(subgroups.index)}
    subgroups_weight = np.zeros(len(subgroups.index))
    subgroups_cnt = np.zeros(len(subgroups.index))
    courses_dropna = courses.dropna()
    for idx, course in enumerate(courses_dropna.index):
        course_data = courses_dropna.iloc[idx]
        course_subgroups = course_data['sub_groups'].split(',')
        for subgroup in course_subgroups:
            subgroup_idx = sub_group2idx[subgroup]
            subgroups_weight[subgroup_idx] += course_data['weight']
            subgroups_cnt[subgroup_idx] += 1
    subgroups['weight'] = subgroups_weight / subgroups_cnt
    with open(args.output_dir / "subgroup_price_weight.pkl", "wb") as f:
        pickle.dump(subgroups['weight'].tolist(), f)
    
    for tag in ["", "_group"]:
        target = "subgroup" if tag else "course_id"
        
        data_files = {}
        data_files["train"] = args.input_dir / f"train{tag}.csv"
        data_files["val_seen"] = args.input_dir / f"val_seen{tag}.csv"
        data_files["val_unseen"] = args.input_dir / f"val_unseen{tag}.csv"
        data_files["test_seen"] = args.input_dir / f"test_seen{tag}.csv"
        data_files["test_unseen"] = args.input_dir / f"test_unseen{tag}.csv"
        datasets = {split: pd.read_csv(file, index_col=0) for split, file in data_files.items()}
        preprocessed_datasets = {split:{'user_id':[], 'user_description': [], target: []} 
                            for split in datasets.keys()}
        
        history_index = datasets["train"].dropna().index
        target_cnt = np.zeros(len(subgroups.index) if tag else len(courses.index))
        for user in history_index:
            history = str(datasets["train"].loc[user][target]).split()
            for target_id in history:
                target_idx = subgroup2idx[target_id] if tag else course2idx[target_id]
                target_cnt[target_idx] += 1
        output_tag = "subgroup" if tag else "course"
        with open(args.output_dir / f"{output_tag}_frequency_weight.pkl", "wb") as f:
            pickle.dump(target_cnt/sum(target_cnt), f)
        
        for split in datasets.keys():
            description = preprocessed_datasets[split]['user_description']
            for k, user in enumerate(datasets[split].index):
                data = datasets[split].iloc[k]
                preprocessed_datasets[split]['user_id'].append(user)
                preprocessed_datasets[split][target].append(data[target])
                
                user_data = users.loc[user]
                fields = user_data.dropna().index
                description.append("")
                for field in fields:
                    if field == "gender":
                        description[k] += f"{field2tag[field]}：{field2tag[user_data[field]]}。"
                    else:
                        description[k] += f"{field2tag[field]}：{user_data[field]}。"
                if user in history_index:
                    description[k] += "記錄："
                    history = str(datasets["train"].loc[user][target]).split()
                    if tag:
                        description[k] += "，".join([subgroups.loc[int(subgroup_id)]["subgroup_name"] for subgroup_id in history])
                    else:
                        description[k] += "".join([course_description[course2idx[course_id]] for course_id in history])
            df = pd.DataFrame(preprocessed_datasets[split])
            df.to_csv(args.output_dir / f"preprocessed_{split}{tag}.csv", index=False)
            tokenized_description = ws_driver(description)
            with open(args.output_dir / f"tokenized_{split}{tag}.pkl", "wb") as f:
                pickle.dump(tokenized_description, f)
            
    