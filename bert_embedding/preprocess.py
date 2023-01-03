import pandas as pd
import numpy as np
import os
import json
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Path
    parser.add_argument('--data_dir', default='../hahow/data', help='path to dataset')
    args = parser.parse_args()
    data_path = args.data_dir
    subgroups = np.array(pd.read_csv(os.path.join(data_path, "subgroups.csv")))
    courses = np.array(pd.read_csv(os.path.join(data_path, "courses.csv")))
    
    label_to_name = {"subgroups":{}, "courses":{}}
    
    for subgroup in subgroups:
        id, label_name = subgroup
        label_to_name["subgroups"][id] = str(label_name)
    
    for id, course in enumerate(courses):
        course_id = course[0]
        label_to_name["courses"][course_id] = id
    
    with open('label_mapping.json', 'w', encoding='utf-8') as f:
        json.dump(label_to_name, f, ensure_ascii=False)
