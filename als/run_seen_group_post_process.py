import os
import math
import implicit
import argparse
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

def cf_group(output_file, train_df, valid_df, courseid2groupid, method, factors=1, regularization=0.01, iterations=15):
    train_users = train_df["user_id"].index.tolist()
    train_x = train_df.drop(['user_id'], axis=1).to_numpy()

    labels = list(train_df.columns.values[1:])
    idx2courseid = {idx:label for idx, label in enumerate(labels)}
    courseid2idx = {label:idx for idx, label in enumerate(labels)}

    userid2idx = dict(zip(list(train_df.user_id), list(train_df.index)))
    idx2userid = dict(zip(list(train_df.index), list(train_df.user_id)))

    user_items = csr_matrix(train_x, dtype=np.float64)

    # initialize a model
    model = implicit.cpu.als.AlternatingLeastSquares(factors=factors, regularization=regularization, iterations=iterations)

    # train the model on a sparse matrix of user/item/confidence weights
    model.fit(user_items)

    # recommend items for a user
    valid_users = list(valid_df.user_id)

    with open(output_file, 'w') as f:
        f.write('user_id,subgroup\n')
        for valid_user in valid_users:
            valid_user_idx = userid2idx[valid_user]
            course_idxs, scores = model.recommend(valid_user_idx, user_items[valid_user_idx], N=50)
            f.write(idx2userid[valid_user_idx] + ',')
            group_ids = []
            for i in ['6156a77fdf426a0007cc5fe1', '6155cda6d425f500065f5c96', '5f7c210b1de7982fb413a3e9', '5f7c209762ad22756c7a1c74', '5f7c212262ad2203e77a1cc9', '60cb0a440dabda80019d5f7c', '6059aee039f2512548c187c6']:
                group_ids.extend(courseid2groupid[i])
            for course_idx in course_idxs:
                course_ids = idx2courseid[course_idx]
                group_ids.extend(courseid2groupid[course_ids])
            group_ids_deduplicate = list(set(group_ids))
            group_ids_deduplicate.sort(key=group_ids.index)
            group_ids_str = ''
            for i in group_ids_deduplicate:
                group_ids_str += str(i) + ' '
            group_ids_str = group_ids_str[:-1] + '\n'
            f.write(group_ids_str)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=str, default='./data', help="courses file path"
    )
    parser.add_argument(
        "--train_file", type=str, default='./cache/course/train.json', help="ground truth file path"
    )
    parser.add_argument(
        "--output_file", type=str, default='./output/test_seen_group.csv', help="output file path"
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    test_file = os.path.join(args.data_dir,'test_seen_group.csv')
    subgroup_file = os.path.join(args.data_dir,'subgroups.csv')
    course_file = os.path.join(args.data_dir,'courses.csv')

    train_df = pd.read_json(args.train_file)

    subgroups_df = pd.read_csv(subgroup_file)
    subgroups_name = list(subgroups_df['subgroup_name'])
    subgroups_df = pd.DataFrame.from_dict(dict(zip(list(subgroups_df.subgroup_id), list(subgroups_df.subgroup_name))), orient='index')

    course_df = pd.read_csv(course_file).drop(['course_name', 'course_price', 'teacher_id', 'teacher_intro', 'groups', 'topics', 'course_published_at_local', 'description', 'will_learn', 'required_tools', 'recommended_background', 'target_group'], axis=1)
    course_df = course_df.fillna('')

    def label_to_code(label):
        code_list = []
        label_list = label.split(",")
        if label_list==['']:
            return None
        for item in label_list:
            code_list.append(subgroups_df[subgroups_df[0] == item].index[0])
        return code_list

    course_df['label'] = course_df['sub_groups'].apply(label_to_code)
    course_df = course_df.fillna('')
    courseid2groupid = dict(zip(list(course_df.course_id), list(course_df.label)))
    
    method = 'als'
    factors = 100
    regularization = 0.03
    iterations = 30

    test_df = pd.read_csv(test_file)
    cf_group(args.output_file, train_df, test_df, courseid2groupid, method, factors, regularization, iterations)


if __name__ == "__main__":
    main()