
import os 
import json
import argparse
import numpy as np
import pandas as pd

def preprocess(input_file, output_file, course_file):

    def course_id_to_label(course_id):
        label_list = [0]*728
        course_id_list = course_id.split(" ")
        for course_id in course_id_list:
            label_list[course_dict[course_id]] = 1
        return label_list
    
    train_df = pd.read_csv(input_file)
    train_df = train_df.fillna('')

    course_df = pd.read_csv(course_file).drop(['course_price', 'teacher_id', 'teacher_intro', 'groups', 'sub_groups', 'topics', 'course_published_at_local', 'description', 'will_learn', 'required_tools', 'recommended_background', 'target_group'], axis=1)
    course_dict = dict(zip(list(course_df.course_id), list(course_df.index)))
    course_id = list(course_df['course_id'])

    preprocessed_df = train_df
    preprocessed_df['label'] = preprocessed_df['course_id'].apply(course_id_to_label)
    preprocessed_df[course_id] = pd.DataFrame(preprocessed_df.label.tolist(), index= preprocessed_df.index)
    preprocessed_df = preprocessed_df.drop(['label', 'course_id'], axis=1)

    preprocessed_df.to_json(output_file, orient="records")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=str, default='./data', help="courses file path"
    )
    parser.add_argument(
        "--train_file_course_out", type=str, default='./cache/train.json', help="output test file path"
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    course_file = os.path.join(args.data_dir,'courses.csv')
    train_file_in = os.path.join(args.data_dir,'train.csv')

    os.makedirs(os.path.dirname(args.train_file_course_out), exist_ok=True)
    preprocess(train_file_in, args.train_file_course_out, course_file)

if __name__ == '__main__':
    main()