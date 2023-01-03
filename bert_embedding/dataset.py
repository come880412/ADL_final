from torch.utils.data import Dataset, DataLoader
import torch

import pandas as pd
import os
import pdb
import numpy as np
import tqdm
import json

from Feature_extractor import Feature_extract, Feature_all_extract

class hahow_dataset(Dataset):

    def __init__(self, args, tokenizer, emb_model, device, is_test=False, label_file="train"):
        # Initialization
        self.data_dir = args.data_dir
        self.max_len = args.max_len
        self.is_test = is_test
        self.feature_name = args.feature_name
        self.num_classes = args.num_classes
        self.fix_encoder = args.fix_encoder
        self.tokenizer = tokenizer
        self.device = device
        self.emb_model = emb_model
        self.dataset_type = args.dataset_type
        self.users_info = pd.read_csv(os.path.join(self.data_dir, "users.csv"))
        self.data_info = pd.read_csv(os.path.join(self.data_dir, f"{label_file}.csv"))
        with open(args.label_mapping_file) as f:
            self.label_map = json.load(f)
        self.label_map = self.label_map[self.dataset_type]
        self.token = {name:[] for name in list(self.users_info.columns)[1:]}
        
        # Get token embedding
        self.token_emb = Feature_all_extract(self.users_info, self.max_len, self.emb_model, self.tokenizer, self.feature_name, self.device)

        # Compute the label for training data
        label_count_dict = {}
        self.label_count = []
        for i in range(self.num_classes):
            label_count_dict[i] = 0

        # Filter out the users
        filt = self.users_info["user_id"].isin(list(self.data_info["user_id"]))
        self.users_info = np.array(self.users_info.loc[filt])
        # self.users_info = self.users_info[:100]

        userid_to_label = self.data_info.set_index('user_id').T.to_dict('list')
        self.label = []
        self.user_id = []
        for user_info in tqdm.tqdm(self.users_info):
            user_id, gender, occupation_titles, interests, recreation_names = user_info
            
            label = str(userid_to_label[user_id][0])
            label_tmp = np.zeros((self.num_classes))

            if is_test:
                if str(interests) == "nan":
                    self.token["gender"].append(self.token["gender"][-1])
                    self.token["occupation_titles"].append(self.token["occupation_titles"][-1])
                    self.token["interests"].append(self.token["interests"][-1])
                    self.token["recreation_names"].append(self.token["recreation_names"][-1])
                self.user_id.append(user_id)
                self.label.append(label_tmp)
                
            if str(interests) != "nan" and str(label) != 'nan':
                self.token["gender"].append(gender)
                self.token["occupation_titles"].append(occupation_titles)
                self.token["interests"].append(interests)
                self.token["recreation_names"].append(recreation_names)

                # Get multi-label ground-truth
                label_list = label.split(' ')
                for label_id in label_list:
                    if self.dataset_type == "subgroups":
                        label_tmp[int(label_id)-1] = 1
                        label_count_dict[int(label_id)-1] += 1
                    elif self.dataset_type == "courses":
                        label_tmp[int(self.label_map[label_id])] = 1
                        label_count_dict[int(self.label_map[label_id])] += 1

                if not is_test:
                    self.label.append(label_tmp)
                    self.user_id.append(user_id)

        # Label count
        for _, values in label_count_dict.items():
            num_label = values
            self.label_count.append(num_label)
        
        assert len(self.user_id) == len(self.label) == len(self.token["interests"])

    def __len__(self):
        return len(self.token["interests"])

    def __getitem__(self, index):
        out_embedding = []
        for i, feature_name in enumerate(self.feature_name):
            tokens = str(self.token[feature_name][index])
            if tokens == "female":
                tokens = "女生"
            elif tokens == "male":
                tokens = "男生"
            elif tokens == "other":
                tokens = "其他"
            elif tokens == "nan":
                continue
            for j, token in enumerate(tokens.split(",")):
                if j == 0:
                    token_embedding = self.token_emb[token].unsqueeze(-1)
                else:
                    token_embedding = torch.cat((token_embedding, self.token_emb[token].unsqueeze(-1)), dim=-1)
            token_embedding = torch.mean(token_embedding, dim=-1)
            out_embedding.append(token_embedding.clone())
            
        out_embedding = torch.stack(out_embedding, dim=-1)
        out_embedding = torch.mean(out_embedding, dim=-1)
            
        return {
            'token_embedding': out_embedding,
            'labels': torch.tensor(self.label[index], dtype=torch.float),
            'user_id': self.user_id[index]
        }