from torch.utils.data import Dataset, DataLoader
import torch

import pandas as pd
import os
import pdb
import numpy as np
import tqdm
import json

# from Feature_extractor import Feature_extract, Feature_all_extract
# interests_all = ['語言_韓文', '生活品味_寵物', '攝影_影像創作', '語言_日文', '職場技能_職場溝通', '手作_手工印刷', '投資理財_比特幣', '音樂_音樂理論', '藝術_表演藝術', '手作_手作小物', '攝影_攝影理論', '手作_更多手作', '設計_平面設計', '設計_設計理論', '程式_手機程式開發', '手作_刺繡', '程式_AI 人工智慧', '生活品味_靈性發展', '程式_遊戲開發', '程式_區塊鏈', '攝影_影視創作', '藝術_素描', '生活品味_壓力舒緩', '人文_更多人文', '音樂_DJ', '生活品味_數學', '生活品味_居家', '語言_翻譯', '職場技能_資料彙整', '職場技能_更多職場技能', '投資理財_量化交易', '音樂_樂器', '投資理財_更多投資理財', '手作_模型', '生活品味_更多生活品味', '程式_資訊安全', '行銷_數位行銷', '生活品味_運動', '設計_應用設計', '職場技能_文書處理', '程式_網頁後端', '行銷_數據分析', '語言_西班牙文', '職場技能_創業', '行銷_社群行銷', '手作_篆刻', '程式_程式入門', '投資理財_金融商品', '攝影_商業攝影', '生活品味_心靈成長與教育', '音樂_音樂創作', '音樂_人聲', '設計_網頁設計', '程式_資料科學', '程式_更多程式', '職場技能_求職', '職場技能_個人品牌經營', '手作_手工書', '攝影_動態攝影', '行銷_文案', '語言_英文', '藝術_電腦繪圖', '藝術_更多藝術', '程式_程式語言', '程式_量化分析', '程式_軟體程式開發與維護', '生活品味_護膚保養與化妝', '職場技能_獨立接案', '設計_動態設計', '設計_更多設計', '音樂_更多音樂', '職場技能_產品設計', '藝術_繪畫與插畫', '職場技能_效率提升', '攝影_更多攝影', '藝術_字體設計', '投資理財_理財', '投資理財_投資觀念', '語言_歐洲語言', '攝影_後製剪輯', '藝術_角色設計', '設計_介面設計', '生活品味_親子教育', '藝術_手寫字', '人文_文學', '生活品味_烹飪料理與甜點', '程式_網站架設', '藝術_色彩學', '程式_網頁前端', '手作_氣球', '人文_社會科學', '行銷_更多行銷', '程式_程式思維', '語言_更多語言', '程式_程式理財']
class hahow_dataset(Dataset):

    def __init__(self, args, tokenizer, emb_model, device, is_test=False, label_file="train"):
        # Initialization
        self.data_dir = args.data_dir
        # self.max_len = args.max_len
        self.is_test = is_test
        self.feature_name = args.feature_name
        self.num_classes = args.num_classes
        # self.fix_encoder = args.fix_encoder
        # self.tokenizer = tokenizer
        self.device = device
        # self.emb_model = emb_model
        self.dataset_type = args.dataset_type
        self.users_info = pd.read_csv(os.path.join(self.data_dir, "users.csv"))
        print("===============\n",self.users_info.nunique())
        self.seen = label_file
        self.data_info = pd.read_csv(os.path.join(self.data_dir, f"{label_file}.csv"))
        with open(args.label_mapping_file) as f:
            self.label_map = json.load(f)
        self.label_map = self.label_map[self.dataset_type]
        # self.token = {name:[] for name in list(self.users_info.columns)[1:]}
        self.interests_all = ['語言_韓文', '生活品味_寵物', '攝影_影像創作', '語言_日文', '職場技能_職場溝通', '手作_手工印刷', '投資理財_比特幣', '音樂_音樂理論', '藝術_表演藝術', '手作_手作小物', '攝影_攝影理論', '手作_更多手作', '設計_平面設計', '設計_設計理論', '程式_手機程式開發', '手作_刺繡', '程式_AI 人工智慧', '生活品味_靈性發展', '程式_遊戲開發', '程式_區塊鏈', '攝影_影視創作', '藝術_素描', '生活品味_壓力舒緩', '人文_更多人文', '音樂_DJ', '生活品味_數學', '生活品味_居家', '語言_翻譯', '職場技能_資料彙整', '職場技能_更多職場技能', '投資理財_量化交易', '音樂_樂器', '投資理財_更多投資理財', '手作_模型', '生活品味_更多生活品味', '程式_資訊安全', '行銷_數位行銷', '生活品味_運動', '設計_應用設計', '職場技能_文書處理', '程式_網頁後端', '行銷_數據分析', '語言_西班牙文', '職場技能_創業', '行銷_社群行銷', '手作_篆刻', '程式_程式入門', '投資理財_金融商品', '攝影_商業攝影', '生活品味_心靈成長與教育', '音樂_音樂創作', '音樂_人聲', '設計_網頁設計', '程式_資料科學', '程式_更多程式', '職場技能_求職', '職場技能_個人品牌經營', '手作_手工書', '攝影_動態攝影', '行銷_文案', '語言_英文', '藝術_電腦繪圖', '藝術_更多藝術', '程式_程式語言', '程式_量化分析', '程式_軟體程式開發與維護', '生活品味_護膚保養與化妝', '職場技能_獨立接案', '設計_動態設計', '設計_更多設計', '音樂_更多音樂', '職場技能_產品設計', '藝術_繪畫與插畫', '職場技能_效率提升', '攝影_更多攝影', '藝術_字體設計', '投資理財_理財', '投資理財_投資觀念', '語言_歐洲語言', '攝影_後製剪輯', '藝術_角色設計', '設計_介面設計', '生活品味_親子教育', '藝術_手寫字', '人文_文學', '生活品味_烹飪料理與甜點', '程式_網站架設', '藝術_色彩學', '程式_網頁前端', '手作_氣球', '人文_社會科學', '行銷_更多行銷', '程式_程式思維', '語言_更多語言', '程式_程式理財']
        print("len",len(self.interests_all))
        # Get token embedding
        # self.token_emb = Feature_all_extract(self.users_info, self.max_len, self.emb_model, self.tokenizer, self.feature_name, self.device)

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
        self.interests_feats = []
        interests_all = []
        for user_info in tqdm.tqdm(self.users_info):
            user_id, gender, occupation_titles, interests, recreation_names = user_info
            
            label = str(userid_to_label[user_id][0])
            label_tmp = np.zeros((self.num_classes))

            if is_test:
                if str(interests) == "nan":
                    # self.token["gender"].append(self.token["gender"][-1])
                    # self.token["occupation_titles"].append(self.token["occupation_titles"][-1])
                    # self.token["interests"].append(self.token["interests"][-1])
                    # self.token["recreation_names"].append(self.token["recreation_names"][-1])
                    self.interests_feats.append(self.interests_feats[-1])
                self.user_id.append(user_id)
                self.label.append(label_tmp)
                
            if str(interests) != "nan" and str(label) != 'nan':
                # self.token["gender"].append(gender)
                # self.token["occupation_titles"].append(occupation_titles)
                
                self.interests = interests.split(",")
                # interests_all += self.interests
                # self.interests_all = list(set(interests_all))

                self.interests_feats.append(self.interests)

                # self.token["recreation_names"].append(recreation_names)

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
        # print("self.token[interests]",len(self.token["interests"]),self.token["interests"])

        
        # Label count
        for _, values in label_count_dict.items():
            num_label = values
            self.label_count.append(num_label)
        
        assert len(self.user_id) == len(self.label) 

    def __len__(self):
        return len(self.label) 

    def __getitem__(self, index):
        # out_embedding = []
        # for i, feature_name in enumerate(self.feature_name):
        #     tokens = str(self.token[feature_name][index])
        #     if tokens == "female":
        #         tokens = "女生"
        #     elif tokens == "male":
        #         tokens = "男生"
        #     elif tokens == "other":
        #         tokens = "其他"
        #     elif tokens == "nan":
        #         continue
        # print("self.interests_all",len(self.interests_all))
        # print("self.interests_feats",self.interests_feats[index])
        # print("self.interests_all",self.interests_all)
        self.onehot_token = [int(elem in self.interests_feats[index]) for elem in self.interests_all]
        # interest_list = set(self.token["interests"])
        # print("self.onehot_token",self.onehot_token)
        # print("self.interests",self.interests)
        # print("self.onehot_token",self.onehot_token)

        # print("self.label[index]",self.label[index]) 
        # print("self.user_id[index]",self.user_id[index]) 
        return (
            torch.tensor(self.onehot_token, dtype=torch.float),
            torch.tensor(self.label[index], dtype=torch.float),
            self.user_id[index]
        )