import torch
import pdb
import numpy as np 

def Feature_extract(tokens, max_len, emb_model, tokenizer, device):
    token_emb = {}

    for token in tokens:
        tokenized_result = tokenizer.encode_plus(
            token,
            None,
            add_special_tokens=True,
            max_length=max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )
        mask = torch.tensor(tokenized_result['attention_mask'], dtype=torch.long).unsqueeze(0).to(device)
        input_ids = torch.tensor(tokenized_result['input_ids'], dtype=torch.long).unsqueeze(0).to(device)

        embedding = emb_model(input_ids, mask)
        embedding = torch.stack(embedding[2], dim=-1).squeeze(0).cpu().detach()[0] # Get [CLS] token embedding

        # embedding = torch.mean(embedding, dim=0) # (dim, 13)

        token_emb[token] = embedding

    return token_emb

def Feature_all_extract(users_info, max_len, emb_model, tokenizer, feature_name, device):
    token_emb = {}
    for title in feature_name: # Discard user_id
        # Get the unique interests
        features = np.array(users_info[title].dropna())
        unique_features = set()
        for feature in features:
            records = feature.split(",")
            for record in records:
                unique_features.add(record) 
        unique_features = list(unique_features)

        for token in unique_features:
            if token == "female":
                token = "女生"
            elif token == "male":
                token = "男生"
            elif token == "other":
                token = "其他"

            tokenized_result = tokenizer.encode_plus(
                token,
                None,
                add_special_tokens=True,
                max_length=max_len,
                padding='max_length',
                return_token_type_ids=True,
                truncation=True
            )
            mask = torch.tensor(tokenized_result['attention_mask'], dtype=torch.long).unsqueeze(0).to(device)
            input_ids = torch.tensor(tokenized_result['input_ids'], dtype=torch.long).unsqueeze(0).to(device)

            embedding = emb_model(input_ids, mask)
            embedding = torch.stack(embedding[2], dim=-1).squeeze(0).cpu().detach()[0] # Get [CLS] token embedding

            # embedding = torch.mean(embedding, dim=0) # (dim, 13)
            
            token_emb[token] = embedding # (768, 13)

    return token_emb