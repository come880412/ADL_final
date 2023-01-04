import pandas as pd
import numpy as np
import argparse
import os
from transformers import BertTokenizerFast, logging, AutoModel
import json
import tqdm
import pdb

import torch
from torch.utils.data import DataLoader

from dataset import hahow_dataset
from model import RS_model
from config import get_config
from utils import fixed_seed, LinearWarmupCosineAnnealingLR, mapk, calculate_pos_weights

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.set_verbosity_error()
# magic = ['6156a77fdf426a0007cc5fe1']
def valid(args, model, val_loader, criterion, data_name, device):
    model.eval()
    pbar = tqdm.tqdm(total=len(val_loader), ncols=0, desc=data_name, unit="step")

    with torch.no_grad():
        total_loss = 0
        total_data = 0

        pred_list = []
        label_list = []
        save_list = [["user_id", "prediction", "label_id"]]
        for batch in val_loader:
            token_emb, labels = batch["token_embedding"].to(device), batch["labels"].to(device)
            user_id = batch["user_id"]

            preds = model(token_emb) #(B, num_classes)
            loss = criterion(preds, labels)

            total_data += len(labels)
            total_loss += loss.item() * len(labels)
            
            pbar.update()
            pbar.set_postfix(
                loss=f"{(total_loss / total_data):.4f}",
            )

            preds = torch.sigmoid(preds).cpu().detach().numpy()
            labels = labels.cpu().detach().numpy()

            for i, (pred, label) in enumerate(zip(preds, labels)):
                # print(pred, label)
                # pdb.set_trace()
                index_sort = np.argsort(pred)[::-1]
                pred_label = np.where(pred >= args.thres)[0].tolist()
                label_id = np.where(label >= 1.0)[0].tolist()

                pred_list.append(index_sort[:len(pred_label)])
                label_list.append(label_id)

                save_list.append([user_id[i], index_sort[:len(pred_label)], label_id])
        
        map_score = mapk(label_list, pred_list, args.k)
        pbar.set_postfix(
            loss=f"{(total_loss / total_data):.4f}",
            map_score=f"{map_score:.3f}"
        )

        pbar.close()
        return map_score, save_list

def test(args, model, test_loader, data_name, device, label_map):
    model.eval()
    pbar = tqdm.tqdm(total=len(test_loader), ncols=0, desc=data_name, unit="step")

    id_to_courses = {v:k for k, v in label_map.items()}
    with torch.no_grad():
        pred_list = []
        label_list = []

        title_name = "course_id" if args.dataset_type == "courses" else "subgroup"
        save_list = [["user_id", title_name]]
        for i, batch in enumerate(test_loader):
            # token_emb, user_id = batch["token_embedding"].to(device), batch["user_id"]
            feats, labels, user_id = batch
            preds = model(feats.to(device)) #(B, num_classes)

            
            pbar.update()

            preds = torch.sigmoid(preds).cpu().detach().numpy()

            for i, pred in enumerate(preds):
                index_sort = np.argsort(pred)[::-1]
                pred_label = np.where(pred >= args.thres)[0].tolist()

                dominant_pred = index_sort[:50]
                try:
                    dominant_pred.remove(665)
                except:
                    print("except")
                # print("type",type(dominant_pred))
                dominant_pred = np.array([665] + dominant_pred.tolist())
                # print("dominant_pred",len(dominant_pred))
                save_str = ""
                for j, dominant in enumerate(dominant_pred):
                    save_str +=  id_to_courses[int(dominant)] if args.dataset_type == "courses" else str(dominant+1)
                    if j != len(dominant_pred) -1:
                        save_str += " "

                save_list.append([user_id[i], save_str])

        pbar.close()
        return save_list

def main(args):
    with open(args.label_mapping_file) as f:
        label_map = json.load(f)
    label_map = label_map[args.dataset_type]
    args.num_classes = len(label_map)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # tokenizer = BertTokenizerFast.from_pretrained(args.model_name, max_length=args.max_len, padding='max_length', truncation=True)
    # emb_model = AutoModel.from_pretrained(args.model_name, output_hidden_states = True).eval().to(device) # Whether the model returns all hidden-states.
    model = RS_model(args.num_classes).eval().to(device)

    if args.resume:
        # print(f"Load checkpoints from {args.resume}")
        checkpoints = torch.load(args.resume)
        model.load_state_dict(checkpoints["model_state_dict"])

    name = "" if args.dataset_type == "courses" else "_group"
    # valid_seen_data = hahow_dataset(args, tokenizer, emb_model, device, label_file=f"val_seen{name}")
    # valid_unseen_data = hahow_dataset(args, tokenizer, emb_model, device, label_file=f"val_unseen{name}")
    tokenizer = 0
    emb_model = 0
    test_seen_data = hahow_dataset(args, tokenizer, emb_model, device, is_test=True, label_file=f"test_seen{name}")
    test_unseen_data = hahow_dataset(args, tokenizer, emb_model, device, is_test=True, label_file=f"test_unseen{name}")
    
    # val_seen_loader = DataLoader(valid_seen_data, batch_size=args.batch_size, shuffle=False, num_workers=args.n_cpus, drop_last=False)
    # val_unseen_loader = DataLoader(valid_unseen_data, batch_size=args.batch_size, shuffle=False, num_workers=args.n_cpus, drop_last=False)
    test_seen_loader = DataLoader(test_seen_data, batch_size=args.batch_size, shuffle=False, num_workers=args.n_cpus, drop_last=False)
    test_unseen_loader = DataLoader(test_unseen_data, batch_size=args.batch_size, shuffle=False, num_workers=args.n_cpus, drop_last=False)

    criterion = torch.nn.BCEWithLogitsLoss().to(device)

    # valid_map_seen_score = valid(args, model, val_seen_loader, criterion, "Valid_seen", device)
    # valid_map_unseen_score = valid(args, model, val_unseen_loader, criterion, "Valid_unseen", device)
    test_seen_list = test(args, model, test_seen_loader, "Test_seen", device, label_map)
    test_unseen_list = test(args, model, test_unseen_loader, "Test_unseen", device, label_map)
    print("---------Save precdiction---------")
    np.savetxt(os.path.join(args.pred_dir, args.pred_dir, f"seen_{args.dataset_type}.csv"),  test_seen_list, fmt='%s', delimiter=',')
    np.savetxt(os.path.join(args.pred_dir, args.pred_dir, f"unseen_{args.dataset_type}.csv"),  test_unseen_list, fmt='%s', delimiter=',')

if __name__ == "__main__":
    args = get_config()
    fixed_seed(args.seed)
    os.makedirs(os.path.join(args.pred_dir, args.pred_dir), exist_ok=True)
    
    main(args)
    # print(ids, mask, token_type_ids)
    # print(token_embeddings.shape)