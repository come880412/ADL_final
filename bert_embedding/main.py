import pandas as pd
import numpy as np
import argparse
import os
from transformers import BertTokenizerFast, logging, AutoModel
import json
import tqdm
import pdb

import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

from dataset import hahow_dataset
from model import RS_model
from config import get_config
from utils import fixed_seed, LinearWarmupCosineAnnealingLR, mapk, calculate_pos_weights

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.set_verbosity_error()

def train(args, emb_model, model, train_loader, val_seen_loader, val_unseen_loader, criterion, optimizer, scheduler, device):
    # Reload checkpoint
    if args.resume:
        print(f"Reload checkpoints from {args.resume}")
        checkpoints = torch.load(args.resume)
        model.load_state_dict(checkpoints["model_state_dict"])
        optimizer.load_state_dict(checkpoints["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoints["scheduler_state_dict"])
        start_epoch = checkpoints["epoch"]
        max_seen_score = checkpoints["best_valid_seen_score"]
        max_unseen_score = checkpoints["best_valid_unseen_score"]
        max_avg_score = checkpoints["best_valid_avg_score"]
    else:
        start_epoch = 0

    print('------------Start training------------')
    max_avg_score = 0
    for epoch in range(start_epoch, args.n_epochs):
        model.train()
        if not args.fix_encoder:
            emb_model.train()
        pbar = tqdm.tqdm(total=len(train_loader), ncols=0, desc="Train[%d/%d]"%(epoch+1, args.n_epochs), unit="step")

        total_loss = 0
        total_data = 0
        lr = optimizer.param_groups[0]['lr']
        for batch in train_loader:
            token_emb, labels = batch["token_embedding"].to(device), batch["labels"].to(device)
            
            optimizer.zero_grad()
            preds = model(token_emb) #(B, num_classes)
            loss = criterion(preds, labels)
            # for pred, label in zip(preds, labels):
            #     pdb.set_trace()
            #     print(pred, label)
            loss.backward()
            optimizer.step()

            total_data += len(labels)
            total_loss += loss.item() * len(labels)
            
            pbar.update()
            pbar.set_postfix(
                loss=f"{(total_loss / total_data):.4f}",
                Lr=f"{lr:.8f}"
            )
        pbar.close()

        seen_score, seen_save_list = valid(args, model, val_seen_loader, criterion, "Valid_seen", device)
        np.savetxt(os.path.join(args.pred_dir, args.model_name, f"val_seen_pred_epoch{epoch+1}_score{seen_score:.4f}.csv"),  seen_save_list, fmt='%s', delimiter=',')
        unseen_score, unseen_save_list = valid(args, model, val_unseen_loader, criterion, "Valid_unseen", device)
        np.savetxt(os.path.join(args.pred_dir, args.model_name, f"val_unseen_pred_epoch{epoch+1}_score{unseen_score:.4f}.csv"),  unseen_save_list, fmt='%s', delimiter=',')

        avg_score = (seen_score + unseen_score) / 2
        
        if avg_score >= max_avg_score:
            best_epoch = epoch
            max_avg_score = avg_score
            max_seen_score, max_unseen_score = seen_score, unseen_score
            print("-------Save model-------")

            checkpoints = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_valid_avg_score": max_avg_score,
                "best_valid_seen_score": max_seen_score,
                "best_valid_unseen_score": max_unseen_score,
            }
            torch.save(checkpoints, os.path.join(args.model_dir, args.model_name, args.dataset_type, f"model_best_epoch{epoch+1}_seen{max_seen_score:.4f}_unseen{max_unseen_score:.4f}.pth"))
            if not args.fix_encoder:
                torch.save(emb_model.state_dict(), os.path.join(args.model_dir, args.model_name, args.dataset_type, f"emb_model_best_epoch{epoch+1}_seen{max_seen_score:.4f}_unseen{max_unseen_score:.4f}.pth"))
        
        checkpoints = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_valid_avg_score": max_avg_score,
            "best_valid_seen_score": max_seen_score,
            "best_valid_unseen_score": max_unseen_score,
        }
        torch.save(checkpoints, os.path.join(args.model_dir, args.model_name, args.dataset_type, f"model_last.pth"))
        if not args.fix_encoder:
            torch.save(emb_model.state_dict(), os.path.join(args.model_dir, args.model_name, args.dataset_type, f"emb_model_last.pth"))

        print(f"Best map@50 score | Epoch {best_epoch + 1} | val_seen_score {max_seen_score:.4f} | val_unseen_score {max_unseen_score:.4f}\n")
        scheduler.step()

def valid(args, model, val_loader, criterion, data_name, device):
    model.eval()
    if not args.fix_encoder:
        emb_model.eval()
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

def main(args):
    with open(args.label_mapping_file) as f:
        label_map = json.load(f)
    label_map = label_map[args.dataset_type]
    args.num_classes = len(label_map)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = BertTokenizerFast.from_pretrained(args.model_name, max_length=args.max_len, padding='max_length', truncation=True)
    emb_model = AutoModel.from_pretrained(args.model_name, output_hidden_states = True).eval().to(device) # Whether the model returns all hidden-states.
    model = RS_model(args.num_classes).to(device)

    name = "" if args.dataset_type == "courses" else "_group"
    train_data = hahow_dataset(args, tokenizer, emb_model, device, label_file=f"train{name}")
    valid_seen_data = hahow_dataset(args, tokenizer, emb_model, device, label_file=f"val_seen{name}")
    valid_unseen_data = hahow_dataset(args, tokenizer, emb_model, device, label_file=f"val_unseen{name}")
    
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.n_cpus, drop_last=True)
    val_seen_loader = DataLoader(valid_seen_data, batch_size=args.batch_size, shuffle=False, num_workers=args.n_cpus, drop_last=False)
    val_unseen_loader = DataLoader(valid_unseen_data, batch_size=args.batch_size, shuffle=False, num_workers=args.n_cpus, drop_last=False)

    if args.fix_encoder:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay = args.weight_decay)
    else:
        optimizer = torch.optim.AdamW([
                    {'params': model.parameters(), 'lr': args.lr},
                    {'params': emb_model.parameters(), 'lr': args.lr}], weight_decay = args.weight_decay)
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_start_lr=args.warmup_lr, warmup_epochs=args.warmup_epochs, max_epochs=args.n_epochs, eta_min=1e-5)

    if args.unbalanced_weight:
        print('----------Use unbalanced weight----------')
        pos_weight = calculate_pos_weights(train_data.label_count, len(train_data))
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)
    else:
        criterion = torch.nn.BCEWithLogitsLoss().to(device)

    train(args, emb_model, model, train_loader, val_seen_loader, val_unseen_loader, criterion, optimizer, scheduler, device)

if __name__ == "__main__":
    args = get_config()
    fixed_seed(args.seed)
    os.makedirs(os.path.join(args.model_dir, args.model_name, args.dataset_type), exist_ok=True)
    os.makedirs(os.path.join(args.pred_dir, args.model_name, args.dataset_type), exist_ok=True)
    
    main(args)
    # print(ids, mask, token_type_ids)
    # print(token_embeddings.shape)