from csv import writer
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import pdb
import pandas as pd
import torch.nn as nn
import argparse
from transformers import (
    AdamW,
    BertModel,
    RobertaModel,
    RobertaTokenizerFast,
    BertTokenizerFast)

from pasta_dataset import BERT_RoBERTa_dataset, create_train_test_dataset
from copy import deepcopy as cc
import os
import gc
gc.disable()
import torch
from torch.utils.tensorboard import SummaryWriter
import sys, datetime, dateutil.relativedelta
from time import time

# =========================================================================================
# Model defination - defining the BERT/RoBERTa based sequence classification model
# =========================================================================================
class classifier(nn.Module):

    def __init__(self, args):
        super().__init__()
        # pdb.set_trace()
        if str(args.task_list[0]) == '3':
            self.no_labels = 5

        if str(args.task_list[0]) == '8':
            self.no_labels = 1
        
        if 'roberta' in args.model_type:
            self.LM_base = RobertaModel.from_pretrained(args.model_type)
        else :
            self.LM_base = BertModel.from_pretrained(args.model_type)

        if 'large' in args.model_type:
            self.linear_op = nn.Linear(in_features=1024, out_features=self.no_labels)
        else:
            self.linear_op = nn.Linear(in_features=768, out_features=self.no_labels)

    def forward(self, input_id, attn_mask):
        LM_op = self.LM_base(input_ids = input_id, attention_mask = attn_mask, return_dict = True)
        LM_op_cls = LM_op['last_hidden_state'][:, 0, :]
        output = self.linear_op(LM_op_cls)               # dim --> (B, C)
        return { 'output': output, 'LM_feat': LM_op_cls}


# =========================================================================================
# Validation - performance metric generation
# =========================================================================================
def validate(model, data_loader, args):

    model.eval()
    val_avg_loss = np.array([])

    with torch.no_grad():

        for idx, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
            max_seq_len = batch['ip_attn_mask'].sum(dim = 1).max().item()
            batch['ip_input_ids'] = batch['ip_input_ids'][:, :max_seq_len].to(args.DEVICE)
            batch['ip_attn_mask'] = batch['ip_attn_mask'][:, :max_seq_len].to(args.DEVICE)

            # Generate the op for loss computation 
            model_op = model(input_id = batch['ip_input_ids'], 
                            attn_mask = batch['ip_attn_mask'])
            model_pred = model_op['output']

            if str(args.task_list[0]) == '8':
                loss = bce_loss(model_pred, batch['op'].to(args.DEVICE).unsqueeze(1).float())
            else:
                loss = bce_loss(model_pred, batch['op'].to(args.DEVICE).float())

            val_avg_loss = np.append(val_avg_loss, loss.item())

    val_avg_loss = val_avg_loss.mean()
    return {'val_loss': val_avg_loss}

def parse():
    parser = argparse.ArgumentParser(description="BERT/RoBERTa modelling for PASTA (clf only) !!")
    parser.add_argument('-batch_size', type=int, default=16, help='batch size')
    parser.add_argument('-num_epochs', type=int, default=7, help='No. of epochs')
    parser.add_argument('-eval_steps', type=int, default=200, help='No. of steps for model validation')
    parser.add_argument('-lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('-wt_decay', type=float, default=1e-6, help='L2 weight decay')
    parser.add_argument('-task_list', type=str, default = "8", help='List of task for which the T5 has to be trained.')
    parser.add_argument('-epoch', type=int, default=-1, help='dummy epoch')
    parser.add_argument('-DEVICE', type=str, default='cuda:0', help='Device on which the model is to be trained: cpu or cuda')
    parser.add_argument('-task8_setting', type=str, default = '1', help=f'0: Story, \n1: union(story, justificaion), \n2: Justification set, \n3: setdiff(Story \ justification)')
    parser.add_argument('-checkpoint_path', type=str, help='Model checkpoint for loading the model')
    parser.add_argument('-model_type', type=str, default = 'bert-base-uncased', help='type of model (t5-base, t5-large, bert-base-uncased, bert-large-uncased, roberta-base or roberta-large)')
    parser.add_argument('-random_seed', type=int, default = 0, help='Random seed for the model')
    parser.add_argument('-story_type', type=str, help='type of story, not input required, it is a place holder that is then sent to functions')
    parser.add_argument('-tr_story_type', type=str, default= 'full', help='type of story (story or mod_story or full) that is to be used for creating the traing data')
    parser.add_argument('-val_story_type', type=str, default= 'full', help='type of story (story or mod_story or full) that is to be used for creating the validation data')
    parser.add_argument('-te_story_type', type=str, default= 'full', help='type of story (story or mod_story or full) that is to be used for creating the test data')
    args = parser.parse_args()    
    return args

if __name__ == '__main__':

    # Start time
    time_init = datetime.datetime.fromtimestamp(time())

    args = parse()
    perf_logger_dict = {'steps': [], 'train_loss': [], 'val_loss': []}

    # Setting the random seed
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    steps = 0

    args.task_list = [int(t) for t in args.task_list.split()]
    print(args)

    # Defining the tokenizer
    if "roberta" in args.model_type:
        tokenizer = RobertaTokenizerFast.from_pretrained(args.model_type)
    else:
        tokenizer = BertTokenizerFast.from_pretrained(args.model_type)

    data_dict = create_train_test_dataset()
    train_data, val_data = data_dict['tr_dat'], data_dict['val_dat']

    task_list = [str(task) for task in args.task_list]
    task_list = "_".join(task_list)
    task_list = task_list

    if (task_list == "8") and (int(args.task8_setting) >0):
        path_prefix = f"t_{task_list}_{args.task8_setting}_m_{args.model_type}_b_{args.batch_size}_lr_{args.lr}_w_{args.wt_decay}_s_{args.random_seed}"
    else:
        path_prefix = f"t_{task_list}_m_{args.model_type}_b_{args.batch_size}_lr_{args.lr}_w_{args.wt_decay}_s_{args.random_seed}"

    if args.tr_story_type != "full":
        path_prefix = f'tr_{args.tr_story_type}_{path_prefix}' 

    print(path_prefix)
    # Defining the tesorboard directory
    writer = SummaryWriter(f'./runs/{path_prefix}')

    # =============================================
    # Defining the dataset and the dataloader
    # =============================================
    args.story_type = args.tr_story_type
    tr_dataset = BERT_RoBERTa_dataset(cleaned_df = train_data, args = args, task_list = args.task_list)
    
    args.story_type = args.val_story_type
    val_dataset = BERT_RoBERTa_dataset(cleaned_df=val_data, args = args, task_list=args.task_list)

    tr_dl = DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # =============================================
    # Declaring the model
    # =============================================
    model = classifier(args)
    model.to(args.DEVICE)
    
    # ========================================================
    # Declaring the optimizer and the loss function (BCE loss)
    # ========================================================
    no_decay = ['bias' , 'gamma', 'beta']
    optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.wt_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
    bce_loss = nn.BCEWithLogitsLoss()

    # ========================================================
    # Training and validating the model
    # ========================================================

    for epoch in range(args.num_epochs):
        args.epoch = epoch
        model.train()

        for idx, batch in enumerate(tr_dl):
            optimizer.zero_grad()

            max_seq_len = batch['ip_attn_mask'].sum(dim = 1).max().item()
            batch['ip_input_ids'] = batch['ip_input_ids'][:, :max_seq_len].to(args.DEVICE)
            batch['ip_attn_mask'] = batch['ip_attn_mask'][:, :max_seq_len].to(args.DEVICE)

            # Generate the op for loss computation 
            model_op = model(input_id = batch['ip_input_ids'], 
                            attn_mask = batch['ip_attn_mask'])
            model_pred = model_op['output']
            # pdb.set_trace()

            if str(args.task_list[0]) == '8':
                tr_step_loss = bce_loss(model_pred, batch['op'].to(args.DEVICE).unsqueeze(1).float())
            else:
                tr_step_loss = bce_loss(model_pred, batch['op'].to(args.DEVICE).float())

            perf_logger_dict['steps'].append(steps) 
            perf_logger_dict['train_loss'].append(tr_step_loss)
            perf_logger_dict['val_loss'].append(-1.0)

            writer.add_scalar('Training loss', tr_step_loss, global_step = steps)
            
            tr_step_loss.backward()
            optimizer.step()

            # Evaluation - check the validation and the test loss
            if ((steps + 1) % args.eval_steps == 0):
                val_op = validate(model = model, data_loader = val_dl, args = args)
                val_loss = val_op['val_loss']
                writer.add_scalar('Validation loss', val_loss, global_step = steps)
                perf_logger_dict['val_loss'][-1] = val_loss


                model.train()
            steps = steps + 1

    # At the last epoch - save the model
    checkpoint_path = path_prefix + f'_epoch_{epoch}' + f'.pt'
    checkpoint_dict = {'model_state_dict': model.state_dict(), 'performance_tracker': perf_logger_dict}
    torch.save(model.state_dict(), os.path.join('model_checkpoint', checkpoint_path))