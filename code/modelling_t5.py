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
    T5ForConditionalGeneration,
    T5TokenizerFast)
from pasta_dataset import T5_dataset, create_train_test_dataset
from copy import deepcopy as cc
import os
import gc
gc.disable()
import torch
from torch.utils.tensorboard import SummaryWriter
import sys

# =========================================================================================
# Validation - performance metric generation (CE loss)
# =========================================================================================
def validate(model, data_loader, args):

    model.eval()
    val_avg_loss = np.array([])

    with torch.no_grad():

        for idx, batch in tqdm(enumerate(data_loader), total = len(data_loader)):
            max_seq_len = batch['ip_attn_mask'].sum(dim = 1).max().item()
            batch['ip_input_ids'] = batch['ip_input_ids'][:, :max_seq_len].to(args.DEVICE)
            batch['ip_attn_mask'] = batch['ip_attn_mask'][:, :max_seq_len].to(args.DEVICE)

            batch['op_input_ids'][batch['op_input_ids'] == tokenizer.pad_token_id] = -100
            batch['op_input_ids'] = batch['op_input_ids'].to(args.DEVICE)

            # Generate the op for loss computation 
            model_op = model(input_ids = batch['ip_input_ids'], 
                            attention_mask = batch['ip_attn_mask'], 
                            labels = batch['op_input_ids'])
            val_avg_loss = np.append(val_avg_loss, model_op.loss.item())

    val_avg_loss = val_avg_loss.mean()
    return {'val_loss': val_avg_loss}

def parse():
    parser = argparse.ArgumentParser(description="T5 finetuning on PASTA story-state inference task!!")
    parser.add_argument('-batch_size', type=int, default=16, help='batch size')
    parser.add_argument('-num_epochs', type=int, default=2, help='No. of epochs')
    parser.add_argument('-eval_steps', type=int, default=1000, help='No. of steps for model validation')
    parser.add_argument('-lr', type=float, default=1e-4, help='Learning rate of AdamW')
    parser.add_argument('-wt_decay', type=float, default=1e-7, help='L2 weight decay of AdamW')
    parser.add_argument('-task_list', type=str, default = "8", help='List of task for which the T5 has to be trained. Simply specify char "8" for this task.')
    parser.add_argument('-epoch', type=int, default=-1, help='Dummy epoch params (dont care)')
    parser.add_argument('-DEVICE', type=str, default='cuda:0', help='Device on which the model is to be trained: cpu or cuda')
    parser.add_argument('-task8_setting', type=str, default = '1', help=f'0: Story, \n1: Union(story, justification), \n2: Justification set, \n3: setdiff(Story \ justification)')
    parser.add_argument('-checkpoint_path', type=str, help='Model checkpoint for loading the model')
    parser.add_argument('-model_type', type=str, default="t5-base", help='type of model (t5-base, t5-large, etc.)')
    parser.add_argument('-random_seed', type=int, default = 0, help='Random seed for the model')
    parser.add_argument('-story_type', type=str, help='Place holder param')
    parser.add_argument('-tr_story_type', type=str, default= 'full', help='type of story (story or mod_story or both) used for creating the traing data')
    parser.add_argument('-val_story_type', type=str, default= 'full', help='type of story (story or mod_story or both) used for creating the validation data')
    parser.add_argument('-te_story_type', type=str, default= 'full', help='type of story (story or mod_story or both) used for creating the test data')
    args = parser.parse_args()    
    return args

if __name__ == '__main__':
    args = parse()
    perf_logger_dict = {'steps': [], 'train_loss': [], 'val_loss': []}

    # Setting the random seed
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    steps = 0
    # val_history = np.array([np.inf, np.inf, np.inf])

    args.task_list = [int(t) for t in args.task_list.split()]
    print(args)
    tokenizer = T5TokenizerFast.from_pretrained(args.model_type)

    # pdb.set_trace()
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
    tr_dataset = T5_dataset(cleaned_df = train_data, args = args, task_list = args.task_list)
    
    args.story_type = args.val_story_type
    val_dataset = T5_dataset(cleaned_df=val_data, args = args, task_list=args.task_list)

    tr_dl = DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size=30, shuffle=False)

    # =============================================
    # Declaring the model
    # =============================================
    model = T5ForConditionalGeneration.from_pretrained(args.model_type)
    model.to(args.DEVICE)
    
    # ========================================================
    # Declaring the optimizer and the loss function (BCE loss)
    # ========================================================
    no_decay = ['bias' , 'gamma', 'beta']
    optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.wt_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)

    # ========================================================
    # Training and validating the model
    # ========================================================
    
    for epoch in range(args.num_epochs):
        args.epoch = epoch
        model.train()

        for idx, batch in tqdm(enumerate(tr_dl), total = len(tr_dl)):
            optimizer.zero_grad()

            max_seq_len = batch['ip_attn_mask'].sum(dim = 1).max().item()
            batch['ip_input_ids'] = batch['ip_input_ids'][:, :max_seq_len].to(args.DEVICE)
            batch['ip_attn_mask'] = batch['ip_attn_mask'][:, :max_seq_len].to(args.DEVICE)  

            batch['op_input_ids'][batch['op_input_ids'] == tokenizer.pad_token_id] = -100
            batch['op_input_ids'] = batch['op_input_ids'].to(args.DEVICE)

            model_op = model(input_ids = batch['ip_input_ids'], 
                            attention_mask = batch['ip_attn_mask'], 
                            labels = batch['op_input_ids'])
            tr_step_loss = model_op.loss.item()

            perf_logger_dict['steps'].append(steps) 
            perf_logger_dict['train_loss'].append(tr_step_loss)
            perf_logger_dict['val_loss'].append(-1.0)

            writer.add_scalar('Training loss', tr_step_loss, global_step = steps)            

            model_op.loss.backward()
            optimizer.step()

            # Evaluation - check the validation and the test loss
            if ((steps + 1) % args.eval_steps == 0):
                val_op = validate(model = model, data_loader = val_dl, args = args)
                val_loss = val_op['val_loss']
                writer.add_scalar('Validation loss', val_loss, global_step = steps)
                perf_logger_dict['val_loss'][-1] = val_loss
                model.train()

            steps = steps + 1

    # save the model
    checkpoint_path = path_prefix + f'_epoch_{epoch}' + f'.pt'
    checkpoint_dict = {'model_state_dict': model.state_dict(), 'performance_tracker': perf_logger_dict}
    torch.save(model.state_dict(), os.path.join('model_checkpoint', checkpoint_path))