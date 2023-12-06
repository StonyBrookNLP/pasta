import pdb
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
    T5ForConditionalGeneration,
    T5TokenizerFast,
    RobertaTokenizerFast,
    BertTokenizerFast)
from sklearn.metrics import classification_report
from pasta_dataset import T5_dataset, create_train_test_dataset, BERT_RoBERTa_dataset
from copy import deepcopy as cc
import os
import gc
gc.disable()
import torch
from datasets import load_metric
from evaluate import load

def generate_op(model, dl, ds):
    dat_ip = []
    dat_id = []
    dat_op = []
    dat_task_id = []
    dat_op_gen = []
    val_avg_loss = np.array([])

    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(dl):

            max_seq_len = batch['ip_attn_mask'].sum(dim = 1).max().item()
            batch['ip_input_ids'] = batch['ip_input_ids'][:, :max_seq_len].to(args.DEVICE)
            batch['ip_attn_mask'] = batch['ip_attn_mask'][:, :max_seq_len].to(args.DEVICE)
            batch['op_input_ids'] = batch['op_input_ids'].to(args.DEVICE)

            op_generated = model.generate(batch['ip_input_ids'], do_sample=True, max_length=100, top_p=0.93)
            
            # Generate the op for loss computation 
            batch['op_input_ids'][batch['op_input_ids'] == tokenizer.pad_token_id] = -100

            model_op = model(input_ids = batch['ip_input_ids'], 
                            attention_mask = batch['ip_attn_mask'], 
                            labels = batch['op_input_ids'])
            val_avg_loss = np.append(val_avg_loss, model_op.loss.item())

            # Generate the text for the i/p, o/p and generated o/p
            batch['op_input_ids'][batch['op_input_ids'] == -100] = tokenizer.pad_token_id
            op_gen_txt = [tokenizer.decode(op_generated[i], skip_special_tokens=False).replace('<pad>', '').replace('\r', ' ') for i in range(len(op_generated))]
            ip_txt = [tokenizer.decode(batch['ip_input_ids'].cpu()[i], skip_special_tokens=False ).replace('<pad>', '').replace('\r', ' ') for i in range(len(batch['ip_input_ids']))]
            op_txt = [tokenizer.decode(batch['op_input_ids'].cpu()[i], skip_special_tokens=False ).replace('<pad>', '').replace('\r', ' ') for i in range(len(batch['op_input_ids']))]

            dat_ip.extend(ip_txt)
            dat_op.extend(op_txt)
            dat_op_gen.extend(op_gen_txt)

            dat_task_id.extend(batch['task_id'].numpy().tolist())
            dat_id.extend(batch['idx'].numpy().tolist())

    # Creating the output dataframe
    del model_op, batch
    torch.cuda.empty_cache()
    gc.collect()

    temp_df = cc(ds.t5_dataset)
    temp_df['ip'] = temp_df['ip'].apply(lambda x: x.replace('\r', ' '))
    temp_df['model_ip'] = dat_ip
    temp_df['target_op'] = dat_op
    temp_df['gen_op'] = dat_op_gen
    val_avg_loss = val_avg_loss.mean()
    return temp_df, val_avg_loss

def get_perf_task3(df, aid_subset = None):

    target_names = ['Non-Support', 'Support']

    df['target_op'] = df['target_op'].apply(lambda x: [1 if f'<extra_id_{i}>' in x else 0 for i in range(1, 6)])
    df['gen_op'] = df['gen_op'].apply(lambda x: [1 if f'<extra_id_{i}>' in x else 0 for i in range(1, 6)])

    target_op = sum(df['target_op'].tolist(), [])
    gen_op = sum(df['gen_op'].tolist(), [])
    
    clf_report_full = classification_report(target_op, gen_op, target_names=target_names)
    if aid_subset != None:
        df_subset = df.loc[df.AssignmentId.isin(aid_subset)]
        target_op_subset = sum(df_subset['target_op'].tolist(), [])
        gen_op_subset = sum(df_subset['gen_op'].tolist(), [])
        clf_report_subset = classification_report(target_op_subset, gen_op_subset, target_names=target_names)
        print(f'For subset:: Classification report ::\n {clf_report_subset}')

    print(f'For full set:: Classification report ::\n {clf_report_full}')
    return

def get_perf_task8(df, aid_subset = None):
    df['correct_pred'] = df.apply(lambda x: int((str(x['target_op']).strip() == str(x['gen_op']).strip())), axis = 1)
    df['story_type'] = ['story', 'story', 'mod_story', 'mod_story']*int((df.shape[0])/4)

    if aid_subset != None:
        df_subset = df.loc[df.AssignmentId.isin(aid_subset)]
        df2 = df_subset.groupby(['AssignmentId', 'story_type'])['correct_pred'].sum()
        acc = df_subset['correct_pred'].to_numpy().mean()
        cons_acc = sum(df2.to_numpy() == 2)/len(df2)
        print(f'For subset :: Accuracy :: {acc} Contrastive accuracy :: {cons_acc}')

    df2 = df.groupby(['AssignmentId', 'story_type'])['correct_pred'].sum()
    acc = df['correct_pred'].to_numpy().mean()
    cons_acc = sum(df2.to_numpy() == 2)/len(df2)
    print(f'Accuracy :: {acc} Contrastive accuracy :: {cons_acc}')
    return

def get_perf_task7(df, aid_subset = None):
    bertscore = load("bertscore")

    if aid_subset != None:
        df_subset = df.loc[df.AssignmentId.isin(aid_subset)]
        predictions = df_subset['gen_op'].tolist()
        target = df_subset['target_op'].tolist()
        results = bertscore.compute(predictions=predictions, references=target, 
                                    model_type="bert-base-uncased")
        results = np.mean(results['f1'])
        print(f'For subset:: BERT score :: {results}')

    predictions = df['gen_op'].tolist()
    target = df['target_op'].tolist()
    results = bertscore.compute(predictions=predictions, references=target, 
                                model_type="bert-base-uncased")
    results = np.mean(results['f1'])
    print(f'BERT score :: {results}')
    return

def parse():
    parser = argparse.ArgumentParser(description="T5 inference for STATES project !!")
    parser.add_argument('-task_list', type=str, default = "6", help='List of task for which the T5 has to be trained.')
    parser.add_argument('-DEVICE', type=str, default='cuda:0', help='Device on which the model is to be trained: cpu or cuda')    
    parser.add_argument('-random_seed', type=int, default = 1231, help='Random seed for the model')

    parser.add_argument('-task8_setting', type=str, default = '0', help='0: only the story, 1: story marked with justification sentece, 2: Only the justification sentences')
    parser.add_argument('-checkpoint', type=str, help='Model checkpoint')
    parser.add_argument('-model_type', type=str, default = "roberta-large", help='bert-base-uncased OR roberta-base')

    parser.add_argument('-story_type', type=str, help='type of story, is a place holder that is then sent to functions')
    parser.add_argument('-val_story_type', type=str, default= 'full', help='type of story (story or mod_story or full) that is to be used for creating the validation data')
    parser.add_argument('-te_story_type', type=str, default= 'full', help='type of story (story or mod_story or full) that is to be used for creating the test data')
    args = parser.parse_args()    
    return args

if __name__ == '__main__':

    args = parse()
    # Setting the random seed
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    task_no = args.task_list
    args.task_list = [int(t) for t in args.task_list.split()]
            
    # =============================================
    # Defining the dataset and the dataloader
    # =============================================
    data_dict = create_train_test_dataset()
    test_data, val_data = data_dict['te_dat'], data_dict['val_dat']

    args.story_type = args.te_story_type
    te_dataset = T5_dataset(cleaned_df = test_data, args = args, task_list = args.task_list)

    args.story_type = args.val_story_type
    val_dataset = T5_dataset(cleaned_df=val_data, args = args, task_list=args.task_list)

    te_dl = DataLoader(te_dataset, batch_size=16, shuffle=False)
    val_dl = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # =============================================
    # Declaring the model
    # =============================================
    checkpoint_path = args.checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    path_prefix = checkpoint_path.split('/')[-1].replace('.pt', '')

    val_path = f'Val_{args.val_story_type}_{path_prefix}.csv'
    test_path = f'Te_{args.te_story_type}_{path_prefix}.csv'
        
    # Initializing the pre-trained model and loading the checkpoint
    model = T5ForConditionalGeneration.from_pretrained(args.model_type)
    tokenizer = T5TokenizerFast.from_pretrained(args.model_type)
    
    model.load_state_dict(checkpoint)
    model.to(args.DEVICE)

    test_op, test_loss = generate_op(dl = te_dl, ds = te_dataset, model = model)
    val_op, val_loss = generate_op(dl = val_dl, ds = val_dataset, model = model)

    test_op.to_csv(os.path.join('generated_op', test_path), index = False)
    val_op.to_csv(os.path.join('generated_op', val_path), index = False)

    print('-----------------------\nFor test set\n-----------------------')
    print(args)

    print('For test set ::')

    if task_no in ["8"]:
        get_perf_task8(test_op)
    
    if task_no in ["6", "7"]:
        get_perf_task7(test_op)

    if task_no in ["3"]:
        get_perf_task3(test_op)

    print(f'Loss :: {test_loss}')

    print('-----------------------\nFor validation set\n-----------------------')

    if task_no in ["8"]:
        get_perf_task8(val_op)
    
    if task_no in ["6", "7"]:
        get_perf_task7(val_op)

    if task_no in ["3"]:
        get_perf_task3(val_op)

    print(f'Loss :: {val_loss}')