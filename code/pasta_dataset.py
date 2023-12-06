import numpy as np
from transformers import (T5TokenizerFast, BertTokenizerFast, RobertaTokenizerFast)
import torch
import os
import pandas as pd

# ============================================================================================================================
# Reading the worker responses
# Cleaning the data
# Creating the train-test split
# ============================================================================================================================
def create_train_test_dataset(root = './data'):

    te_data = pd.read_json(os.path.join(root, 'te_data.jsonl'), lines=True)
    tr_data = pd.read_json(os.path.join(root, 'tr_data.jsonl'), lines=True)
    val_data = pd.read_json(os.path.join(root, 'val_data.jsonl'), lines=True)

    print(f'----\nTrain_data shape :: {tr_data.shape}\ntest_data shape :: {te_data.shape}\nval_data shape :: {val_data.shape}\n\----')
    return {'tr_dat': tr_data, 'te_dat': te_data, 'val_dat': val_data} 


# ============================================================================================================================
# Creating the modelling dataset from the cleaned worker responses (dat -> Dataframe)
# ============================================================================================================================
def get_model_ready_dataset(dat):
    req_cols = ['AssignmentId', 'Input.Title', 'Input.storyid',
                'Input.line1', 'Input.line2', 'Input.line3','Input.line4', 'Input.line5', 'Answer.assertion',
                'Answer.line1.on', 'Answer.line2.on', 'Answer.line3.on', 'Answer.line4.on', 'Answer.line5.on',
                'Answer.mod_assertion', 'Answer.mod_line1', 'Answer.mod_line2', 'Answer.mod_line3', 'Answer.mod_line4', 'Answer.mod_line5']
    dat_modelling = dat.loc[:, req_cols]
    for col in dat_modelling.columns:
        dat_modelling[col] = dat_modelling[col]

    dat_modelling.columns = [col.replace('Input.', '') for col in dat_modelling.columns]
    dat_modelling.columns = [col.replace('Answer.', '') for col in dat_modelling.columns]
    dat_modelling.columns = [col.replace('.', '_') for col in dat_modelling.columns]

    for i in range(1, 6):
        dat_modelling.loc[:, f'line{i}_on'] = dat_modelling.loc[:, f'line{i}_on'].astype(int)
        dat_modelling.loc[:, f'line{i}_changed'] = dat_modelling.apply(lambda x: x[f'line{i}'] != x[f'mod_line{i}'], axis=1).astype(int)

    # Column for story 
    dat_modelling['Story'] = dat_modelling.apply(lambda x: " ".join( [x[f'line{i}'] for i in range(1, 6)] ), axis=1)
    dat_modelling['Story_marked'] = dat_modelling.apply(lambda x: " ".join( ['* ' * x[f'line{i}_on'] + x[f'line{i}'] for i in range(1, 6)] ), axis=1)
    dat_modelling['Story_subset_marked'] = dat_modelling.apply(lambda x: " ".join( ['* ' * x[f'line{i}_on'] + x[f'line{i}'] for i in range(1, 6) if (x[f'line{i}_on'] == 1) ] ), axis=1)
    dat_modelling['Story_without_marked'] = dat_modelling.apply(lambda x: " ".join( ['* ' * x[f'line{i}_on'] + x[f'line{i}'] for i in range(1, 6) if (x[f'line{i}_on'] != 1) ] ), axis=1)

    dat_modelling['mod_Story'] = dat_modelling.apply(lambda x: " ".join( [x[f'mod_line{i}'] for i in range(1, 6)] ), axis=1)
    dat_modelling['mod_Story_marked'] = dat_modelling.apply(lambda x: " ".join( ['* ' * x[f'line{i}_changed'] + x[f'mod_line{i}'] for i in range(1, 6)] ), axis=1)
    dat_modelling['mod_Story_subset_marked'] = dat_modelling.apply(lambda x: " ".join( ['* ' * x[f'line{i}_changed'] + x[f'mod_line{i}'] for i in range(1, 6) if (x[f'line{i}_changed'] == 1) ] ), axis=1)
    dat_modelling['mod_Story_without_marked'] = dat_modelling.apply(lambda x: " ".join( ['* ' * x[f'line{i}_changed'] + x[f'mod_line{i}'] for i in range(1, 6) if (x[f'line{i}_changed'] != 1) ] ), axis=1)

    dat_modelling['Story_delim'] = dat_modelling.apply(lambda x: [ [f'<extra_id_{i}>: ' + x[f'line{i}'] ] for i in range(1, 6)], axis=1).apply(lambda x: " ".join(sum(x, [])))
    dat_modelling['mod_Story_delim'] = dat_modelling.apply(lambda x: [ [f'<extra_id_{i}>: ' + x[f'mod_line{i}'] ] for i in range(1, 6)], axis=1).apply(lambda x: " ".join(sum(x, [])))

    dat_modelling['changed_Story_delim'] = dat_modelling.apply(lambda x: [ [f'<extra_id_{i}>: ' + x[f'line{i}'] ] for i in range(1, 6) if x[f'line{i}_changed'] == 1], axis=1)
    dat_modelling['changed_Story_delim'] = dat_modelling['changed_Story_delim'].apply(lambda x: " ".join(sum(x, [])))

    dat_modelling['changed_mod_Story_delim'] = dat_modelling.apply(lambda x: [ [f'<extra_id_{i}>: ' + x[f'mod_line{i}'] ] for i in range(1, 6) if   x[f'line{i}_changed'] == 1], axis=1)
    dat_modelling['changed_mod_Story_delim'] = dat_modelling['changed_mod_Story_delim'].apply(lambda x: " ".join(sum(x, [])))
    return dat_modelling


# ============================================================================================================================
# T5 format: Conditional state generation - given a story generate a likely state based on the marked sentences
#   i/p format: [STORY*: story with few of its sentences marked]
#   o/p format: state inferred from the marked sentences
# ============================================================================================================================
def get_task2_dataset(dat_modelling, args):
    t5_dat_task2 = []
    task_no = 2
    task_prefix = 'generate state'

    for idx, row in dat_modelling.iterrows():
        row = row.to_dict()

        # For (story, marked sentence subset marked by *) -> inferred state
        t5_ip1 =  f'story: ' + row['Story_marked']
        t5_op1 = row['assertion']

        # For (mod_story, mod_state) -> justification lines
        t5_ip2 =  f'story: ' + row['mod_Story_marked']
        t5_op2 = row['mod_assertion']

        t5_dict = {'task_no': task_no, 'AssignmentId': row['AssignmentId'], 
                    'storyid': row['storyid'], 'prefix': task_prefix, 
                    'ip': t5_ip1, 'op': t5_op1}
        t5_dat_task2.append(t5_dict)
        
        t5_dict = {'task_no': task_no, 'AssignmentId': row['AssignmentId'], 
                   'storyid': row['storyid'], 'prefix': task_prefix, 
                   'ip': t5_ip2, 'op': t5_op2}
        t5_dat_task2.append(t5_dict)

    t5_dat_task2 = pd.DataFrame(t5_dat_task2)
    return t5_dat_task2


# ============================================================================================================================
# T5 format: choose the justification sentences for a given [story + state]
#   i/p format: [STORY: story_description] + [STATE: state_description]
#   o/p format: line1, line3, ... (basically list of lines that together justifies the state)
# ============================================================================================================================
def get_task3_dataset(dat_modelling, args):
    t5_dat_task3 = []
    task_no = 3
    task_prefix = 'multichoice'

    for idx, row in dat_modelling.iterrows():
        row = row.to_dict()

        # For (story, state) -> justification lines
        t5_ip1 =  f'story: ' + row['Story_delim'] + ' state: ' + row['assertion']
        t5_op1 = ", ".join([f'<extra_id_{i}>' for i in range(1, 6) if row[f'line{i}_on'] == 1])
        bert_op1 = [1 if (row[f'line{i}_on'] == 1) else 0 for i in range(1, 6) ]

        # For (mod_story, mod_state) -> justification lines
        t5_ip2 =  f'story: ' + row['mod_Story_delim'] + ' state: ' + row['mod_assertion']
        t5_op2 = ", ".join([f'<extra_id_{i}>' for i in range(1, 6) if row[f'line{i}_changed'] == 1])
        bert_op2 = [1 if (row[f'line{i}_changed'] == 1) else 0 for i in range(1, 6) ]

        t5_dict = {'task_no': task_no, 'AssignmentId': row['AssignmentId'], 
                    'storyid': row['storyid'], 'prefix': task_prefix, 
                    'ip': t5_ip1, 'op': t5_op1, 'bert_op': bert_op1}
        t5_dat_task3.append(t5_dict)
        
        t5_dict = {'task_no': task_no, 'AssignmentId': row['AssignmentId'], 
                   'storyid': row['storyid'], 'prefix': task_prefix, 
                   'ip': t5_ip2, 'op': t5_op2, 'bert_op': bert_op2}
        t5_dat_task3.append(t5_dict)

    t5_dat_task3 = pd.DataFrame(t5_dat_task3)
    return t5_dat_task3

# ============================================================================================================================
# T5 format: Counterfactual story revision
#   i/p format: story: story_description state: state_description (unlikely-state)
#   o/p format: line3: modified_line3, line5: modified_line_5
# ============================================================================================================================
def get_task6_dataset(dat_modelling, args):
    t5_dat_task6 = []
    task_no = 6
    task_prefix = 'revise'

    for idx, row in dat_modelling.iterrows():
        row = row.to_dict()

        t5_ip1 =  f'story: ' + row['Story_delim'] + ' state: ' + row['mod_assertion']
        t5_op1 = row['changed_mod_Story_delim']

        t5_ip2 =  f'story: ' + row['mod_Story_delim'] + ' state: ' + row['assertion']
        t5_op2 = row['changed_Story_delim']

        t5_dict = {'task_no': task_no, 'AssignmentId': row['AssignmentId'], 'storyid': row['storyid'], 'prefix': task_prefix, 'ip': t5_ip1, 'op': t5_op1}
        t5_dat_task6.append(t5_dict)
        
        t5_dict = {'task_no': task_no, 'AssignmentId': row['AssignmentId'], 'storyid': row['storyid'], 'prefix': task_prefix, 'ip': t5_ip2, 'op': t5_op2}
        t5_dat_task6.append(t5_dict)

    t5_dat_task6 = pd.DataFrame(t5_dat_task6)
    return t5_dat_task6

# ============================================================================================================================
# T5 format: World change generation
#   i/p format: story1: story1_description story2: story2_description
#   o/p format: state1: state1_description state2: state2_description
# ============================================================================================================================
def get_task7_dataset(dat_modelling, args):
    t5_dat_task7 = []
    task_no = 7
    task_prefix = 'change'

    for idx, row in dat_modelling.iterrows():
        row = row.to_dict()

        t5_ip1 =  f'story1: ' + row['Story'] + ' story2: ' + row['mod_Story']
        t5_op1 = f'state1: ' + row['assertion'] + ' state2: ' + row['mod_assertion']

        t5_ip2 =  f'story1: ' + row['mod_Story'] + ' story2: ' + row['Story']
        t5_op2 = f'state1: ' + row['mod_assertion'] + ' state2: ' + row['assertion']

        t5_dict = {'task_no': task_no, 'AssignmentId': row['AssignmentId'], 'storyid': row['storyid'], 'prefix': task_prefix, 'ip': t5_ip1, 'op': t5_op1}
        t5_dat_task7.append(t5_dict)
        
        t5_dict = {'task_no': task_no, 'AssignmentId': row['AssignmentId'], 'storyid': row['storyid'], 'prefix': task_prefix, 'ip': t5_ip2, 'op': t5_op2}
        t5_dat_task7.append(t5_dict)
                
    t5_dat_task7 = pd.DataFrame(t5_dat_task7)
    return t5_dat_task7

# ============================================================================================================================
# T5 format: Story-State inference
#   i/p format: story: story1_description + state:Query state
#   o/p format: TRUE/FALSE
# ============================================================================================================================

def get_task8_dataset(dat_modelling, args):
    t5_dat_task8 = []
    task_no = 8
    task_prefix = 'infer_state'

    if args.task8_setting == "0":
        story_col = 'Story'
        mod_story_col = 'mod_Story'

    if args.task8_setting == "1":
        story_col = 'Story_marked'
        mod_story_col = 'mod_Story_marked'

    if args.task8_setting == "2":
        story_col = 'Story_subset_marked'
        mod_story_col = 'mod_Story_subset_marked'

    if args.task8_setting == "3":
        story_col = 'Story_without_marked'
        mod_story_col = 'mod_Story_without_marked'

    for idx, row in dat_modelling.iterrows():
        row = row.to_dict()

        # Story + state --> True
        t5_ip1 =  f'story: ' + row[story_col] + ' state: ' + row['assertion']
        t5_op1, bert_op1 = 'true', 1

        # Story + mod_state --> False
        t5_ip2 =  f'story: ' + row[story_col] + ' state: ' + row['mod_assertion']
        t5_op2, bert_op2 = 'false', 0

        # mod_Story + mod_state --> True
        t5_ip3 =  f'story: ' + row[mod_story_col] + ' state: ' + row['mod_assertion']
        t5_op3, bert_op3 = 'true', 1

        # mod_Story + state --> False
        t5_ip4 =  f'story: ' + row[mod_story_col] + ' state: ' + row['assertion']
        t5_op4, bert_op4 = 'false', 0

        # instance (story, state/mod_state) will be in the story / full data partition
        if args.story_type in ['full', 'story']:
            t5_dict = {'task_no': task_no, 'AssignmentId': row['AssignmentId'], 
                        'storyid': row['storyid'], 'prefix': task_prefix, 
                        'ip': t5_ip1, 'op': t5_op1, 'bert_op': bert_op1}
            t5_dat_task8.append(t5_dict)

            t5_dict = {'task_no': task_no, 'AssignmentId': row['AssignmentId'], 
                        'storyid': row['storyid'], 'prefix': task_prefix, 
                        'ip': t5_ip2, 'op': t5_op2, 'bert_op': bert_op2}
            t5_dat_task8.append(t5_dict)

        # instance (story, state/mod_state) will be in the mod_story / full data partition
        if args.story_type in ['full', 'mod_story']:
            t5_dict = {'task_no': task_no, 'AssignmentId': row['AssignmentId'], 
                       'storyid': row['storyid'], 'prefix': task_prefix, 
                       'ip': t5_ip3, 'op': t5_op3, 'bert_op': bert_op3}
            t5_dat_task8.append(t5_dict)

            t5_dict = {'task_no': task_no, 'AssignmentId': row['AssignmentId'], 
                        'storyid': row['storyid'], 'prefix': task_prefix, 
                        'ip': t5_ip4, 'op': t5_op4, 'bert_op': bert_op4}
            t5_dat_task8.append(t5_dict)

    t5_dat_task8 = pd.DataFrame(t5_dat_task8)
    return t5_dat_task8

# ============================================================================================================================
# Dataset class for T5 model (txt --> txt) model
# ============================================================================================================================
class T5_dataset():
    def __init__(self, cleaned_df, args, task_list = None, story_id_list = None) -> None:
        
        # Get modeling dataset
        self.cleaned_df = cleaned_df
        self.dat_modelling = get_model_ready_dataset(self.cleaned_df)
        self.task_df_list = []

        for task_id in [2, 3, 6, 7, 8]:
            task_df_getter = eval(f'get_task{task_id}_dataset')
            task_df = task_df_getter(self.dat_modelling, args)
            self.task_df_list.append( task_df )

        self.t5_dataset = pd.concat( self.task_df_list , axis=0).reset_index(drop=True)

        if task_list != None:
            self.t5_dataset = self.t5_dataset.loc[self.t5_dataset['task_no'].isin(task_list)]

        if story_id_list != None:
            self.t5_dataset = self.t5_dataset.loc[self.t5_dataset['storyid'].isin(story_id_list)]
        
        self.t5_dataset['ip'] = self.t5_dataset.apply(lambda x: x['prefix'] + ' ' + x['ip'], axis = 1)

        self.tokenizer = T5TokenizerFast.from_pretrained(args.model_type)
        self.ip_encodings = self.tokenizer.batch_encode_plus(self.t5_dataset['ip'].tolist(), padding=True)
        self.op_encodings = self.tokenizer.batch_encode_plus(self.t5_dataset['op'].tolist(), padding=True)

    def __len__(self):
        return len(self.t5_dataset)

    def __getitem__(self, idx):
        return {'idx': idx, 
                'task_id': self.t5_dataset.iloc[idx]['task_no'],
                'ip_input_ids': torch.tensor(self.ip_encodings['input_ids'][idx]).to(torch.long), 
                'ip_attn_mask': torch.tensor(self.ip_encodings['attention_mask'][idx]).to(torch.long),
                'op_input_ids': torch.tensor(self.op_encodings['input_ids'][idx]).to(torch.long), 
                'op_attn_mask': torch.tensor(self.op_encodings['attention_mask'][idx]).to(torch.long)}

# ============================================================================================================================
# Dataset class for BERT/RoBERTa type model (txt --> label) model
# ============================================================================================================================
class BERT_RoBERTa_dataset():
    def __init__(self, cleaned_df, args, task_list = None, story_id_list = None) -> None:
        # Get modeling dataset
        self.cleaned_df = cleaned_df
        self.dat_modelling = get_model_ready_dataset(self.cleaned_df)
        self.task_df_list = []

        for task_id in task_list:
            task_df_getter = eval(f'get_task{task_id}_dataset')
            task_df = task_df_getter(self.dat_modelling, args)
            self.task_df_list.append( task_df )

        self.t5_dataset = pd.concat( self.task_df_list , axis=0).reset_index(drop=True)

        if story_id_list != None:
            self.t5_dataset = self.t5_dataset.loc[self.t5_dataset['storyid'].isin(story_id_list)]
        
        if "roberta" in args.model_type:
            self.tokenizer = RobertaTokenizerFast.from_pretrained(args.model_type)
        else:
            self.tokenizer = BertTokenizerFast.from_pretrained(args.model_type)

        self.ip_encodings = self.tokenizer.batch_encode_plus(self.t5_dataset['ip'].tolist(), padding=True)

        # Truncating the sequence to the max length of 512
        self.ip_encodings['input_ids'] = list(map(lambda x: x[:110], self.ip_encodings['input_ids']))
        self.ip_encodings['attention_mask'] = list(map(lambda x: x[:110], self.ip_encodings['attention_mask']))

    def __len__(self):
        return len(self.t5_dataset)

    def __getitem__(self, idx):
        return {'idx': idx, 
                'task_id': self.t5_dataset.iloc[idx]['task_no'],
                'ip_input_ids': torch.tensor(self.ip_encodings['input_ids'][idx]).to(torch.long), 
                'ip_attn_mask': torch.tensor(self.ip_encodings['attention_mask'][idx]).to(torch.long),
                'op': torch.tensor(self.t5_dataset['bert_op'][idx]).to(torch.long)}