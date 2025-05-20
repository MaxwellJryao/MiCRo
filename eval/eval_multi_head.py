import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from safetensors import safe_open
from datasets import load_dataset, Dataset
from tqdm import tqdm
import pandas as pd
import os
from accelerate import Accelerator
import argparse
import json
import numpy as np
from criteria import criteria_dict

import eval_utils

parser = argparse.ArgumentParser()
parser.add_argument('--pretrained_model_name_or_path', type=str, default="output_models/GRM-Llama3.2-3B-rewardmodel-ft_mixture_BT_700k_heads10_lr2e-3_activate-none_loss_loadBalance0.5")
parser.add_argument('--num_heads', type=int, default=10)
parser.add_argument('--use_router', type=bool, default=True)
parser.add_argument('--data', type=str, default='pku_alignment_safe_pairwise')
parser.add_argument('--mixture_type', type=str, default='BT')
parser.add_argument('--add_criteria', type=int, default=0)
parser.add_argument('--router_type', type=str, default='softmax')
args = parser.parse_args()

print('Arguments:')
for arg in vars(args):
    print(format(arg, '<30'), format(str(getattr(args, arg)), '<'))

# check path for saving results
def check_path():
    assert os.path.exists('../results/')
        
# check_path()

pretrained_model_name_or_path = args.pretrained_model_name_or_path
num_heads = args.num_heads

if 'helpsteer2_per_attribute_pairwise' in args.data:
    comps, rm_types = eval_utils.process_helpsteer2_per_attr()
elif 'ultrafeedback_per_attribute_pairwise_30k' in args.data:
    comps, rm_types = eval_utils.process_ultrafeedback_per_attribute_pairwise_30k()
elif 'pku_alignment_safe_pairwise' in args.data:
    comps, rm_types = eval_utils.process_pku_alignment_safe_pairwise()
elif 'anthropic_rlhf_hh_pairwise' in args.data:
    comps, rm_types = eval_utils.process_anthropic_rlhf_hh_pairwise()
if 'hh-rlhf' in args.data:
    comps, rm_types = eval_utils.process_hh_rlhf()
elif 'ultrafeedback' in args.data:
    comps, rm_types = eval_utils.process_ultrafeedback()
elif 'helpsteer2' in args.data:
    comps, rm_types = eval_utils.process_helpsteer2()
elif 'reward-bench' in args.data:
    comps, rm_types = eval_utils.process_reward_bench()
res = {}

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path, num_labels=num_heads, torch_dtype=torch.bfloat16).cuda()

tensors = {}
with open(f"{pretrained_model_name_or_path}/model.safetensors.index.json", 'r') as f:
    index = json.load(f)
    for k in index['weight_map']:
        if 'router' in k:
            file_name = index['weight_map'][k]
            break
with safe_open(f"{pretrained_model_name_or_path}/{file_name}", framework='pt', device=0) as f:
    for k in f.keys():
        if 'router' in k:
            if args.use_router:
                tensors[k.lstrip('router.')] = f.get_tensor(k)
            else:
                tensors[k] = f.get_tensor(k)

if args.use_router:
    weight_layer = nn.Linear(model.config.hidden_size, num_heads, dtype=torch.bfloat16)
else:
    weight_layer = torch.zeros(num_heads, dtype=torch.bfloat16)
device = Accelerator().local_process_index
weight_layer.to(device)
model.router = weight_layer

if args.use_router:
    model.router.load_state_dict({'weight': tensors['weight'], 'bias': tensors['bias']})
else:
    model.router = tensors['router']
model.eval()

router_dist = {}

with torch.no_grad():
    for rm_type in rm_types:
        print(f'RM Type: {rm_type}')
        router_dist[rm_type] = []
        ds = Dataset.from_list(comps[rm_type])
        res[rm_type] = []
        # rm_id = 0

        correct = 0
        for item in tqdm(ds):
            chosen_input = item['chosen']
            rejected_input = item['rejected']
            if args.add_criteria != 0:
                if 'chat' in rm_type:
                    criteria = criteria_dict['chat']
                elif 'safe' in rm_type or 'harm' in rm_type:
                    criteria = criteria_dict['safety']
                elif 'reason' in rm_type:
                    criteria = criteria_dict['reasoning']
                elif 'help' in rm_type:
                    criteria = criteria_dict['helpful']
                elif 'correct' in rm_type or 'truth' in rm_type:
                    criteria = criteria_dict['correct']
                elif 'verbose' in rm_type:
                    criteria = criteria_dict['verbose']
                elif 'complex' in rm_type:
                    criteria = criteria_dict['complex']
                else:
                    criteria = criteria_dict['other'].format(criteria=rm_type)
                
                if isinstance(chosen_input, list):
                    chosen_input[0]['content'] = f'[criteria] {criteria}\n[context] {chosen_input[0]["content"]}'
                    rejected_input[0]['content'] = f'[criteria] {criteria}\n[context] {rejected_input[0]["content"]}'
                else:
                    chosen_input = f'[criteria] {criteria}\n[context] {chosen_input}'
                    rejected_input = f'[criteria] {criteria}\n[context] {rejected_input}'
            if isinstance(chosen_input, list):
                chosen_prompt_len = len(tokenizer.encode_plus(tokenizer.apply_chat_template(chosen_input[:1], add_generation_prompt=True, tokenize=False)))
                rejected_prompt_len = len(tokenizer.encode_plus(tokenizer.apply_chat_template(rejected_input[:1], add_generation_prompt=True, tokenize=False)))
                chosen_input_chat = tokenizer.apply_chat_template(chosen_input, return_tensors='pt').cuda()
                rejected_input_chat = tokenizer.apply_chat_template(rejected_input, return_tensors='pt').cuda()
            else:
                chosen_prompt_len = len(tokenizer(chosen_input[:-11-len(chosen_input.split('Assistant: ')[-1])]))
                rejected_prompt_len = len(tokenizer(rejected_input[:-11-len(rejected_input.split('Assistant: ')[-1])]))
                chosen_input_chat = tokenizer(chosen_input, return_tensors='pt').input_ids.cuda()
                rejected_input_chat = tokenizer(rejected_input, return_tensors='pt').input_ids.cuda()
            
            chosen_output = model(chosen_input_chat, output_hidden_states=True)
            rewards, last_hidden_states = chosen_output.logits, chosen_output.hidden_states[-1][:, chosen_prompt_len, :]
            if args.use_router:
                if args.router_type == 'softmax':
                    router_weight = model.router(last_hidden_states).softmax(dim=-1)
                else:
                    router_weight = model.router(last_hidden_states).argmax(dim=-1)
                    router_weight = F.one_hot(router_weight, num_classes=num_heads).float()
                if args.mixture_type == 'BT':
                    chosen_reward = (nn.functional.logsigmoid(rewards) * router_weight).sum()
                else:
                    chosen_reward = (rewards * router_weight).sum()
                router_dist[rm_type].append(router_weight.cpu().squeeze())
            else:
                if args.router_type == 'softmax':
                    router_weight = model.router.softmax(dim=-1)
                else:
                    router_weight = model.router.argmax(dim=-1)
                    router_weight = F.one_hot(router_weight, num_classes=num_heads).float()
                if args.mixture_type == 'BT':
                    chosen_reward = (nn.functional.logsigmoid(rewards) * router_weight).sum()
                else:
                    chosen_reward = (rewards * router_weight).sum()
                router_dist[rm_type].append(router_weight.cpu().squeeze())

            rejected_output = model(rejected_input_chat, output_hidden_states=True)
            rewards, last_hidden_states = rejected_output.logits, rejected_output.hidden_states[-1][:, rejected_prompt_len, :]
            
            if args.use_router:
                if args.mixture_type == 'BT':
                    rejected_reward = (nn.functional.logsigmoid(rewards) * router_weight).sum()
                else:
                    rejected_reward = (rewards * router_weight).sum()
            else:
                if args.mixture_type == 'BT':
                    rejected_reward = (nn.functional.logsigmoid(rewards) * router_weight).sum()
                else:
                    rejected_reward = (rewards * router_weight).sum()
            correct += int(chosen_reward > rejected_reward)
            # print(chosen_output, rejected_output)

        print(f'Accuracy: {correct}/{len(ds)}, {correct/len(ds)}')
        res[rm_type].append(correct/len(ds))
        router_dist[rm_type] = torch.stack(router_dist[rm_type])

df = pd.DataFrame(res)
print('='*30)
print(df)
print('\n\n')

torch.save(router_dist, f'../results/{pretrained_model_name_or_path.split("/")[-1]}_{args.data}_{args.router_type}_router_dist.pt')
# print('done')