import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from safetensors import safe_open
from datasets import load_dataset, Dataset
from tqdm import tqdm
import pandas as pd
import os
import argparse
from typing import List
import json
from criteria import criteria_dict

import eval_utils

parser = argparse.ArgumentParser()
parser.add_argument('--pretrained_model_name_or_path', type=str, default="output_models/GRM-Llama3.2-3B-rewardmodel-ft_origin_700k_heads1_lr2e-3_activate-none_loss")
parser.add_argument('--base_model_name_or_path', type=str, default='Ray2333/GRM-Llama3.2-3B-rewardmodel-ft')
parser.add_argument('--num_heads', type=int, default=1)
parser.add_argument('--data', type=str, default='pku_alignment_safe_pairwise')
parser.add_argument('--add_criteria', type=int, default=0)
args = parser.parse_args()

print('Arguments:')
for arg in vars(args):
    print(format(arg, '<30'), format(str(getattr(args, arg)), '<'))

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
if 'ultrafeedback' in args.data:
    comps, rm_types = eval_utils.process_ultrafeedback()
elif 'hh-rlhf' in args.data:
    comps, rm_types = eval_utils.process_hh_rlhf()
elif 'helpsteer2' in args.data:
    comps, rm_types = eval_utils.process_helpsteer2()
elif 'reward-bench' in args.data:
    comps, rm_types = eval_utils.process_reward_bench()
# rm_types = ['helpful-base']
res = {}

base_model_name_or_path = args.base_model_name_or_path
tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
model = AutoModelForSequenceClassification.from_pretrained(base_model_name_or_path, torch_dtype=torch.bfloat16).cuda()

tensors = {}
with open(f'{pretrained_model_name_or_path}/model.safetensors.index.json', 'r') as f:
    tensors_index = json.load(f)

for name in tensors_index['weight_map']:
    if 'score' in name:
        score_file = tensors_index['weight_map'][name]
        break

with safe_open(f"{pretrained_model_name_or_path}/{score_file}", framework='pt', device=0) as f:
    for k in f.keys():
        if 'score' in k:
            tensors[k.lstrip('score.')] = f.get_tensor(k).cpu()

with torch.no_grad():
    for rm_type in rm_types:
        print(f'RM Type: {rm_type}')
        ds = Dataset.from_list(comps[rm_type])
        res[rm_type] = []

        # rm_id = 0
        for rm_id in range(num_heads):
            # single_rm = torch.nn.Linear(4096, 1, dtype=torch.bfloat16, bias=False)
            model.score.load_state_dict({'weight': tensors['weight'][rm_id].unsqueeze(dim=0)})

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
                if isinstance(chosen_input, List):
                    chosen_input_chat = tokenizer.apply_chat_template(chosen_input, return_tensors='pt').cuda()
                    rejected_input_chat = tokenizer.apply_chat_template(rejected_input, return_tensors='pt').cuda()
                else:
                    chosen_input_chat = tokenizer(chosen_input, return_tensors='pt').input_ids.cuda()
                    rejected_input_chat = tokenizer(rejected_input, return_tensors='pt').input_ids.cuda()
                chosen_output = model(chosen_input_chat).logits
                rejected_output = model(rejected_input_chat).logits
                correct += int(chosen_output > rejected_output)
                # print(chosen_output, rejected_output)

            print(f'Head {rm_id}, Accuracy: {correct}/{len(ds)}, {correct/len(ds)}')
            res[rm_type].append(correct/len(ds))

df = pd.DataFrame(res)
print('='*30)
print(df)
print('\n\n')

print('done')