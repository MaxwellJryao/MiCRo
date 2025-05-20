import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from safetensors import safe_open
from datasets import load_dataset, Dataset
from tqdm import tqdm
import pandas as pd
import os
import argparse
from typing import List

import eval_utils

parser = argparse.ArgumentParser()
parser.add_argument('--pretrained_model_name_or_path', type=str, default="Skywork/Skywork-Reward-Llama-3.1-8B")
args = parser.parse_args()

base_model_name_or_path = args.pretrained_model_name_or_path

comps, rm_types = eval_utils.process_hh_rlhf()
res = {}

tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
model = AutoModelForSequenceClassification.from_pretrained(base_model_name_or_path, num_labels=1, torch_dtype=torch.bfloat16).cuda()

with torch.no_grad():
    for rm_type in rm_types:
        print(f'RM Type: {rm_type}')
        ds = Dataset.from_list(comps[rm_type])
        res[rm_type] = []

        # rm_id = 0
        # single_rm = torch.nn.Linear(4096, 1, dtype=torch.bfloat16, bias=False)

        correct = 0
        for item in tqdm(ds):
            chosen_input = item['chosen']
            rejected_input = item['rejected']
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

        print(f'Accuracy: {correct}/{len(ds)}, {correct/len(ds)}')
        res[rm_type].append(correct/len(ds))

# df = pd.DataFrame(res)
# print(df)

# print('done')