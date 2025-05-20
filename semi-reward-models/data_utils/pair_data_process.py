import os
import numpy as np
import torch
from tqdm.auto import tqdm
from argparse import ArgumentParser
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, HfArgumentParser
from safetensors.torch import save_file
import pickle
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from reward_model.eval.criteria import REWARDBENCH_CONTEXT_MAP
from datasets import load_from_disk
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field
import torch
import copy
import random


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

token_patterns = {
    # Llama3 token IDs of "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    "llama3": [128009, 128006, 78191, 128007, 271],
    # Gemma2 token IDs of "<end_of_turn>\n<start_of_turn>model\n"
    "gemma2": [107, 108, 106, 2516, 108],
}


@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """
    per_device_train_batch_size: Optional[int] = field(default=8) 
    per_device_eval_batch_size: Optional[int] = field(default=8)
    gradient_accumulation_steps: Optional[int] = field(default=8)
    learning_rate: Optional[float] = field(default=2e-3)
    num_train_epochs: Optional[int] = field(
        default=1,
        metadata={"help": "The number of training epochs for the reward model."},
    )
    optim: Optional[str] = field(
        default="adamw_hf",
        metadata={"help": "The optimizer to use."},
    )
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "The lr scheduler"},)
    max_length: Optional[int] = field(default=4096) 
    use_lora: Optional[bool] = field(default=False)
    # base_model: Optional[str] =  field(default="mistralai/Mistral-7B-Instruct-v0.2") 
    # base_model: Optional[str] = field(default='Skywork/Skywork-Reward-Llama-3.1-8B')
    # base_model: Optional[str] = field(default='Ray2333/GRM-Llama3.2-3B-rewardmodel-ft')
    base_model: Optional[str] = field(default='Ray2333/Gemma-2B-rewardmodel-baseline')
    # base_model: Optional[str] =  field(default="google/gemma-2b-it")
    # base_model: Optional[str] = field(default="meta-llama/Llama-2-7b-chat-hf")
    wandb_name: Optional[str] = field(default="origin",)
    log_dir: Optional[str] = field(default='./output_models')
    loss_type: Optional[str] = field(default='origin')
    use_smallset: Optional[bool] = field(default=False)
    freeze_pretrained: Optional[bool] = field(default=True)
    # data_path: Optional[str] = field(default='llm-blender/Unified-Feedback')
    data_path: Optional[str] = field(default='hh-rlhf')
    num_heads: Optional[int] = field(default=1)
    orthogonal_loss_weight: Optional[float] = field(default=0)
    norm_loss_weight: Optional[float] = field(default=0)
    corr_loss_weight: Optional[float] = field(default=0.5)
    load_balance_loss_weight: Optional[float] = field(default=0.5)
    use_router: Optional[bool] = field(default=False)
    category: Optional[str] = field(default=None)
    dataset_split: Optional[str] = field(default='train')
    device: Optional[str] = field(default='cuda:0')
    load_previous_heads: Optional[str] = field(default='/scratch/jiarui14/multi-rm/Mixture-RM/output_models/Skywork-Reward-Llama-3.1-8B_origin_hh-rlhf_helpful_heads1_lr2e-3', metadata={"help": "load previous heads, paths separated by comma"})

def process_batch(features):
    merged_chosen_features, merged_rejected_features = [],[]
    margins = []
    merged_chosen_features = {
        "input_ids": features["input_ids"],
        "attention_mask": features["attention_mask"],
        "labels": features["labels"],
    }
    chosen_batch = rm_tokenizer.pad(
        merged_chosen_features,
        padding=True,
        max_length=4096,
        pad_to_multiple_of=None,
        return_tensors='pt',
    )

    return chosen_batch

def find_token_for_gating(lst, model_family):
    """Find the last occurrence of a token_pattern in a list."""
    if not isinstance(lst,List):
        lst = lst.cpu().numpy().tolist()
    token_pattern = token_patterns[model_family]
    token_pattern_len = len(token_pattern)
    search_end = len(lst)
    for j in range(search_end - token_pattern_len, -1, -1):
        if lst[j : j + token_pattern_len] == token_pattern:
            return j
    raise ValueError("Token pattern not found in the list.")

def build_dataset(ds,rm_tokenizer,script_args):

    tokenizer = copy.deepcopy(rm_tokenizer)

    prompt_template = "[CONTEXT] {context} [RESPONSE A] {response_A} [RESPONSE B] {response_B} \n"

    x1 = "\n{% for message in messages %}{% if loop.index0 % 2 == 0 %}\n\n<turn> user"
    x2 = "\n {{ message['content'] }}{% else %}\n\n<turn> assistant\n"
    x3 = " {{ message['content'] }}{% endif %}{% endfor %}\n\n\n"
    my_template = x1 + x2 + x3

    rm_tokenizer.chat_template = my_template
    
    token_id_A = rm_tokenizer.encode("A", add_special_tokens=False)
    token_id_B = rm_tokenizer.encode("B", add_special_tokens=False)
    assert len(token_id_A) == 1 and len(token_id_B) == 1
    token_id_A = token_id_A[0]
    token_id_B = token_id_B[0]

    def formatting_func(example):
        chosen_messages = example['chosen']
        rejected_messages = example['rejected']
        if isinstance(chosen_messages, List):
            context = rm_tokenizer.apply_chat_template(chosen_messages[:-1], tokenize=False)
            responses = [chosen_messages[-1]["content"], rejected_messages[-1]["content"]]
        else:
            prompt = example["prompt"]
            chosen_messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": example["chosen"]}
                ]
            rejected_messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": example["rejected"]}
                ]
            context = rm_tokenizer.apply_chat_template(chosen_messages[:-1], tokenize=False)
            responses = [chosen_messages[-1]["content"], rejected_messages[-1]["content"]]

        chosen_position = random.choice([0,1])
        response_A = responses[chosen_position]
        response_B = responses[1 - chosen_position]
        prompt = prompt_template.format(context=context, response_A=response_A, response_B=response_B)
        message = [
            {"role": "user", "content": prompt},
        ]
        
        chosen_tokens = tokenizer.encode_plus(
            tokenizer.apply_chat_template(message, tokenize=False).replace(tokenizer.bos_token, ""),
            return_tensors="pt",
            add_special_tokens=False)
        
        if chosen_position == 0:
            label = 1
        else:
            label = 0

        return {
            "input_ids": chosen_tokens["input_ids"][0], "attention_mask": chosen_tokens["attention_mask"][0], "labels":label,
        }

    ds = ds.map(formatting_func, batched=False, num_proc=30) 
    ds = ds.filter(lambda x: len(x["input_ids"]) <= script_args.max_length, num_proc=30)

    remove_columns = []
    for col in ds.column_names:
        if 'input' not in col and 'attention' not in col and 'margin' not in col and 'length' not in col and 'label' not in col:
            remove_columns.append(col)
    ds = ds.remove_columns(remove_columns)

    ds.set_format(type="torch")

    return ds
    
if __name__ == '__main__':

    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    dataset_names = ["../dataset/rpr_per_category_pairwise_add_criterion","../dataset/anthropic_rlhf_hh_pairwise",
                     "../dataset/rpr_per_category_pairwise",
                     "../dataset/ultrafeedback_per_attribute_pairwise","../dataset/helpsteer2_per_attribute_pairwise",
    "../dataset/rpr_per_category_pairwise_add_criterion_template2","../dataset/400K_pairwise"]
    # dataset_names = ["../dataset/rpr_per_category_pairwise_add_criterion_template2","../dataset/400K_pairwise"]
    batch_size = 15
    
    for dataset_name in dataset_names:
        
        d_name = dataset_name.split('/')[2]
        ds = load_from_disk(dataset_name)[script_args.dataset_split]
        
        # Load dataset and shuffle
        model_name = script_args.base_model.split('/')[1].replace('-','_')
        
        # Load model and tokenizer
        device = script_args.device
        rm = AutoModel.from_pretrained(script_args.base_model, torch_dtype=torch.bfloat16,cache_dir="/srv/local/hf").to(device)
        rm_tokenizer = AutoTokenizer.from_pretrained(script_args.base_model)
        if 'Llama' in model_name:
            rm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        else:
            rm_tokenizer.pad_token = rm_tokenizer.eos_token

        # formatting dataset
        ds = build_dataset(ds,rm_tokenizer,script_args)
        
        # Prepare save path for embeddings
        save_path = '../dataset/pair_datasets/' 
        os.makedirs(save_path, exist_ok=True)

        chosen_embeddings,labels = [], []

        for i in tqdm(range(0, len(ds), batch_size), desc="Processing dataset in batches"):
            batch = ds[i:i + batch_size]
            chosen_batch = process_batch(batch)
            labels.extend(batch['labels'].numpy().tolist())
            
            # Pass through the model manually for chosen and rejected separately
            with torch.no_grad():
                output = rm(
                    input_ids=chosen_batch["input_ids"].to(device),
                    attention_mask=chosen_batch["attention_mask"].to(device)
                )

            # Extract the last hidden state for chosen and rejected
            chosen_last_hidden_state = output.last_hidden_state

            # Compute embeddings
            attention_mask_chosen = chosen_batch["attention_mask"]
            last_non_padding_idx = (attention_mask_chosen == 1).cumsum(dim=1).argmax(dim=1)
            for i, idx in enumerate(last_non_padding_idx):
                chosen_embeddings.append(chosen_last_hidden_state[i,idx,:].cpu())

        # Stack embeddings and labels for saving
        chosen_embeddings = torch.stack(chosen_embeddings)
        labels=torch.tensor(labels)

        file_name = '_'.join(["SemiMultiRM", "embeddings", model_name, d_name, script_args.dataset_split]) + ".safetensors"
        # Save as safetensors
        save_file(
            {"embeddings": chosen_embeddings, 
            "labels":labels},
            os.path.join(save_path,file_name)
        )

        print(f"Saved embeddings to {save_path}")
