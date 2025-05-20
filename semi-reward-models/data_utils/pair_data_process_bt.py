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
        "input_ids": features["input_ids_chosen"],
        "attention_mask": features["attention_mask_chosen"],
    }
    merged_rejected_features = {
        "input_ids": features["input_ids_rejected"],
        "attention_mask": features["attention_mask_rejected"],
    }
    chosen_batch = rm_tokenizer.pad(
        merged_chosen_features,
        padding=True,
        max_length=4096,
        pad_to_multiple_of=None,
        return_tensors='pt',
    )
    rejected_batch = rm_tokenizer.pad(
        merged_rejected_features,
        padding=True,
        max_length=4096,
        pad_to_multiple_of=None,
        return_tensors='pt',
    )
    return chosen_batch, rejected_batch

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
    def formatting_func(example):
        chosen_messages = example['chosen']
        rejected_messages = example['rejected']
        if isinstance(chosen_messages, List):
            prompt_plus_chosen_response = rm_tokenizer.apply_chat_template(chosen_messages, tokenize=False)
            prompt_plus_rejected_response = rm_tokenizer.apply_chat_template(rejected_messages, tokenize=False)
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
            prompt_plus_chosen_response = rm_tokenizer.apply_chat_template(chosen_messages, tokenize=False)
            prompt_plus_rejected_response = rm_tokenizer.apply_chat_template(rejected_messages, tokenize=False)

        tokens_chosen = rm_tokenizer.encode_plus(prompt_plus_chosen_response, return_tensors="pt")
        tokens_rejected  = rm_tokenizer.encode_plus(prompt_plus_rejected_response, return_tensors="pt")

        prompt_template = rm_tokenizer.apply_chat_template(chosen_messages[:-1], tokenize=False, add_generation_prompt=True)
        tokens_prompt = rm_tokenizer.encode_plus(prompt_template, return_tensors="pt")['input_ids'][0]

        return {
            "input_ids_chosen": tokens_chosen["input_ids"][0], "attention_mask_chosen": tokens_chosen["attention_mask"][0],
            "input_ids_rejected": tokens_rejected["input_ids"][0], "attention_mask_rejected": tokens_rejected["attention_mask"][0],
            'prompt_length': len(tokens_prompt),
        }

    ds = ds.map(formatting_func, batched=False, num_proc=30) 
    ds = ds.filter(lambda x: len(x["input_ids_chosen"]) <= script_args.max_length and len(x["input_ids_rejected"]) <= script_args.max_length, num_proc=30)
    
    remove_columns = []
    for col in ds.column_names:
        if 'input' not in col and 'attention' not in col and 'margin' not in col and 'length' not in col:
            remove_columns.append(col)
    ds = ds.remove_columns(remove_columns)

    ds.set_format(type="torch")

    return ds
    
if __name__ == '__main__':

    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    # dataset_names = ["../dataset/rpr_per_category_pairwise_add_criterion",
    #                 "../dataset/rpr_per_category_pairwise",
    #                 "../dataset/ultrafeedback_per_attribute_pairwise",
    #                  "../dataset/anthropic_rlhf_hh_pairwise",
                    #  "../dataset/rpr_per_category_pairwise_add_criterion_template2",
                    #  "../dataset/helpsteer2_per_attribute_pairwise",
                    #  "../dataset/400K_pairwise",
                    #  ] #del chosen_output, rejected_output
    dataset_names = ["../dataset/reward_bench_pairwise_augmented_context",
                     ]
    device = script_args.device
    batch_size = 15
    rm = AutoModel.from_pretrained(script_args.base_model, torch_dtype=torch.bfloat16,cache_dir="/srv/local/hf").to(device)
    rm_tokenizer = AutoTokenizer.from_pretrained(script_args.base_model)
    
    for dataset_name in dataset_names:
        
        d_name = dataset_name.split('/')[2]
        ds = load_from_disk(dataset_name)[script_args.dataset_split]
        
        # Load dataset and shuffle
        model_name = script_args.base_model.split('/')[1].replace('-','_')
        
        # Load model and tokenizer
        if 'Llama' in model_name:
            rm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        else:
            rm_tokenizer.pad_token = rm_tokenizer.eos_token

        # formatting dataset
        ds = build_dataset(ds,rm_tokenizer,script_args)
        
        # Prepare save path for embeddings
        save_path = '../dataset' 
        os.makedirs(save_path, exist_ok=True)

        chosen_embeddings, rejected_embeddings, chosen_prompt_embeddings, rejected_prompt_embeddings,source = [], [],[], [],[]

        for i in tqdm(range(0, len(ds), batch_size), desc="Processing dataset in batches"):
            batch = ds[i:i + batch_size]
            chosen_batch, rejected_batch = process_batch(batch)
            
            # Pass through the model manually for chosen and rejected separately
            with torch.no_grad():
                chosen_output = rm(
                    input_ids=chosen_batch["input_ids"].to(device),
                    attention_mask=chosen_batch["attention_mask"].to(device)
                )
                rejected_output = rm(
                    input_ids=rejected_batch["input_ids"].to(device),
                    attention_mask=rejected_batch["attention_mask"].to(device)
                )

            # Compute embeddings
            attention_mask_chosen = chosen_batch["attention_mask"]
            last_non_padding_idx = (attention_mask_chosen == 1).cumsum(dim=1).argmax(dim=1)
            for i, idx in enumerate(last_non_padding_idx):
                chosen_embeddings.append(chosen_output.last_hidden_state[i,idx,:].cpu())
            
            attention_mask_rejected = rejected_batch["attention_mask"]
            last_non_padding_idx = (attention_mask_rejected == 1).cumsum(dim=1).argmax(dim=1)
            for i, idx in enumerate(last_non_padding_idx):
                rejected_embeddings.append(rejected_output.last_hidden_state[i,idx,:].cpu())

            # prompt embeddings
            for i, idx in enumerate(batch['prompt_length']):
                prompt_idx = find_token_for_gating(chosen_batch["input_ids"][i], 'gemma2')
                chosen_prompt_embeddings.append(chosen_output.last_hidden_state[i,prompt_idx,:].cpu())
            for i, idx in enumerate(batch['prompt_length']):
                prompt_idx = find_token_for_gating(rejected_batch["input_ids"][i], 'gemma2')
                rejected_prompt_embeddings.append(rejected_output.last_hidden_state[i,prompt_idx,:].cpu())

            del chosen_output, rejected_output

        # Stack embeddings and labels for saving
        chosen_embeddings = torch.stack(chosen_embeddings)
        rejected_embeddings = torch.stack(rejected_embeddings)
        chosen_prompt_embeddings = torch.stack(chosen_prompt_embeddings)
        rejected_prompt_embeddings = torch.stack(rejected_prompt_embeddings)

        file_name = '_'.join(["SemiMultiRM", "embeddings", model_name, d_name, script_args.dataset_split]) + ".safetensors"
        # Save as safetensors
        save_file(
            {"chosen_embeddings": chosen_embeddings, 
            "chosen_prompt_embeddings":chosen_prompt_embeddings,
            "rejected_embeddings": rejected_embeddings,
            "rejected_prompt_embeddings":rejected_prompt_embeddings},
            os.path.join(save_path,file_name)
        )

        print(f"Saved embeddings to {save_path}")


# source.append(example['attribute'])

# add context
# context = REWARDBENCH_CONTEXT_MAP[example['subset']]
# if isinstance(example["chosen"], str):
#     prompt = example["prompt"]
#     prompt = f'[context] {example["prompt"]} {context}'
#     resp = example["chosen"]
#     message = [
#         {"role": "user", "content": prompt},
#         {"role": "assistant", "content": resp}
#         ]
#     conv_formatted = rm_tokenizer.apply_chat_template(message, tokenize=False)
# else:
#     conv_formatted = rm_tokenizer.apply_chat_template(example["chosen"], tokenize=False)
