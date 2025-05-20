from glob import glob
from safetensors.torch import load_file
from tqdm import tqdm
import torch
import pickle
import pandas as pd
from eval.criteria import ATTRIBUTES_LIST
import numpy as np
from datasets import load_from_disk
import random
from typing import List
from transformers import AutoTokenizer, AutoModel, HfArgumentParser



def build_dataset(ds,rm_tokenizer,max_length=4096):
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
    ds = ds.filter(lambda x: len(x["input_ids_chosen"]) <= max_length and len(x["input_ids_rejected"]) <= max_length, num_proc=30)

    return ds

def load_embeddings(
    embeddings_path,
    pattern=None,
    keys=[
        "embeddings",
        "prompt_embeddings",
        "labels"]):
    files = sorted(glob(f"{embeddings_path}-{pattern}.safetensors")
                   if pattern else [embeddings_path])
    data = {key: [] for key in keys}

    for file in tqdm(files, desc="Loading embeddings"):
        file_data = load_file(file)
        for key in keys:
            data[key].append(file_data[key])

    return {key: torch.cat(data[key], dim=0) for key in keys}


def load_data(path, keys, source_path=None, filter_condition=None):
    print(f"Loading data from {path}...")
    data = load_embeddings(path, keys=keys)

    if source_path:
        with open(source_path, 'rb') as file:
            source_data = pickle.load(file)
        if filter_condition:
            filtered_indices = [i for i, source in enumerate(
                source_data) if filter_condition(source)]

            # Filter data based on the condition
            for key in keys:
                data[key] = data[key][filtered_indices]
            print(f"Filtered data: {len(filtered_indices)} items")
            return data, filtered_indices
        else:
            return data, source_data

    return data


def load_labeled_data(labeled_path):
    print("Loading labeled embeddings and labels...")
    labeled_data = load_embeddings(
        labeled_path,
        keys=[
            "embeddings",
            "prompt_embeddings",
            "labels"])
    labeled_texts = labeled_data["embeddings"]
    labeled_prompts = labeled_data["prompt_embeddings"]
    labeled_labels = labeled_data["labels"]
    print(
        f"Total labeled embeddings: {labeled_texts.shape[0]}, labels: {labeled_labels.shape[0]}")
    return labeled_texts, labeled_prompts, labeled_labels


def load_skywork_data(unlabeled_path, unlabeled_source_path):
    print("Loading unlabeled embeddings...")
    unlabeled_data, selected_indices = load_data(
        unlabeled_path,
        keys=["chosen_embeddings", "chosen_prompt_embeddings", "rejected_prompt_embeddings", "rejected_embeddings"],
        source_path=unlabeled_source_path,
        filter_condition=lambda source: source != 'helpsteer2'
    )
    unlabeled_chosen_embeddings = unlabeled_data["chosen_embeddings"]
    unlabeled_rejected_embeddings = unlabeled_data["rejected_embeddings"]
    unlabeled_chosen_prompt_embeddings = unlabeled_data["chosen_prompt_embeddings"]
    unlabeled_rejected_prompt_embeddings = unlabeled_data["rejected_prompt_embeddings"]

    unlabeled_texts = torch.cat(
        [unlabeled_chosen_embeddings, unlabeled_rejected_embeddings], dim=0)
    unlabeled_prompts = torch.cat(
        [unlabeled_chosen_prompt_embeddings, unlabeled_rejected_prompt_embeddings], dim=0)
    print(f"Total unlabeled embeddings: {unlabeled_texts.shape[0]}")
    return unlabeled_texts, unlabeled_prompts


def load_validation_data(validation_path):
    print("Loading validation embeddings and labels...")
    val_data = load_embeddings(
        validation_path,
        keys=[
            "embeddings",
            "prompt_embeddings",
            "labels"])
    val_texts = val_data["embeddings"]
    val_prompts = val_data["prompt_embeddings"]
    val_labels = val_data["labels"]
    print(
        f"Total validation embeddings: {val_texts.shape[0]}, labels: {val_labels.shape[0]}")
    return val_texts, val_prompts, val_labels

# Load reward bench data


def load_reward_bench_data(reward_bench_path, reward_bench_source_path):
    print("Loading reward bench embeddings and labels...")
    reward_bench_data = load_data(
        reward_bench_path,
        keys=[
            "chosen_embeddings",
            "chosen_prompt_embeddings",
            "rejected_prompt_embeddings",
            "rejected_embeddings"])
    with open(reward_bench_source_path, 'rb') as file:
        subset = pickle.load(file)

    reward_bench_chosen_embeddings = reward_bench_data["chosen_embeddings"]
    reward_bench_rejected_embeddings = reward_bench_data["rejected_embeddings"]
    reward_bench_chosen_prompt_embeddings = reward_bench_data["chosen_prompt_embeddings"]
    reward_bench_rejected_prompt_embeddings = reward_bench_data["rejected_prompt_embeddings"]

    return {
        "chosen_embeddings": reward_bench_chosen_embeddings,
        "rejected_embeddings": reward_bench_rejected_embeddings,
        "chosen_prompt_embeddings": reward_bench_chosen_prompt_embeddings,
        "rejected_prompt_embeddings": reward_bench_rejected_prompt_embeddings,
        "subset": subset
    }


def load_rlhf_hh_data(
        train_path,
        test_path,
        config):
    
    dataset_name = "./dataset/anthropic_rlhf_hh_pairwise"
    attributes = ['rlhf-hh-helpfulness', 'rlhf-hh-harmlessness']

    train_data = load_data(
        train_path,
        keys=[
            "chosen_embeddings",
            "chosen_prompt_embeddings",
            "rejected_prompt_embeddings",
            "rejected_embeddings"],
    )
    
    n_samples_per_attribute = config['n_labeled_samples']
    ds = load_from_disk(dataset_name)['train']
    rm_tokenizer = AutoTokenizer.from_pretrained(config['model_path'])
    ds = build_dataset(ds,rm_tokenizer)
    pairwise_df_per_attribute = pd.DataFrame(ds)
    

    pairwise_per_attribute_data = load_data(
        train_path,
        keys=[
            "chosen_embeddings",
            "chosen_prompt_embeddings",
            "rejected_prompt_embeddings",
            "rejected_embeddings"],
    )
    assert len(pairwise_df_per_attribute) == pairwise_per_attribute_data['chosen_embeddings'].shape[0]


    eval_ratio = 0.15
    train_data = []
    eval_data = []

    # Group by attribute and split into training and evaluation
    for attribute, group in pairwise_df_per_attribute.groupby('attribute'):
        n_samples_per_attribute_i = min(n_samples_per_attribute,len(group))
        n_eval = max(1, int(eval_ratio * n_samples_per_attribute_i))
        n_train = n_samples_per_attribute_i - n_eval

        sampled_group = group.sample(
            n=n_samples_per_attribute_i,
            random_state=42,
            replace=False)
        train_data.append(sampled_group.iloc[:n_train])
        eval_data.append(sampled_group.iloc[n_train:n_train + n_eval])

    train_pairs = pd.concat(train_data, ignore_index=True)
    eval_pairs = pd.concat(eval_data, ignore_index=True)

    train_chosen_embeddings = torch.stack(
        [pairwise_per_attribute_data["chosen_embeddings"][i] for i in train_pairs.index]
    )
    train_rejected_embeddings = torch.stack(
        [pairwise_per_attribute_data["rejected_embeddings"][i] for i in train_pairs.index]
    )
    train_chosen_prompt_embeddings = torch.stack(
        [pairwise_per_attribute_data["chosen_prompt_embeddings"][i] for i in train_pairs.index]
    )
    train_rejected_prompt_embeddings = torch.stack(
        [pairwise_per_attribute_data["rejected_prompt_embeddings"][i] for i in train_pairs.index]
    )

    eval_chosen_embeddings = torch.stack(
        [pairwise_per_attribute_data["chosen_embeddings"][i] for i in eval_pairs.index]
    )
    eval_rejected_embeddings = torch.stack(
        [pairwise_per_attribute_data["rejected_embeddings"][i] for i in eval_pairs.index]
    )
    eval_chosen_prompt_embeddings = torch.stack(
        [pairwise_per_attribute_data["chosen_prompt_embeddings"][i] for i in eval_pairs.index]
    )
    eval_rejected_prompt_embeddings = torch.stack(
        [pairwise_per_attribute_data["rejected_prompt_embeddings"][i] for i in eval_pairs.index]
    )


    train_head_masks = []
    for attribute in train_pairs['attribute']:
        mask = torch.tensor(
            [1.0 if attr == attribute else 0.0 for attr in attributes])
        train_head_masks.append(mask)
    train_head_masks = torch.stack(train_head_masks, dim=0)
    print('Count for each attribute (RLHF-HH, training)',
          torch.sum(train_head_masks, dim=0))

    pairwise_train_data_dict = {
        "chosen_embeddings": train_chosen_embeddings,
        "rejected_embeddings": train_rejected_embeddings,
        "chosen_prompt_embeddings": train_chosen_prompt_embeddings,
        "rejected_prompt_embeddings": train_rejected_prompt_embeddings,
        "subset": train_pairs['attribute'],
        "head_mask": train_head_masks.unsqueeze(-1),
        "labels": train_head_masks,
    }

 
    eval_head_masks = []
    for attribute in eval_pairs['attribute'].tolist():
        mask = torch.tensor(
            [1.0 if attr == attribute else 0.0 for attr in attributes])
        eval_head_masks.append(mask)
    eval_head_masks = torch.stack(eval_head_masks, dim=0)

    pairwise_eval_data_dict = {
        "chosen_embeddings": eval_chosen_embeddings,
        "rejected_embeddings": eval_rejected_embeddings,
        "chosen_prompt_embeddings": eval_chosen_prompt_embeddings,
        "rejected_prompt_embeddings": eval_rejected_prompt_embeddings,
        "subset": eval_pairs['attribute'].tolist(),
        "head_mask": eval_head_masks
    }

  
    ds = load_from_disk(dataset_name)['test']
    rm_tokenizer = AutoTokenizer.from_pretrained(config['model_path'])
    ds = build_dataset(ds,rm_tokenizer)
    test_pairwise_df = pd.DataFrame(ds)
    test_pairwise_data = load_data(
        test_path,
        keys=[
            "chosen_embeddings",
            "chosen_prompt_embeddings",
            "rejected_prompt_embeddings",
            "rejected_embeddings"],
    )
    assert len(test_pairwise_df) == test_pairwise_data['chosen_embeddings'].shape[0]

    test_chosen_embeddings = test_pairwise_data['chosen_embeddings']
    test_rejected_embeddings = test_pairwise_data['rejected_embeddings']
    test_chosen_prompt_embeddings = test_pairwise_data['chosen_prompt_embeddings']
    test_rejected_prompt_embeddings = test_pairwise_data['rejected_prompt_embeddings']

    test_head_masks = []
    for attribute in test_pairwise_df['attribute']:
        mask = torch.tensor(
            [1.0 if attr == attribute else 0 for attr in attributes])
        test_head_masks.append(mask)
    test_head_masks = torch.stack(test_head_masks, dim=0)
    print('Count for each attribute (RLHF-HH, test)',
          torch.sum(test_head_masks, dim=0))

    pairwise_test_data_dict = {
        "chosen_embeddings": test_chosen_embeddings,
        "rejected_embeddings": test_rejected_embeddings,
        "chosen_prompt_embeddings": test_chosen_prompt_embeddings,
        "rejected_prompt_embeddings": test_rejected_prompt_embeddings,
        "subset": [attribute for attribute in test_pairwise_df['attribute']],
        "head_mask": test_head_masks}

    return pairwise_train_data_dict, pairwise_eval_data_dict, pairwise_test_data_dict


def load_helpsteer_data(train_path, test_path, config):

    attributes = [
        'helpfulness',
        'correctness',
        'coherence',
        'complexity',
        'verbosity']
    n_samples_per_attribute = config['n_labeled_samples']

    dataset_name = "./dataset/helpsteer2_per_attribute_pairwise"
    ds = load_from_disk(dataset_name)['train']
    rm_tokenizer = AutoTokenizer.from_pretrained(config['model_path'])
    ds = build_dataset(ds,rm_tokenizer)
    pairwise_df_per_attribute = pd.DataFrame(ds)

    pairwise_per_attribute_data = load_data(
        train_path,
        keys=[
            "chosen_embeddings",
            "chosen_prompt_embeddings",
            "rejected_prompt_embeddings",
            "rejected_embeddings"],
    )
    assert len(pairwise_df_per_attribute) == pairwise_per_attribute_data['chosen_embeddings'].shape[0]

    eval_ratio = 0.15
    train_data = []
    eval_data = []

    pairwise_df_per_attribute.reset_index(inplace = True)

    # Group by attribute and split into training and evaluation
    for attribute, group in pairwise_df_per_attribute.groupby('attribute'):
        n_samples_per_attribute_i = min(n_samples_per_attribute,len(group))
        n_eval = max(1, int(eval_ratio * n_samples_per_attribute_i))
        n_train = n_samples_per_attribute_i - n_eval

        sampled_group = group.sample(
            n=n_samples_per_attribute_i,
            random_state=42,
            replace=False)
        train_data.append(sampled_group.iloc[:n_train])
        eval_data.append(sampled_group.iloc[n_train:n_train + n_eval])

    train_pairs = pd.concat(train_data, ignore_index=True)
    eval_pairs = pd.concat(eval_data, ignore_index=True)
        
    train_pairs = (train_pairs.groupby(['prompt', 'chosen', 'rejected'], as_index=False)
    .agg({
        'attribute': lambda x: list(x), 
        'index': lambda x: random.choice(list(x)) 
    }))
    print(len(train_pairs))
    train_pairs = train_pairs[train_pairs['attribute'].apply(len) < len(attributes)]
    print(len(train_pairs))
    
    train_chosen_embeddings = torch.stack(
        [pairwise_per_attribute_data["chosen_embeddings"][i] for i in train_pairs['index']]
    )
    train_rejected_embeddings = torch.stack(
        [pairwise_per_attribute_data["rejected_embeddings"][i] for i in train_pairs['index']]
    )
    train_chosen_prompt_embeddings = torch.stack(
        [pairwise_per_attribute_data["chosen_prompt_embeddings"][i] for i in train_pairs['index']]
    )
    train_rejected_prompt_embeddings = torch.stack(
        [pairwise_per_attribute_data["rejected_prompt_embeddings"][i] for i in train_pairs['index']]
    )

    eval_chosen_embeddings = torch.stack(
        [pairwise_per_attribute_data["chosen_embeddings"][i] for i in eval_pairs.index]
    )
    eval_rejected_embeddings = torch.stack(
        [pairwise_per_attribute_data["rejected_embeddings"][i] for i in eval_pairs.index]
    )
    eval_chosen_prompt_embeddings = torch.stack(
        [pairwise_per_attribute_data["chosen_prompt_embeddings"][i] for i in eval_pairs.index]
    )
    eval_rejected_prompt_embeddings = torch.stack(
        [pairwise_per_attribute_data["rejected_prompt_embeddings"][i] for i in eval_pairs.index]
    )


    train_head_masks = []
    for attrs in train_pairs['attribute']:
        mask = torch.tensor(
            [1.0 if attr in attrs else 0.0 for attr in attributes])
        train_head_masks.append(mask)
    train_head_masks = torch.stack(train_head_masks, dim=0)
    print('Count for each attribute (HelpSteer2)',
          torch.sum(train_head_masks, dim=0))

    pairwise_train_data_dict = {
        "chosen_embeddings": train_chosen_embeddings,
        "rejected_embeddings": train_rejected_embeddings,
        "chosen_prompt_embeddings": train_chosen_prompt_embeddings,
        "rejected_prompt_embeddings": train_rejected_prompt_embeddings,
        "head_mask": train_head_masks.unsqueeze(-1),
        "labels": train_head_masks,
    }

    eval_head_masks = []
    for attribute in eval_pairs['attribute']:
        mask = torch.tensor(
            [1.0 if attr == attribute else 0.0 for attr in attributes])
        eval_head_masks.append(mask)
    eval_head_masks = torch.stack(eval_head_masks, dim=0)

    pairwise_eval_data_dict = {
        "chosen_embeddings": eval_chosen_embeddings,
        "rejected_embeddings": eval_rejected_embeddings,
        "chosen_prompt_embeddings": eval_chosen_prompt_embeddings,
        "rejected_prompt_embeddings": eval_rejected_prompt_embeddings,
        "subset": [
            'helpsteer2-'+attribute for attribute in eval_pairs['attribute']],
        "head_mask": eval_head_masks}

    ds = load_from_disk(dataset_name)['test']
    rm_tokenizer = AutoTokenizer.from_pretrained(config['model_path'])
    ds = build_dataset(ds,rm_tokenizer)
    test_pairwise_df = pd.DataFrame(ds)
    test_pairwise_data = load_data(
        test_path,
        keys=[
            "chosen_embeddings",
            "chosen_prompt_embeddings",
            "rejected_prompt_embeddings",
            "rejected_embeddings"],
    )
    assert len(test_pairwise_df) == test_pairwise_data['chosen_embeddings'].shape[0]

    test_chosen_embeddings = test_pairwise_data['chosen_embeddings']
    test_rejected_embeddings = test_pairwise_data['rejected_embeddings']
    test_chosen_prompt_embeddings = test_pairwise_data['chosen_prompt_embeddings']
    test_rejected_prompt_embeddings = test_pairwise_data['rejected_prompt_embeddings']

    test_head_masks = []
    for attribute in test_pairwise_df['attribute']:
        mask = torch.tensor(
            [1.0 if attr == attribute else 0 for attr in attributes])
        test_head_masks.append(mask)
    test_head_masks = torch.stack(test_head_masks, dim=0)
    print('Count for each attribute (Helpsteer2, test)',
          torch.sum(test_head_masks, dim=0))
    

    pairwise_test_data_dict = {
        "chosen_embeddings": test_chosen_embeddings,
        "rejected_embeddings": test_rejected_embeddings,
        "chosen_prompt_embeddings": test_chosen_prompt_embeddings,
        "rejected_prompt_embeddings": test_rejected_prompt_embeddings,
        "subset": ['helpsteer2-'+attribute for attribute in test_pairwise_df['attribute']],
        "head_mask": test_head_masks}

    return pairwise_train_data_dict, pairwise_eval_data_dict, pairwise_test_data_dict


def load_ultrafeedback_data(train_path, test_path, config):

    attributes = [
        'helpfulness',
        'honesty',
        'instruction_following',
        'truthfulness']
    n_samples_per_attribute = config['n_labeled_samples']
    
    ds = load_from_disk("./dataset/ultrafeedback_per_attribute_pairwise")['train']
    rm_tokenizer = AutoTokenizer.from_pretrained(config['model_path'])
    ds = build_dataset(ds,rm_tokenizer)

    pairwise_df_per_attribute = pd.DataFrame(ds)
    pairwise_per_attribute_data = load_data(
        train_path,
        keys=[
            "chosen_embeddings",
            "chosen_prompt_embeddings",
            "rejected_prompt_embeddings",
            "rejected_embeddings"],
    )
    assert len(pairwise_df_per_attribute) == pairwise_per_attribute_data['chosen_embeddings'].shape[0]

    eval_ratio = 0.15
    train_data = []
    eval_data = []

    pairwise_df_per_attribute.reset_index(inplace = True)
    # Group by attribute and split into training and evaluation
    for attribute, group in pairwise_df_per_attribute.groupby('attribute'):
        n_samples_per_attribute_i = min(n_samples_per_attribute,len(group))
        n_eval = max(1, int(eval_ratio * n_samples_per_attribute_i))
        n_train = n_samples_per_attribute_i - n_eval

        sampled_group = group.sample(
            n=n_samples_per_attribute_i,
            random_state=42,
            replace=False)
        train_data.append(sampled_group.iloc[:n_train])
        eval_data.append(sampled_group.iloc[n_train:n_train + n_eval])

    train_pairs = pd.concat(train_data, ignore_index=True)
    eval_pairs = pd.concat(eval_data, ignore_index=True)
    
    train_pairs = (train_pairs.groupby(['prompt', 'chosen', 'rejected'], as_index=False)
    .agg({
        'attribute': lambda x: list(x), 
        'index': lambda x: random.choice(list(x)) 
    }))
    print(len(train_pairs))
    train_pairs = train_pairs[train_pairs['attribute'].apply(len) < len(attributes)]
    print(len(train_pairs))

    train_chosen_embeddings = torch.stack(
        [pairwise_per_attribute_data["chosen_embeddings"][i] for i in train_pairs['index']]
    )
    train_rejected_embeddings = torch.stack(
        [pairwise_per_attribute_data["rejected_embeddings"][i] for i in train_pairs['index']]
    )
    train_chosen_prompt_embeddings = torch.stack(
        [pairwise_per_attribute_data["chosen_prompt_embeddings"][i] for i in train_pairs['index']]
    )
    train_rejected_prompt_embeddings = torch.stack(
        [pairwise_per_attribute_data["rejected_prompt_embeddings"][i] for i in train_pairs['index']]
    )

    eval_chosen_embeddings = torch.stack(
        [pairwise_per_attribute_data["chosen_embeddings"][i] for i in eval_pairs.index]
    )
    eval_rejected_embeddings = torch.stack(
        [pairwise_per_attribute_data["rejected_embeddings"][i] for i in eval_pairs.index]
    )
    eval_chosen_prompt_embeddings = torch.stack(
        [pairwise_per_attribute_data["chosen_prompt_embeddings"][i] for i in eval_pairs.index]
    )
    eval_rejected_prompt_embeddings = torch.stack(
        [pairwise_per_attribute_data["rejected_prompt_embeddings"][i] for i in eval_pairs.index]
    )
    
    train_head_masks = []
    for attrs in train_pairs['attribute']:
        mask = torch.tensor(
            [1.0 if attr in attrs else 0.0 for attr in attributes])
        train_head_masks.append(mask)
    train_head_masks = torch.stack(train_head_masks, dim=0)
    print('Count for each attribute (UltraFeedback)',
          torch.sum(train_head_masks, dim=0))

    pairwise_train_data_dict = {
        "chosen_embeddings": train_chosen_embeddings,
        "rejected_embeddings": train_rejected_embeddings,
        "chosen_prompt_embeddings": train_chosen_prompt_embeddings,
        "rejected_prompt_embeddings": train_rejected_prompt_embeddings,
        "head_mask": train_head_masks.unsqueeze(-1),
        "labels": train_head_masks,
    }

    eval_head_masks = []
    for attribute in eval_pairs['attribute']:
        mask = torch.tensor(
            [1.0 if attr == attribute else 0.0 for attr in attributes])
        eval_head_masks.append(mask)
    eval_head_masks = torch.stack(eval_head_masks, dim=0)

    pairwise_eval_data_dict = {
        "chosen_embeddings": eval_chosen_embeddings,
        "rejected_embeddings": eval_rejected_embeddings,
        "chosen_prompt_embeddings": eval_chosen_prompt_embeddings,
        "rejected_prompt_embeddings": eval_rejected_prompt_embeddings,
        "subset": ['ultrafeedback-'+attribute for attribute in eval_pairs['attribute']],
        "head_mask": eval_head_masks}

    ds = load_from_disk("./dataset/ultrafeedback_per_attribute_pairwise")['test']
    rm_tokenizer = AutoTokenizer.from_pretrained(config['model_path'])
    ds = build_dataset(ds,rm_tokenizer)
    test_pairwise_df = pd.DataFrame(ds)
    test_pairwise_data = load_data(
        test_path,
        keys=[
            "chosen_embeddings",
            "chosen_prompt_embeddings",
            "rejected_prompt_embeddings",
            "rejected_embeddings"],
    )
    assert len(test_pairwise_df) == test_pairwise_data['chosen_embeddings'].shape[0]

    test_chosen_embeddings = test_pairwise_data['chosen_embeddings']
    test_rejected_embeddings = test_pairwise_data['rejected_embeddings']
    test_chosen_prompt_embeddings = test_pairwise_data['chosen_prompt_embeddings']
    test_rejected_prompt_embeddings = test_pairwise_data['rejected_prompt_embeddings']

    test_head_masks = []
    for attribute in test_pairwise_df['attribute']:
        mask = torch.tensor(
            [1.0 if attr == attribute else 0 for attr in attributes])
        test_head_masks.append(mask)
    test_head_masks = torch.stack(test_head_masks, dim=0)
    print('Count for each attribute (Ultrafeedback, test)',
          torch.sum(test_head_masks, dim=0))

    pairwise_test_data_dict = {
        "chosen_embeddings": test_chosen_embeddings,
        "rejected_embeddings": test_rejected_embeddings,
        "chosen_prompt_embeddings": test_chosen_prompt_embeddings,
        "rejected_prompt_embeddings": test_rejected_prompt_embeddings,
        "subset": ['ultrafeedback-'+attribute for attribute in test_pairwise_df['attribute']],
        "head_mask": test_head_masks}

    return pairwise_train_data_dict, pairwise_eval_data_dict, pairwise_test_data_dict


def load_rpr_data(
        train_path,
        test_path,
        config):

    CATEGORY = ATTRIBUTES_LIST['rpr']
    n_samples_per_attribute = config['n_labeled_samples']
    dataset_name = "./dataset/rpr_per_category_pairwise_add_criterion"
    rm_tokenizer = AutoTokenizer.from_pretrained(config['model_path'])

    pairwise_per_attribute_data = load_data(
        train_path,
        keys=["chosen_embeddings", "chosen_prompt_embeddings", "rejected_prompt_embeddings", "rejected_embeddings"],
    )
    rpr_test_data = load_data(
        test_path,
        keys=["chosen_embeddings", "chosen_prompt_embeddings", "rejected_prompt_embeddings", "rejected_embeddings"],
    )
    
    ds = load_from_disk(dataset_name)['train']
    ds = build_dataset(ds,rm_tokenizer)
    pairwise_df_per_attribute = pd.DataFrame(ds)
    assert len(pairwise_df_per_attribute) == pairwise_per_attribute_data['chosen_embeddings'].shape[0]
    
    pairwise_df_per_attribute = pairwise_df_per_attribute[pairwise_df_per_attribute['attribute'].isin(CATEGORY)]

    eval_ratio = 0.15
    train_data = []
    eval_data = []

    pairwise_df_per_attribute.reset_index(inplace = True)
    # Group by attribute and split into training and evaluation
    for attribute, group in pairwise_df_per_attribute.groupby('attribute'):
        n_samples_per_attribute_i = min(n_samples_per_attribute,len(group))
        n_eval = max(1, int(eval_ratio * n_samples_per_attribute_i))
        n_train = n_samples_per_attribute_i - n_eval

        sampled_group = group.sample(
            n=n_samples_per_attribute_i,
            random_state=42,
            replace=False)
        train_data.append(sampled_group.iloc[:n_train])
        eval_data.append(sampled_group.iloc[n_train:n_train + n_eval])

    train_pairs = pd.concat(train_data, ignore_index=True)
    eval_pairs = pd.concat(eval_data, ignore_index=True)

    train_chosen_embeddings = torch.stack(
        [pairwise_per_attribute_data["chosen_embeddings"][i] for i in train_pairs['index']]
    )
    train_rejected_embeddings = torch.stack(
        [pairwise_per_attribute_data["rejected_embeddings"][i] for i in train_pairs['index']]
    )
    train_chosen_prompt_embeddings = torch.stack(
        [pairwise_per_attribute_data["chosen_prompt_embeddings"][i] for i in train_pairs['index']]
    )
    train_rejected_prompt_embeddings = torch.stack(
        [pairwise_per_attribute_data["rejected_prompt_embeddings"][i] for i in train_pairs['index']]
    )

    eval_chosen_embeddings = torch.stack(
        [pairwise_per_attribute_data["chosen_embeddings"][i] for i in eval_pairs['index']]
    )
    eval_rejected_embeddings = torch.stack(
        [pairwise_per_attribute_data["rejected_embeddings"][i] for i in eval_pairs['index']]
    )
    eval_chosen_prompt_embeddings = torch.stack(
        [pairwise_per_attribute_data["chosen_prompt_embeddings"][i] for i in eval_pairs['index']]
    )
    eval_rejected_prompt_embeddings = torch.stack(
        [pairwise_per_attribute_data["rejected_prompt_embeddings"][i] for i in eval_pairs['index']]
    )

    train_head_masks = []
    for subset in train_pairs['attribute']:
        mask = torch.tensor(
            [1.0 if attr in subset else 0.0 for attr in CATEGORY])
        train_head_masks.append(mask)
    train_head_masks = torch.stack(train_head_masks, dim=0)
    print('Count for each attribute (RPR)', torch.sum(train_head_masks, dim=0))

    pairwise_train_data_dict = {
        "chosen_embeddings": train_chosen_embeddings,
        "rejected_embeddings": train_rejected_embeddings,
        "chosen_prompt_embeddings": train_chosen_prompt_embeddings,
        "rejected_prompt_embeddings": train_rejected_prompt_embeddings,
        "head_mask": train_head_masks.unsqueeze(-1),
        "labels": train_head_masks,
    }

    eval_head_masks = []
    for subset in eval_pairs['attribute']:
        mask = torch.tensor(
            [1.0 if attr == subset else 0.0 for attr in CATEGORY])
        eval_head_masks.append(mask)
    eval_head_masks = torch.stack(eval_head_masks, dim=0)
    print('Count for each attribute (RPR eval)',
          torch.sum(eval_head_masks, dim=0))

    pairwise_eval_data_dict = {
        "chosen_embeddings": eval_chosen_embeddings,
        "rejected_embeddings": eval_rejected_embeddings,
        "chosen_prompt_embeddings": eval_chosen_prompt_embeddings,
        "rejected_prompt_embeddings": eval_rejected_prompt_embeddings,
        "head_mask": eval_head_masks,
        "subset": eval_pairs['attribute'].tolist(),
    }

    ds = load_from_disk(dataset_name)['test']
    ds = build_dataset(ds,rm_tokenizer)
    test_categories = [item for item in ds['attribute']]
    test_indices = [i for i, item in enumerate(
        test_categories) if item in CATEGORY]
    assert len(ds) == rpr_test_data['chosen_embeddings'].shape[0]
    
    test_subset = [test_categories[i] for i in test_indices]
    test_chosen_embeddings = rpr_test_data["chosen_embeddings"][test_indices]
    test_rejected_embeddings = rpr_test_data["rejected_embeddings"][test_indices]
    test_chosen_prompt_embeddings = rpr_test_data["chosen_prompt_embeddings"][test_indices]
    test_rejected_prompt_embeddings = rpr_test_data["rejected_prompt_embeddings"][test_indices]

    test_head_masks = []
    for attrs in test_subset:
        mask = torch.tensor(
            [1.0 if attr in attrs else 0.0 for attr in CATEGORY])
        test_head_masks.append(mask)
    test_head_masks = torch.stack(test_head_masks, dim=0)
    print('Count for each attribute (RPR)', torch.sum(test_head_masks, dim=0))

    pairwise_test_data_dict = {
        "chosen_embeddings": test_chosen_embeddings,
        "rejected_embeddings": test_rejected_embeddings,
        "chosen_prompt_embeddings": test_chosen_prompt_embeddings,
        "rejected_prompt_embeddings": test_rejected_prompt_embeddings,
        "head_mask": test_head_masks,
        "subset": test_subset,
    }

    return pairwise_train_data_dict, pairwise_eval_data_dict, pairwise_test_data_dict


def load_pku_data(train_path, test_path, config):

    attributes = [
        'harmlessness']
    n_samples_per_attribute = config['n_labeled_samples']
    
    ds = load_from_disk("./dataset/pku_alignment_safe_pairwise")['train']
    rm_tokenizer = AutoTokenizer.from_pretrained(config['model_path'])
    ds = build_dataset(ds,rm_tokenizer)

    pairwise_df_per_attribute = pd.DataFrame(ds)
    pairwise_per_attribute_data = load_data(
        train_path,
        keys=[
            "chosen_embeddings",
            "chosen_prompt_embeddings",
            "rejected_prompt_embeddings",
            "rejected_embeddings"],
    )
    assert len(pairwise_df_per_attribute) == pairwise_per_attribute_data['chosen_embeddings'].shape[0]

    eval_ratio = 0.15
    train_data = []
    eval_data = []

    pairwise_df_per_attribute.reset_index(inplace = True)
    # Group by attribute and split into training and evaluation
    for attribute, group in pairwise_df_per_attribute.groupby('attribute'):
        n_samples_per_attribute_i = min(n_samples_per_attribute,len(group))
        n_eval = max(1, int(eval_ratio * n_samples_per_attribute_i))
        n_train = n_samples_per_attribute_i - n_eval

        sampled_group = group.sample(
            n=n_samples_per_attribute_i,
            random_state=42,
            replace=False)
        train_data.append(sampled_group.iloc[:n_train])
        eval_data.append(sampled_group.iloc[n_train:n_train + n_eval])

    train_pairs = pd.concat(train_data, ignore_index=True)
    eval_pairs = pd.concat(eval_data, ignore_index=True)
    
    train_pairs = (train_pairs.groupby(['prompt', 'chosen', 'rejected'], as_index=False)
    .agg({
        'attribute': lambda x: list(x), 
        'index': lambda x: random.choice(list(x)) 
    }))
    print(len(train_pairs))

    train_chosen_embeddings = torch.stack(
        [pairwise_per_attribute_data["chosen_embeddings"][i] for i in train_pairs['index']]
    )
    train_rejected_embeddings = torch.stack(
        [pairwise_per_attribute_data["rejected_embeddings"][i] for i in train_pairs['index']]
    )
    train_chosen_prompt_embeddings = torch.stack(
        [pairwise_per_attribute_data["chosen_prompt_embeddings"][i] for i in train_pairs['index']]
    )
    train_rejected_prompt_embeddings = torch.stack(
        [pairwise_per_attribute_data["rejected_prompt_embeddings"][i] for i in train_pairs['index']]
    )

    eval_chosen_embeddings = torch.stack(
        [pairwise_per_attribute_data["chosen_embeddings"][i] for i in eval_pairs.index]
    )
    eval_rejected_embeddings = torch.stack(
        [pairwise_per_attribute_data["rejected_embeddings"][i] for i in eval_pairs.index]
    )
    eval_chosen_prompt_embeddings = torch.stack(
        [pairwise_per_attribute_data["chosen_prompt_embeddings"][i] for i in eval_pairs.index]
    )
    eval_rejected_prompt_embeddings = torch.stack(
        [pairwise_per_attribute_data["rejected_prompt_embeddings"][i] for i in eval_pairs.index]
    )
    
    train_head_masks = []
    for attrs in train_pairs['attribute']:
        mask = torch.tensor(
            [1.0 if attr in attrs else 0.0 for attr in attributes])
        train_head_masks.append(mask)
    train_head_masks = torch.stack(train_head_masks, dim=0)
    print('Count for each attribute (PKU-SAFE)',
          torch.sum(train_head_masks, dim=0))

    pairwise_train_data_dict = {
        "chosen_embeddings": train_chosen_embeddings,
        "rejected_embeddings": train_rejected_embeddings,
        "chosen_prompt_embeddings": train_chosen_prompt_embeddings,
        "rejected_prompt_embeddings": train_rejected_prompt_embeddings,
        "head_mask": train_head_masks.unsqueeze(-1),
        "labels": train_head_masks,
    }

    eval_head_masks = []
    for attribute in eval_pairs['attribute']:
        mask = torch.tensor(
            [1.0 if attr == attribute else 0.0 for attr in attributes])
        eval_head_masks.append(mask)
    eval_head_masks = torch.stack(eval_head_masks, dim=0)

    pairwise_eval_data_dict = {
        "chosen_embeddings": eval_chosen_embeddings,
        "rejected_embeddings": eval_rejected_embeddings,
        "chosen_prompt_embeddings": eval_chosen_prompt_embeddings,
        "rejected_prompt_embeddings": eval_rejected_prompt_embeddings,
        "subset": ['pku-'+attribute for attribute in eval_pairs['attribute']],
        "head_mask": eval_head_masks}

    ds = load_from_disk("./dataset/pku_alignment_safe_pairwise")['test']
    rm_tokenizer = AutoTokenizer.from_pretrained(config['model_path'])
    ds = build_dataset(ds,rm_tokenizer)
    test_pairwise_df = pd.DataFrame(ds)
    test_pairwise_data = load_data(
        test_path,
        keys=[
            "chosen_embeddings",
            "chosen_prompt_embeddings",
            "rejected_prompt_embeddings",
            "rejected_embeddings"],
    )
    assert len(test_pairwise_df) == test_pairwise_data['chosen_embeddings'].shape[0]

    test_chosen_embeddings = test_pairwise_data['chosen_embeddings']
    test_rejected_embeddings = test_pairwise_data['rejected_embeddings']
    test_chosen_prompt_embeddings = test_pairwise_data['chosen_prompt_embeddings']
    test_rejected_prompt_embeddings = test_pairwise_data['rejected_prompt_embeddings']

    test_head_masks = []
    for attribute in test_pairwise_df['attribute']:
        mask = torch.tensor(
            [1.0 if attr == attribute else 0 for attr in attributes])
        test_head_masks.append(mask)
    test_head_masks = torch.stack(test_head_masks, dim=0)
    print('Count for each attribute (Ultrafeedback, test)',
          torch.sum(test_head_masks, dim=0))

    pairwise_test_data_dict = {
        "chosen_embeddings": test_chosen_embeddings,
        "rejected_embeddings": test_rejected_embeddings,
        "chosen_prompt_embeddings": test_chosen_prompt_embeddings,
        "rejected_prompt_embeddings": test_rejected_prompt_embeddings,
        "subset": ['pku-'+attribute for attribute in test_pairwise_df['attribute']],
        "head_mask": test_head_masks}

    return pairwise_train_data_dict, pairwise_eval_data_dict, pairwise_test_data_dict


def load_mix_data(datasets, config, dataset_paths, total_attributes=7):
    mixed_train_data_dict = {
        "chosen_embeddings": [],
        "rejected_embeddings": [],
        "chosen_prompt_embeddings": [],
        "rejected_prompt_embeddings": [],
        "head_mask": [],
        "labels": []
    }
    mixed_eval_data_dict = {
        "chosen_embeddings": [],
        "rejected_embeddings": [],
        "chosen_prompt_embeddings": [],
        "rejected_prompt_embeddings": [],
        "subset": [],
        "head_mask": []
    }
    mixed_test_data_dict = {
        "chosen_embeddings": [],
        "rejected_embeddings": [],
        "chosen_prompt_embeddings": [],
        "rejected_prompt_embeddings": [],
        "subset": [],
        "head_mask": []
    }
    current_attribute_index = 0
    for dataset in datasets:
        if dataset == 'helpsteer2':
            labeled_path = dataset_paths['helpsteer2']['train_path']
            validation_path = dataset_paths['helpsteer2']['test_path']
            pairwise_train_data_dict, pairwise_eval_data_dict, pairwise_test_data_dict = load_helpsteer_data(
                labeled_path, validation_path, config)
            n_attributes = 5
        elif dataset == 'rlhf-hh':
            hh_helpful_labeled_path = dataset_paths['rlhf-hh']['train_path']
            hh_harmless_labeled_path = dataset_paths['rlhf-hh']['test_path']
            pairwise_train_data_dict, pairwise_eval_data_dict, pairwise_test_data_dict = load_rlhf_hh_data(
                hh_helpful_labeled_path, hh_harmless_labeled_path, config)
            n_attributes = 2
        elif dataset == 'ultrafeedback':
            hh_helpful_labeled_path = dataset_paths['ultrafeedback']['train_path']
            hh_harmless_labeled_path = dataset_paths['ultrafeedback']['test_path']
            pairwise_train_data_dict, pairwise_eval_data_dict, pairwise_test_data_dict = load_ultrafeedback_data(
                hh_helpful_labeled_path, hh_harmless_labeled_path, config)
            n_attributes = 4
        elif dataset == 'rpr':
            rpr_data_path = dataset_paths['rpr']['train_path']
            rpr_test_data_path = dataset_paths['rpr']['test_path']
            pairwise_train_data_dict, pairwise_eval_data_dict, pairwise_test_data_dict = load_rpr_data(
                rpr_data_path, rpr_test_data_path, config)
            n_attributes = 10
        elif dataset == 'pku-safe':
            pku_data_path = dataset_paths['pku-safe']['train_path']
            pku_test_data_path = dataset_paths['pku-safe']['test_path']
            pairwise_train_data_dict, pairwise_eval_data_dict, pairwise_test_data_dict = load_pku_data(
                pku_data_path, pku_test_data_path, config)
            n_attributes = 1

        left_padding = current_attribute_index
        right_padding = total_attributes - (left_padding + n_attributes)

        pairwise_train_data_dict["head_mask"] = torch.nn.functional.pad(
            pairwise_train_data_dict["head_mask"].squeeze(-1), (left_padding, right_padding), "constant", 0).unsqueeze(-1)
        pairwise_train_data_dict["labels"] = torch.nn.functional.pad(
            pairwise_train_data_dict["labels"], (left_padding, right_padding), "constant", 0)
        pairwise_eval_data_dict["head_mask"] = torch.nn.functional.pad(
            pairwise_eval_data_dict["head_mask"], (left_padding, right_padding), "constant", 0)
        pairwise_test_data_dict["head_mask"] = torch.nn.functional.pad(
            pairwise_test_data_dict["head_mask"], (left_padding, right_padding), "constant", 0)

        current_attribute_index += n_attributes

        for key in mixed_train_data_dict.keys():
            mixed_train_data_dict[key].append(pairwise_train_data_dict[key])
        for key in mixed_eval_data_dict.keys():
            mixed_eval_data_dict[key].append(pairwise_eval_data_dict[key])
        for key in mixed_test_data_dict.keys():
            mixed_test_data_dict[key].append(pairwise_test_data_dict[key])

    for key in mixed_train_data_dict:
        if key == "subset":
            mixed_train_data_dict[key] = [
                item for sublist in mixed_train_data_dict[key] for item in sublist]
        else:
            mixed_train_data_dict[key] = torch.cat(
                mixed_train_data_dict[key], dim=0)

    for key in mixed_eval_data_dict:
        if key == "subset":
            mixed_eval_data_dict[key] = [
                item for sublist in mixed_eval_data_dict[key] for item in sublist]
        else:
            mixed_eval_data_dict[key] = torch.cat(
                mixed_eval_data_dict[key], dim=0)

    for key in mixed_test_data_dict:
        if key == "subset":
            mixed_test_data_dict[key] = [
                item for sublist in mixed_test_data_dict[key] for item in sublist]
        else:
            mixed_test_data_dict[key] = torch.cat(
                mixed_test_data_dict[key], dim=0)
    return mixed_train_data_dict, mixed_eval_data_dict, mixed_test_data_dict
