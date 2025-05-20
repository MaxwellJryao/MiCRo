from typing import List
import pandas as pd
import random
import torch
from datasets import Dataset
from eval.criteria import ATTRIBUTES_LIST

def build_dataset(ds, rm_tokenizer, script_args):
    def formatting_func(example):
        chosen_messages = example['chosen']
        rejected_messages = example['rejected']
        if isinstance(chosen_messages, List):
            prompt_plus_chosen_response = rm_tokenizer.apply_chat_template(
                chosen_messages, tokenize=False)
            prompt_plus_rejected_response = rm_tokenizer.apply_chat_template(
                rejected_messages, tokenize=False)
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
            prompt_plus_chosen_response = rm_tokenizer.apply_chat_template(
                chosen_messages, tokenize=False)
            prompt_plus_rejected_response = rm_tokenizer.apply_chat_template(
                rejected_messages, tokenize=False)

        tokens_chosen = rm_tokenizer.encode_plus(
            prompt_plus_chosen_response, return_tensors="pt")
        tokens_rejected = rm_tokenizer.encode_plus(
            prompt_plus_rejected_response, return_tensors="pt")

        prompt_template = rm_tokenizer.apply_chat_template(
            chosen_messages[:-1], tokenize=False, add_generation_prompt=True)
        tokens_prompt = rm_tokenizer.encode_plus(
            prompt_template, return_tensors="pt")['input_ids'][0]

        return {
            "input_ids_chosen": tokens_chosen["input_ids"][0],
            "attention_mask_chosen": tokens_chosen["attention_mask"][0],
            "input_ids_rejected": tokens_rejected["input_ids"][0],
            "attention_mask_rejected": tokens_rejected["attention_mask"][0],
            'prompt_length': len(tokens_prompt),
        }

    ds = ds.map(formatting_func, batched=False, num_proc=30)
    ds = ds.filter(
        lambda x: len(
            x["input_ids_chosen"]) <= script_args.max_length and len(
            x["input_ids_rejected"]) <= script_args.max_length,
        num_proc=30)

    return ds


def process_train_dataset(ds, script_args, attri_index, total_attributes,attributes = None):
    pairwise_df_per_attribute = pd.DataFrame(ds)
    if attributes is None:
        attributes = pairwise_df_per_attribute['attribute'].unique()
    pairwise_df_per_attribute.reset_index(inplace=True)
    pairwise_df_per_attribute = pairwise_df_per_attribute[pairwise_df_per_attribute['attribute'].isin(attributes)]
    n_samples_per_attribute = script_args.n_samples_per_attribute
    train_data = []

    for attribute, group in pairwise_df_per_attribute.groupby('attribute'):
        n_samples_per_attribute_i = min(n_samples_per_attribute, len(group))
        sampled_group = group.sample(
            n=n_samples_per_attribute_i,
            random_state=42,
            replace=False)
        train_data.append(sampled_group.iloc[:n_samples_per_attribute_i])

    train_pairs = pd.concat(train_data, ignore_index=True)

    if 'prompt' in train_pairs.columns:
        other_cols = [col for col in train_pairs.columns if col not in [
            "prompt", "chosen", "rejected", "attribute", "index"]]

        agg_dict = {
            "attribute": lambda x: list(x),
            "index": lambda x: random.choice(list(x)),
        }
        for col in other_cols:
            agg_dict[col] = "first"

        train_pairs = (
            train_pairs
            .groupby(["prompt", "chosen", "rejected"], as_index=False)
            .agg(agg_dict)
        )
        print(len(train_pairs))
        if len(attributes) > 1:
            train_pairs = train_pairs[train_pairs['attribute'].apply(len) < len(attributes)]
        print(len(train_pairs))

    train_head_masks = torch.zeros(
        len(train_pairs['attribute']), total_attributes)
    for i, attribute in enumerate(train_pairs['attribute']):
        if isinstance(attribute, list):
            mask = torch.tensor(
                [1.0 if attr in attribute else 0.0 for attr in attributes])
        else:
            mask = torch.tensor(
                [1.0 if attr == attribute else 0.0 for attr in attributes])
        train_head_masks[i, attri_index:attri_index + len(attributes)] = mask

    print(train_pairs.columns)

    ds_processed = {
        'input_ids_chosen': train_pairs['input_ids_chosen'].tolist(),
        'attention_mask_chosen': train_pairs['attention_mask_chosen'].tolist(),
        'input_ids_rejected': train_pairs['input_ids_rejected'].tolist(),
        'attention_mask_rejected': train_pairs['attention_mask_rejected'].tolist(),
        'prompt_length': train_pairs['prompt_length']}
    ds_processed['head_masks'] = train_head_masks
    ds_processed['labels'] = train_head_masks

    ds_processed = Dataset.from_dict(ds_processed)

    ds_processed.set_format(type="torch")

    return ds_processed


def process_test_dataset(ds, script_args, attri_index, total_attributes,attributes=None):
    pairwise_df_per_attribute = pd.DataFrame(ds)
    if attributes is None:
        attributes = pairwise_df_per_attribute['attribute'].unique()
    pairwise_df_per_attribute.reset_index(inplace=True)
    pairwise_df_per_attribute = pairwise_df_per_attribute[pairwise_df_per_attribute['attribute'].isin(attributes)]
    pairwise_df_per_attribute.reset_index(inplace=True)

    test_head_masks = torch.zeros(
        len(pairwise_df_per_attribute['attribute']), total_attributes)
    for i, attribute in enumerate(pairwise_df_per_attribute['attribute']):
        mask = torch.tensor(
            [1.0 if attr == attribute else 0 for attr in attributes])
        test_head_masks[i, attri_index:attri_index + len(attributes)] = mask

    ds_processed = {
        'input_ids_chosen': pairwise_df_per_attribute['input_ids_chosen'].tolist(),
        'attention_mask_chosen': pairwise_df_per_attribute['attention_mask_chosen'].tolist(),
        'input_ids_rejected': pairwise_df_per_attribute['input_ids_rejected'].tolist(),
        'attention_mask_rejected': pairwise_df_per_attribute['attention_mask_rejected'].tolist(),
        'prompt_length': pairwise_df_per_attribute['prompt_length']}
    ds_processed['head_masks'] = test_head_masks
    ds_processed['labels'] = test_head_masks

    ds_processed = Dataset.from_dict(ds_processed)

    ds_processed.set_format(type="torch")

    return ds_processed, pairwise_df_per_attribute['attribute']