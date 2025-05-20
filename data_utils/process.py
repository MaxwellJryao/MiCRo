from datasets import load_dataset, Dataset, concatenate_datasets
import numpy as np
import pandas as pd

# ds_name = 'nvidia/HelpSteer2'
# ds = load_dataset(ds_name)
helpfulsteer_features = ['helpfulness', 'correctness', 'coherence', 'complexity', 'verbosity']

# ds_train = ds['train']

def get_corrs(ds):
    corrs = {}
    for i in range(len(helpfulsteer_features)):
        for j in range(i+1, len(helpfulsteer_features)):
            feature1 = helpfulsteer_features[i]
            feature2 = helpfulsteer_features[j]
            feature1_scores = ds[feature1]
            feature2_scores = ds[feature2]
            corr = np.corrcoef(feature1_scores, feature2_scores)[0, 1]
            corrs[(feature1, feature2)] = corr

    corrs = sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=False)
    print(corrs)

def load_coherence_complexity_ds(ds):
    half = len(ds) // 2
    coherence_ds = ds.select(range(half))
    complexity_ds = ds.select(range(half, len(ds)))
    res = []
    for i in range(0, len(coherence_ds), 2):
        assert ds[i]['prompt'] == ds[i+1]['prompt']
        if ds[i]['coherence'] > ds[i+1]['coherence']:
            res.append({'chosen': [{'content': ds[i]['prompt'], 'role': 'user'}, {'content': ds[i]['response'], 'role': 'assistant'}], 'rejected': [{'content': ds[i]['prompt'], 'role': 'user'}, {'content': ds[i+1]['response'], 'role': 'assistant'}]})
        elif ds[i]['coherence'] < ds[i+1]['coherence']:
            res.append({'chosen': [{'content': ds[i]['prompt'], 'role': 'user'}, {'content': ds[i+1]['response'], 'role': 'assistant'}], 'rejected': [{'content': ds[i]['prompt'], 'role': 'user'}, {'content': ds[i]['response'], 'role': 'assistant'}]})
    print('coherence dataset size', len(res))

    res2 = []
    for i in range(0, len(complexity_ds), 2):
        assert ds[i]['prompt'] == ds[i+1]['prompt']
        if ds[i]['complexity'] > ds[i+1]['complexity']:
            res2.append({'chosen': [{'content': ds[i]['prompt'], 'role': 'user'}, {'content': ds[i]['response'], 'role': 'assistant'}], 'rejected': [{'content': ds[i]['prompt'], 'role': 'user'}, {'content': ds[i+1]['response'], 'role': 'assistant'}]})
        elif ds[i]['complexity'] < ds[i+1]['complexity']:
            res2.append({'chosen': [{'content': ds[i]['prompt'], 'role': 'user'}, {'content': ds[i+1]['response'], 'role': 'assistant'}], 'rejected': [{'content': ds[i]['prompt'], 'role': 'user'}, {'content': ds[i]['response'], 'role': 'assistant'}]})
    print('complexity dataset size', len(res2))
    res.extend(res2)

    ds = Dataset.from_list(res).shuffle(seed=42)
    return ds

def load_hh_rlhf_ds_chat():
    ds_name = 'Anthropic/hh-rlhf'
    harmless_ds = load_dataset(ds_name, data_dir="harmless-base")['train']
    helpful_ds = load_dataset(ds_name, data_dir="helpful-base")['train']
    dataset = concatenate_datasets([harmless_ds, helpful_ds]).shuffle(seed=42)
    new_ds = []
    for item in dataset:
        chosen = item['chosen']
        rejected = item['rejected']
        chosen_splits = chosen.split('\n\nHuman: ')[1:]
        rejected_splits = rejected.split('\n\nHuman: ')[1:]
        verified = True
        chosen_turns = []
        rejected_turns = []
        for chosen_split in chosen_splits:
            pair = chosen_split.split('\n\nAssistant: ')
            if len(pair) != 2:
                verified = False
                break
            chosen_turns.append({'content': pair[0], 'role': 'user'})
            chosen_turns.append({'content': pair[1], 'role': 'assistant'})
        for rejected_split in rejected_splits:
            pair = rejected_split.split('\n\nAssistant: ')
            if len(pair) != 2:
                verified = False
                break
            rejected_turns.append({'content': pair[0], 'role': 'user'})
            rejected_turns.append({'content': pair[1], 'role': 'assistant'})
        if verified:
            new_ds.append({'chosen': chosen_turns, 'rejected': rejected_turns})

    return Dataset.from_list(new_ds)

# ds = load_hh_rlhf_ds_chat()

# ds = load_coherence_complexity_ds(ds_train)
# print(ds)

# get_corrs(ds_train)