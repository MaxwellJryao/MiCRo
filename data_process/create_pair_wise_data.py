import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
import pickle
from datasets import Dataset, DatasetDict
from datasets import load_from_disk
from huggingface_hub import login
from eval.criteria import REWARDBENCH_CONTEXT_MAP
# login()

def create_pairwise_dataset_per_attribute_helpsteer2(dataset_path):
    ds1 = load_dataset(dataset_path)['train'].shuffle(seed=0)
    ds2 = load_dataset(dataset_path)['validation'].shuffle(seed=0)

    df = pd.DataFrame(ds1).reset_index().rename(columns={'index': 'original_index'})
    
    attributes = ['helpfulness', 'correctness', 'coherence', 'complexity', 'verbosity']
    pairwise_data = []
    cnt_pair_wise = {key:0 for key in attributes}
    cnt_pair_wise_test = {key:0 for key in attributes}
    training_set = []
    eval_set = []
    test_set = []

    for attr in attributes:
        assert attr in df.columns, f"Missing attribute in dataset: {attr}"
        cnt = 0

        for prompt, group in tqdm(df.groupby('prompt')):
            sorted_group = group.sort_values(by=attr, ascending=False)

            for i in range(len(sorted_group) - 1):
                chosen = sorted_group.iloc[i]
                for j in range(i + 1, len(sorted_group)):
                    rejected = sorted_group.iloc[j]
                    if chosen[attr] >= rejected[attr] + 1 and cnt < 500:
                        test_set.append({
                            'prompt': prompt,
                            'chosen': chosen['response'],
                            'rejected': rejected['response'],
                            'chosen_index': chosen['original_index'],
                            'rejected_index': rejected['original_index'],
                            'attribute': attr,  # Include the attribute in the pair data
                            'chosen_rating':chosen[attr],
                            'rejected_rating':rejected[attr],
                        })
                        cnt += 1
                        cnt_pair_wise_test[attr]+=1
                    elif chosen[attr] >= rejected[attr] + 1:  # Ensure a meaningful difference in score
                        training_set.append({
                            'prompt': prompt,
                            'chosen': chosen['response'],
                            'rejected': rejected['response'],
                            'chosen_index': chosen['original_index'],
                            'rejected_index': rejected['original_index'],
                            'attribute': attr,  # Include the attribute in the pair data
                            'chosen_rating':chosen[attr],
                            'rejected_rating':rejected[attr],
                        })
                        cnt_pair_wise[attr]+=1
                        cnt += 1

    df = pd.DataFrame(ds2).reset_index().rename(columns={'index': 'original_index'})
    for attr in attributes:
        assert attr in df.columns, f"Missing attribute in dataset: {attr}"
        cnt = 0

        for prompt, group in tqdm(df.groupby('prompt')):
            sorted_group = group.sort_values(by=attr, ascending=False)

            for i in range(len(sorted_group) - 1):
                chosen = sorted_group.iloc[i]
                for j in range(i + 1, len(sorted_group)):
                    rejected = sorted_group.iloc[j]
                    if chosen[attr] >= rejected[attr] + 1:
                        test_set.append({
                            'prompt': prompt,
                            'chosen': chosen['response'],
                            'rejected': rejected['response'],
                            'chosen_index': chosen['original_index'],
                            'rejected_index': rejected['original_index'],
                            'attribute': attr,  # Include the attribute in the pair data
                            'chosen_rating':chosen[attr],
                            'rejected_rating':rejected[attr],
                        })
                        cnt_pair_wise_test[attr]+=1
    
    print("Train:", cnt_pair_wise)
    print("Test:",cnt_pair_wise_test)
    pairwise_df_train = pd.DataFrame(training_set)
    pairwise_df_test = pd.DataFrame(test_set)

    train_dataset = Dataset.from_list(training_set)
    test_dataset = Dataset.from_list(test_set)

    dataset = DatasetDict({
    'train': train_dataset,
    'test': test_dataset
})

    dataset.save_to_disk("./dataset/helpsteer2_per_attribute_pairwise")
    # dataset.push_to_hub("pair/helpsteer2_per_attribute_pairwise")

    return pairwise_df_train, pairwise_df_test

def create_pairwise_dataset_per_attribute_ultra(dataset_path):

    with open('./dataset/ultrafeedback_original_content.pkl','rb') as file:
        ds1 = pickle.load(file)

    df = pd.DataFrame(ds1).reset_index().rename(columns={'index': 'original_index'})
    
    attributes = ['ultrafeedback-helpfulness', 'ultrafeedback-honesty', 'ultrafeedback-instruction-following', 'ultrafeedback-truthfulness']
    pairwise_data = []
    training_set = []
    test_set = []

    sampled_prompts = []
    for source, group in df.groupby('source'):
        sampled_prompts.extend(
            group['prompt'].drop_duplicates().sample(frac=0.8, random_state=0).tolist()
        ) 
    train_df = df[df['prompt'].isin(sampled_prompts)]
    remaining_df = df[~df['prompt'].isin(sampled_prompts)]
    print('Num of Training Data',len(train_df))

    cnt_pair_wise = {key:0 for key in attributes}
    for attr in attributes:
        assert attr in df.columns, f"Missing attribute in dataset: {attr}"
        for prompt, group in tqdm(train_df.groupby('prompt')):
                sorted_group = group.sort_values(by=attr, ascending=False)
                for i in range(len(sorted_group) - 1):
                    chosen = sorted_group.iloc[i]
                    for j in range(i + 1, len(sorted_group)):
                        rejected = sorted_group.iloc[j]
                        if chosen[attr] >= rejected[attr] + 1:
                            training_set.append({
                                'prompt': prompt,
                                'chosen': chosen['content'],
                                'rejected': rejected['content'],
                                'attribute': attr,
                                'chosen_rating':chosen[attr],
                            'rejected_rating':rejected[attr],
                            })
                            cnt_pair_wise[attr] += 1
    print("Train:", cnt_pair_wise)

    cnt_pair_wise = {key:0 for key in attributes}
    for attr in attributes:
        assert attr in df.columns, f"Missing attribute in dataset: {attr}"
        for prompt, group in tqdm(remaining_df.groupby('prompt')):
            sorted_group = group.sort_values(by=attr, ascending=False)
            for i in range(len(sorted_group) - 1):
                chosen = sorted_group.iloc[i]
                for j in range(i + 1, len(sorted_group)):
                    rejected = sorted_group.iloc[j]
                    if chosen[attr] >= rejected[attr] + 1:
                        pair = {
                            'prompt': prompt,
                            'chosen': chosen['content'],
                            'rejected': rejected['content'],
                            'attribute': attr,
                            'chosen_rating':chosen[attr],
                            'rejected_rating':rejected[attr],
                        }
                        test_set.append(pair)
                        cnt_pair_wise[attr] += 1

    print("Test:",cnt_pair_wise)
    pairwise_df_train = pd.DataFrame(training_set)
    pairwise_df_test = pd.DataFrame(test_set)

    train_dataset = Dataset.from_list(training_set)
    test_dataset = Dataset.from_list(test_set)

    dataset = DatasetDict({
    'train': train_dataset,
    'test': test_dataset
})

    dataset.save_to_disk("./dataset/ultrafeedback_per_attribute_pairwise")
    # dataset.push_to_hub("pair/ultrafeedback_per_attribute_pairwise")

    return pairwise_df_train, pairwise_df_test



def create_pairwise_dataset_rpr(add_criterion = True):
    ds = load_dataset('microsoft/rpr')
    if add_criterion:
        processed_ds = []
        for item in ds['train']:
            # prompt = f'[criteria] {item["criteria_x"]}\n[context] {item["prompt"]}'
            prompt = f'{item["prompt"]} {item["criteria_x"]}'
            processed_ds.append({'chosen': [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': item['response_a']}],
                                'rejected': [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': item['response_b']}],
                                'attribute':'rpr-' + item["category_x"].lower().replace(' ', '-').replace('&', 'and')})
            #prompt = f'[criteria] {item["criteria_y"]}\n[context] {item["prompt"]}'
            prompt = f'{item["prompt"]} {item["criteria_y"]}'
            processed_ds.append({'rejected': [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': item['response_a']}],
                                'chosen': [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': item['response_b']}],
                                'attribute':'rpr-' + item["category_y"].lower().replace(' ', '-').replace('&', 'and')})
            
        processed_ds = Dataset.from_list(processed_ds)
        pairwise_df_train = pd.DataFrame(processed_ds)
        attribute_counts = pairwise_df_train['attribute'].value_counts()
        print("Train:",attribute_counts)

        test_processed_ds = []
        for item in ds['test']:
            # prompt = f'[criteria] {item["criteria_x"]}\n[context] {item["prompt"]}'
            prompt = f'{item["prompt"]} {item["criteria_x"]}'
            test_processed_ds.append({'chosen': [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': item['response_a']}],
                                'rejected': [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': item['response_b']}],
                                'attribute':'rpr-' + item["category_x"].lower().replace(' ', '-').replace('&', 'and')})
            # prompt = f'[criteria] {item["criteria_y"]}\n[context] {item["prompt"]}'
            prompt = f'{item["prompt"]} {item["criteria_y"]}'
            test_processed_ds.append({'rejected': [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': item['response_a']}],
                                'chosen': [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': item['response_b']}],
                                'attribute':'rpr-' + item["category_y"].lower().replace(' ', '-').replace('&', 'and')})
            
        test_processed_ds = Dataset.from_list(test_processed_ds)
        pairwise_df_test = pd.DataFrame(test_processed_ds)
        attribute_counts = pairwise_df_test['attribute'].value_counts()
        print("Test:",attribute_counts)

        processed_ds = DatasetDict({'train': processed_ds, 'test': test_processed_ds})
        processed_ds.save_to_disk("./dataset/rpr_per_category_pairwise_add_criterion_template2")
        return pairwise_df_train, pairwise_df_test
    else:
        processed_ds = []
        for item in ds['train']:
            processed_ds.append({'chosen': [{'role': 'user', 'content': item['prompt']}, {'role': 'assistant', 'content': item['response_a']}],
                                'rejected': [{'role': 'user', 'content': item['prompt']}, {'role': 'assistant', 'content': item['response_b']}],
                                'criteria': item['criteria_x'],
                                'attribute':'rpr-' + item["category_x"].lower().replace(' ', '-').replace('&', 'and')})
            processed_ds.append({'rejected': [{'role': 'user', 'content': item['prompt']}, {'role': 'assistant', 'content': item['response_a']}],
                                'chosen': [{'role': 'user', 'content': item['prompt']}, {'role': 'assistant', 'content': item['response_b']}],
                                'criteria': item['criteria_y'],
                                'attribute':'rpr-' + item["category_y"].lower().replace(' ', '-').replace('&', 'and')})
            
        processed_ds = Dataset.from_list(processed_ds)
        pairwise_df_train = pd.DataFrame(processed_ds)
        attribute_counts = pairwise_df_train['attribute'].value_counts()
        print("Train:",attribute_counts)

        test_processed_ds = []
        for item in ds['test']:
            test_processed_ds.append({'chosen': [{'role': 'user', 'content': item['prompt']}, {'role': 'assistant', 'content': item['response_a']}],
                                'rejected': [{'role': 'user', 'content': item['prompt']}, {'role': 'assistant', 'content': item['response_b']}],
                                'criteria': item['criteria_x'],
                                'attribute':'rpr-' + item["category_x"].lower().replace(' ', '-').replace('&', 'and')})
            test_processed_ds.append({'rejected': [{'role': 'user', 'content': item['prompt']}, {'role': 'assistant', 'content': item['response_a']}],
                                'chosen': [{'role': 'user', 'content': item['prompt']}, {'role': 'assistant', 'content': item['response_b']}],
                                'criteria': item['criteria_y'],
                                'attribute':'rpr-' + item["category_y"].lower().replace(' ', '-').replace('&', 'and')})
            
        test_processed_ds = Dataset.from_list(test_processed_ds)
        pairwise_df_test = pd.DataFrame(test_processed_ds)
        attribute_counts = pairwise_df_test['attribute'].value_counts()
        print("Test:",attribute_counts)

        processed_ds = DatasetDict({'train': processed_ds, 'test': test_processed_ds})
        processed_ds.save_to_disk("./dataset/rpr_per_category_pairwise")
        return pairwise_df_train, pairwise_df_test
    
def create_pairwise_dataset_rlhf_hh():
    ds = load_dataset("Anthropic/hh-rlhf", data_dir='helpful-base')['train'].shuffle(seed=0)
    print('Train Helpfulness:',len(ds))
    processed_ds = []
    
    for item in tqdm(ds, desc="Processing dataset"):
        # source.append(example['subset'])
        
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
            example = {'chosen': chosen_turns, 'rejected': rejected_turns,'attribute':'rlhf-hh-helpfulness'}
            processed_ds.append(example)
        else:
            continue
    
    ds = load_dataset("Anthropic/hh-rlhf", data_dir='harmless-base')['train'].shuffle(seed=0)
    print('Train Harmlessness:',len(ds))
    for item in tqdm(ds, desc="Processing dataset"):
        # source.append(example['subset'])
        
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
            example = {'chosen': chosen_turns, 'rejected': rejected_turns,'attribute':'rlhf-hh-harmlessness'}
            processed_ds.append(example)
        else:
            continue 
    
    processed_ds = Dataset.from_list(processed_ds)


    test_processed_ds = []
    
    ds = load_dataset("Anthropic/hh-rlhf", data_dir='helpful-base')['test'].shuffle(seed=0)
    print('Test Helpfulness:',len(ds))
    for item in tqdm(ds, desc="Processing dataset"):
        # source.append(example['subset'])
        
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
            example = {'chosen': chosen_turns, 'rejected': rejected_turns,'attribute':'rlhf-hh-helpfulness'}
            test_processed_ds.append(example)
        else:
            continue
    
    ds = load_dataset("Anthropic/hh-rlhf", data_dir='harmless-base')['test'].shuffle(seed=0)
    print('Test Harmlessness:',len(ds))
    for item in tqdm(ds, desc="Processing dataset"):
        # source.append(example['subset'])
        
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
            example = {'chosen': chosen_turns, 'rejected': rejected_turns,'attribute':'rlhf-hh-harmlessness'}
            test_processed_ds.append(example)
        else:
            continue 
    
    test_processed_ds = Dataset.from_list(test_processed_ds)

    processed_ds_all = DatasetDict({'train': processed_ds, 'test': test_processed_ds})
    processed_ds_all.save_to_disk("./dataset/anthropic_rlhf_hh_pairwise")


def create_pairwise_dataset_700K(sample_size=400000):
    ds = load_dataset("hendrydong/preference_700K")['train'].shuffle(seed=42)
    processed_ds = ds.select(range(sample_size))
    test_processed_ds = ds.select(range(sample_size, sample_size + 50000,1))
    
    processed_ds_all = DatasetDict({'train': processed_ds, 'test': test_processed_ds})
    processed_ds_all.save_to_disk("./dataset/400K_pairwise")
    
def create_pairwise_dataset_reward_bench():
    ds = load_dataset("allenai/reward-bench")['filtered'].shuffle(seed=42)
    
    processed_ds_all = DatasetDict({'train': ds})
    processed_ds_all.save_to_disk("./dataset/reward_bench_pairwise")
    subset = ds['subset']
    with open('SemiMultiRM_embeddings_Gemma_2B_rewardmodel_baseline_reward_bench_source.pkl','wb') as file:
        pickle.dump(subset,file)
    
def create_pairwise_dataset_reward_bench_add_criterion():
    ds = load_dataset("allenai/reward-bench")['filtered'].shuffle(seed=42)
    processed_ds = []
    for item in ds:
        criterion = REWARDBENCH_CONTEXT_MAP[item['subset']]
        prompt = f'[criteria] {criterion}\n[context] {item["prompt"]}'
        # prompt = f'{item["prompt"]} {item["criteria_x"]}'
        processed_ds.append({'chosen': [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': item['chosen']}],
                            'rejected': [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': item['rejected']}],
                            'subset':item['subset']})
    processed_ds = Dataset.from_list(processed_ds)
    processed_ds_all = DatasetDict({'train': processed_ds})
    processed_ds_all.save_to_disk("./dataset/reward_bench_pairwise_augmented_context")
    
def create_pairwise_dataset_pku_alignment_safe():
    ds = load_dataset("PKU-Alignment/PKU-SafeRLHF", "default")['train'].shuffle(seed=42)
    print(len(ds))
    training_set=[]
    for idx, item in tqdm(enumerate(ds)):
        if item['is_response_0_safe'] is True and item['is_response_1_safe'] is True:
            continue
        prompt = item['prompt']
        responses = [item['response_0'],item['response_1']]
        chosen = responses[item['safer_response_id']]
        rejected = responses[1-item['safer_response_id']]
        training_set.append({
            'prompt':prompt,
            'chosen':chosen,
            'rejected':rejected,
            'attribute':'harmlessness'
        })
    print(len(training_set))
    ds = load_dataset("PKU-Alignment/PKU-SafeRLHF", "default")['test']
    test_set=[]
    for idx, item in tqdm(enumerate(ds)):
        if item['is_response_0_safe'] is True and item['is_response_1_safe'] is True:
            continue
        prompt = item['prompt']
        responses = [item['response_0'],item['response_1']]
        chosen = responses[item['safer_response_id']]
        rejected = responses[1-item['safer_response_id']]
        test_set.append({
            'prompt':prompt,
            'chosen':chosen,
            'rejected':rejected,
            'attribute':'harmlessness'
        })
    print(len(test_set))
    train_dataset = Dataset.from_list(training_set)
    test_dataset = Dataset.from_list(test_set)

    dataset = DatasetDict({
    'train': train_dataset,
    'test': test_dataset
})
    dataset.save_to_disk("./dataset/pku_alignment_safe_pairwise")
    

def create_pairwise_dataset_shp_alignment():
    ds = load_dataset("stanfordnlp/SHP")['train'].shuffle(seed=42)
    print(len(ds))
    training_set=[]
    for idx, item in tqdm(enumerate(ds)):
        if item['score_A'] <= 3 or item['score_B'] <= 3:
            continue
        prompt = item['history']
        responses = [item['human_ref_A'],item['human_ref_B']]
        chosen = responses[1-item['labels']]
        rejected = responses[item['labels']]
        training_set.append({
            'domain':item['domain'],
            'prompt':prompt,
            'chosen':chosen,
            'rejected':rejected,
        })
    print(len(training_set))
    ds = load_dataset("stanfordnlp/SHP")['test']
    test_set=[]
    for idx, item in tqdm(enumerate(ds)):
        if item['score_A'] <= 3 or item['score_B'] <= 3:
            continue
        prompt = item['history']
        responses = [item['human_ref_A'],item['human_ref_B']]
        chosen = responses[1-item['labels']]
        rejected = responses[item['labels']]
        test_set.append({
            'domain':item['domain'],
            'prompt':prompt,
            'chosen':chosen,
            'rejected':rejected,
        })
    print(len(test_set))
    train_dataset = Dataset.from_list(training_set)
    test_dataset = Dataset.from_list(test_set)

    dataset = DatasetDict({
    'train': train_dataset,
    'test': test_dataset
})

    dataset.save_to_disk("./dataset/stanford_shp_pairwise")
    
create_pairwise_dataset_shp_alignment()
    

# pairwise_df_train, pairwise_df_test =  create_pairwise_dataset_per_attribute_helpsteer2('nvidia/Helpsteer2')
# pairwise_df_train.to_csv('./dataset/helpsteer2_pairwise_train_per_attribute_version3.csv', index=False, encoding='utf-8-sig')
# pairwise_df_test.to_csv('./dataset/helpsteer2_pairwise_test_per_attribute_version3.csv', index=False, encoding='utf-8-sig')


# pairwise_df_train, pairwise_df_test =  create_pairwise_dataset_per_attribute_ultra('openbmb/UltraFeedback')
# pairwise_df_train.to_csv('./dataset/ultrafeedback_pairwise_train_per_attribute.csv', index=False, encoding='utf-8-sig',escapechar='\\')
# pairwise_df_test.to_csv(
#     './dataset/ultrafeedback_pairwise_test_per_attribute.csv', 
#     index=False, 
#     encoding='utf-8-sig', 
#     escapechar='\\'
# )

# pairwise_df_train, pairwise_df_test = create_pairwise_dataset_rpr(add_criterion = True)
# pairwise_df_train.to_csv('./dataset/rpr_per_category_pairwise_add_criterion.csv', index=False, encoding='utf-8-sig')
# pairwise_df_test.to_csv('./dataset/rpr_per_category_pairwise_add_criterion.csv', index=False, encoding='utf-8-sig')

# pairwise_df_train, pairwise_df_test = create_pairwise_dataset_rpr(add_criterion = False)
# pairwise_df_train.to_csv('./dataset/rpr_per_category_pairwise.csv', index=False, encoding='utf-8-sig')
# pairwise_df_test.to_csv('./dataset/rpr_per_category_pairwise.csv', index=False, encoding='utf-8-sig')

# create_pairwise_dataset_rlhf_hh()
# create_pairwise_dataset_700K()
# create_pairwise_dataset_rpr(add_criterion = True)
# create_pairwise_dataset_reward_bench()
# create_pairwise_dataset_reward_bench_add_criterion()
# create_pairwise_dataset_pku_alignment_safe()