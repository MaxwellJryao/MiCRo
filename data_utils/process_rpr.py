from datasets import load_dataset, Dataset, DatasetDict

ds = load_dataset('microsoft/rpr')

processed_ds = []
for item in ds['train']:
    processed_ds.append({'chosen': [{'role': 'user', 'content': item['prompt']}, {'role': 'assistant', 'content': item['response_a']}],
                         'rejected': [{'role': 'user', 'content': item['prompt']}, {'role': 'assistant', 'content': item['response_b']}],
                         'criteria': item['criteria_x']})
    processed_ds.append({'rejected': [{'role': 'user', 'content': item['prompt']}, {'role': 'assistant', 'content': item['response_a']}],
                         'chosen': [{'role': 'user', 'content': item['prompt']}, {'role': 'assistant', 'content': item['response_b']}],
                         'criteria': item['criteria_y']})
    
processed_ds = Dataset.from_list(processed_ds)

test_processed_ds = []
for item in ds['test']:
    test_processed_ds.append({'chosen': [{'role': 'user', 'content': item['prompt']}, {'role': 'assistant', 'content': item['response_a']}],
                         'rejected': [{'role': 'user', 'content': item['prompt']}, {'role': 'assistant', 'content': item['response_b']}],
                         'criteria': item['criteria_x']})
    test_processed_ds.append({'rejected': [{'role': 'user', 'content': item['prompt']}, {'role': 'assistant', 'content': item['response_a']}],
                         'chosen': [{'role': 'user', 'content': item['prompt']}, {'role': 'assistant', 'content': item['response_b']}],
                         'criteria': item['criteria_y']})
    
test_processed_ds = Dataset.from_list(test_processed_ds)

processed_ds = DatasetDict({'train': processed_ds, 'test': test_processed_ds})
processed_ds.save_to_disk('../data/rpr')


processed_ds = []
for item in ds['train']:
    prompt = f'[criteria] {item["criteria_x"]}\n[context] {item["prompt"]}'
    processed_ds.append({'chosen': [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': item['response_a']}],
                         'rejected': [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': item['response_b']}]})
    prompt = f'[criteria] {item["criteria_y"]}\n[context] {item["prompt"]}'
    processed_ds.append({'rejected': [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': item['response_a']}],
                         'chosen': [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': item['response_b']}]})
    
processed_ds = Dataset.from_list(processed_ds)

test_processed_ds = []
for item in ds['test']:
    prompt = f'[criteria] {item["criteria_x"]}\n[context] {item["prompt"]}'
    test_processed_ds.append({'chosen': [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': item['response_a']}],
                         'rejected': [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': item['response_b']}]})
    prompt = f'[criteria] {item["criteria_y"]}\n[context] {item["prompt"]}'
    test_processed_ds.append({'rejected': [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': item['response_a']}],
                         'chosen': [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': item['response_b']}]})
    
test_processed_ds = Dataset.from_list(test_processed_ds)

processed_ds = DatasetDict({'train': processed_ds, 'test': test_processed_ds})
processed_ds.save_to_disk('../data/rpr_criteria_context')