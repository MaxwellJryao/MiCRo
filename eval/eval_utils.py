from datasets import load_dataset, load_from_disk
import numpy as np
import random

random.seed(42)

def process_ultrafeedback():
    print('Processing ultrafeedback')
    rm_types = ['helpfulness', 'honesty', 'instruction_following', 'truthfulness']
    ds_name = "openbmb/UltraFeedback"
    ds_start = 0
    ds_size = 1000

    ds = load_dataset(ds_name)['train'].select(range(ds_start, ds_start + ds_size))

    comps = {}
    for rm_type in rm_types:
        comps[rm_type] = []
        for item in ds:
            comp = {}
            for completion in item['completions']:
                comp[completion['response']] = completion['annotations'][rm_type]['Rating']
            comp = {k: v for k, v in sorted(comp.items(), key=lambda item: item[1], reverse=True)}
            keys = list(comp.keys())
            comps[rm_type].append({'chosen': [{'content': item['instruction'], 'role': 'user'}, {'content': keys[0], 'role': 'assistant'}], 'rejected': [{'content': item['instruction'], 'role': 'user'}, {'content': keys[-1], 'role': 'assistant'}]})
    
    return comps, rm_types

def process_helpsteer2():
    print('Processing helpsteer2')
    rm_types = ['helpfulness', 'correctness', 'coherence', 'complexity', 'verbosity']
    ds_name = 'nvidia/HelpSteer2'
    ds_start = 0

    ds = load_dataset(ds_name)['validation']
    ds_size = 10000
    ds_size = min(ds_size, len(ds) - ds_start)
    ds = ds.select(range(ds_start, ds_start + ds_size))

    comps = {}
    for rm_type in rm_types:
        comps[rm_type] = []
        for i in range(0, len(ds), 2):
            assert ds[i]['prompt'] == ds[i+1]['prompt']
            if ds[i][rm_type] > ds[i+1][rm_type]:
                comps[rm_type].append({'chosen': [{'content': ds[i]['prompt'], 'role': 'user'}, {'content': ds[i]['response'], 'role': 'assistant'}], 'rejected': [{'content': ds[i]['prompt'], 'role': 'user'}, {'content': ds[i+1]['response'], 'role': 'assistant'}]})
            elif ds[i][rm_type] < ds[i+1][rm_type]:
                comps[rm_type].append({'chosen': [{'content': ds[i]['prompt'], 'role': 'user'}, {'content': ds[i+1]['response'], 'role': 'assistant'}], 'rejected': [{'content': ds[i]['prompt'], 'role': 'user'}, {'content': ds[i]['response'], 'role': 'assistant'}]})

    # for rm_type in rm_types:
    #     comps[rm_type] = []
    #     last_id = 0
    #     id = 0
    #     prompt = ds[id]['prompt']
    #     while id < len(ds) - 1:
    #         id += 1
    #         if ds[id]['prompt'] == prompt and id < len(ds) - 1:
    #             continue

    #         if id == len(ds) - 1:
    #             id += 1
    #         maxid = np.argmax(ds[rm_type][last_id:id]).item() + last_id
    #         minid = np.argmin(ds[rm_type][last_id:id]).item() + last_id
    #         comps[rm_type].append({'chosen': [{'content': prompt, 'role': 'user'}, {'content': ds[maxid]['response'], 'role': 'assistant'}], 'rejected': [{'content': prompt, 'role': 'user'}, {'content': ds[minid]['response'], 'role': 'assistant'}]})
    #         last_id = id
    #         if id < len(ds):
    #             prompt = ds[id]['prompt']
    
    return comps, rm_types

def process_helpsteer2_per_attr():
    ds = load_from_disk('semi-reward-models/dataset/helpsteer2_per_attribute_pairwise')
    print('Processing helpsteer2')
    rm_types = ['helpfulness', 'correctness', 'coherence', 'complexity', 'verbosity']
    comps = {rm_type: [] for rm_type in rm_types}

    for item in ds['test']:
        comps[item['attribute']].append({
            'chosen': [{'role': 'user', 'content': item['prompt']}, {'role': 'assistant', 'content': item['chosen']}],
            'rejected': [{'role': 'user', 'content': item['prompt']}, {'role': 'assistant', 'content': item['rejected']}]
        })
    
    return comps, rm_types

def process_ultrafeedback_per_attribute_pairwise_30k():
    ds = load_from_disk('semi-reward-models/dataset/ultrafeedback_per_attribute_pairwise_30k')
    print('Processing ultrafeedback')
    rm_types = ['helpfulness', 'honesty', 'instruction_following', 'truthfulness']
    comps = {rm_type: [] for rm_type in rm_types}

    for item in ds['test']:
        comps[item['attribute']].append({
            'chosen': [{'role': 'user', 'content': item['prompt']}, {'role': 'assistant', 'content': item['chosen']}],
            'rejected': [{'role': 'user', 'content': item['prompt']}, {'role': 'assistant', 'content': item['rejected']}]
        })

    return comps, rm_types


def process_pku_alignment_safe_pairwise(size=800):
    ds = load_from_disk('semi-reward-models/dataset/pku_alignment_safe_pairwise')['test']
    if size and size > 0:
        ds = ds.shuffle(seed=42).select(range(size))
    print('Processing pku alignment safe pairwise')
    rm_types = ['harmlessness']
    comps = {rm_type: [] for rm_type in rm_types}

    for item in ds:
        comps[item['attribute']].append({
            'chosen': [{'role': 'user', 'content': item['prompt']}, {'role': 'assistant', 'content': item['chosen']}],
            'rejected': [{'role': 'user', 'content': item['prompt']}, {'role': 'assistant', 'content': item['rejected']}]
        })

    return comps, rm_types

def process_anthropic_rlhf_hh_pairwise(size=800):
    ds = load_from_disk('semi-reward-models/dataset/anthropic_rlhf_hh_pairwise')['test']
    print('Processing pku alignment safe pairwise')
    rm_types = ['rlhf-hh-helpfulness', 'rlhf-hh-harmlessness']
    comps = {rm_type: [] for rm_type in rm_types}

    for item in ds:
        comps[item['attribute']].append({
            'chosen': item['chosen'],
            'rejected': item['rejected']
        })

    for rm_type in rm_types:
        comps[rm_type] = random.sample(comps[rm_type], min(size, len(comps[rm_type])))

    return comps, rm_types

def process_rpr_per_category_pairwise_add_criterion():
    ds = load_from_disk('semi-reward-models/dataset/rpr_per_category_pairwise_add_criterion')['test']
    print('Processing rpr per category pairwise add criterion')
    rm_types = ['rpr-clarity-and-conciseness',
            'rpr-creativity-and-originality',
            'rpr-cultural-sensitivity',
            'rpr-scientific-rigor',
            'rpr-user-friendliness',
            'rpr-narrative-and-storytelling-quality',
            'rpr-pedagogical-effectiveness',
            'rpr-linguistic-creativity',
            'rpr-factual-accuracy',
            'rpr-humor-and-entertainment-value']
    comps = {rm_type: [] for rm_type in rm_types}

    for item in ds:
        if item['attribute'] in rm_types:
            comps[item['attribute']].append({
                'chosen': item['chosen'],
                'rejected': item['rejected']
            })

    return comps, rm_types

def process_reward_bench_pairwise(size=800):
    ds = load_from_disk('semi-reward-models/dataset/reward_bench_pairwise')['train']
    print('Processing reward bench pairwise')
    categories = {
        "chat": ["alpacaeval-easy", 'alpacaeval-length', 'alpacaeval-hard', 'mt-bench-easy', 'mt-bench-med'],
        "chat-hard": ['mt-bench-hard', 'llmbar-natural', 'llmbar-adver-neighbor', 'llmbar-adver-GPTInst',
                    'llmbar-adver-GPTOut', 'llmbar-adver-manual'],
        "safety": ['refusals-dangerous', 'refusals-offensive', 'xstest-should-refuse', 'xstest-should-respond',
                'donotanswer'],
        "reasoning": ['math-prm', 'hep-cpp', 'hep-go', 'hep-java', 'hep-js', 'hep-python', 'hep-rust'],
    }
    rm_types = list(categories.keys())
    comps = {rm_type: [] for rm_type in rm_types}

    for item in ds:
        for rm_type in rm_types:
            if item['subset'].strip() in categories[rm_type]:
                comps[rm_type].append({'chosen': [{'role': 'user', 'content': item['prompt']}, {'role': 'assistant', 'content': item['chosen']}],
                                       'rejected': [{'role': 'user', 'content': item['prompt']}, {'role': 'assistant', 'content': item['rejected']}]}
                                     )
                break

    return comps, rm_types

def process_reward_bench_pairwise_add_criterion(size=800):
    ds = load_from_disk('semi-reward-models/dataset/reward_bench_pairwise_augmented_context')['train']
    print('Processing reward bench pairwise')
    categories = {
        "chat": ["alpacaeval-easy", 'alpacaeval-length', 'alpacaeval-hard', 'mt-bench-easy', 'mt-bench-med'],
        "chat-hard": ['mt-bench-hard', 'llmbar-natural', 'llmbar-adver-neighbor', 'llmbar-adver-GPTInst',
                    'llmbar-adver-GPTOut', 'llmbar-adver-manual'],
        "safety": ['refusals-dangerous', 'refusals-offensive', 'xstest-should-refuse', 'xstest-should-respond',
                'donotanswer'],
        "reasoning": ['math-prm', 'hep-cpp', 'hep-go', 'hep-java', 'hep-js', 'hep-python', 'hep-rust'],
    }
    rm_types = list(categories.keys())
    comps = {rm_type: [] for rm_type in rm_types}

    for item in ds:
        for rm_type in rm_types:
            if item['subset'].strip() in categories[rm_type]:
                comps[rm_type].append({'chosen': item['chosen'],
                                       'rejected': item['rejected']}
                                     )
                break

    return comps, rm_types

def process_hh_rlhf():
    print('Processing hh-rlhf')
    rm_types = ['harmless-base', 'helpful-base']
    ds_name = 'Anthropic/hh-rlhf'
    ds_start = 0
    ds_size = 1000

    comps = {}
    for rm_type in rm_types:
        comps[rm_type] = []
        ds = load_dataset(ds_name, data_dir=rm_type)['test']
        ds_size = min(ds_size, len(ds) - ds_start)
        ds = ds.select(range(ds_start, ds_start + ds_size))
        for item in ds:
            comps[rm_type].append(item)
    
    return comps, rm_types

def process_reward_bench():
    print('Processing reward-bench')
    categories = {
        "chat": ["alpacaeval-easy", 'alpacaeval-length', 'alpacaeval-hard', 'mt-bench-easy', 'mt-bench-med'],
        "chat-hard": ['mt-bench-hard', 'llmbar-natural', 'llmbar-adver-neighbor', 'llmbar-adver-GPTInst',
                    'llmbar-adver-GPTOut', 'llmbar-adver-manual'],
        "safety": ['refusals-dangerous', 'refusals-offensive', 'xstest-should-refuse', 'xstest-should-respond',
                'donotanswer'],
        "reasoning": ['math-prm', 'hep-cpp', 'hep-go', 'hep-java', 'hep-js', 'hep-python', 'hep-rust'],
    }
    rm_types = list(categories.keys())

    ds = load_dataset('allenai/reward-bench', split='filtered')
    comps = {rm_type: [] for rm_type in rm_types}

    for item in ds:
        for rm_type in rm_types:
            if item['subset'].strip() in categories[rm_type]:
                comps[rm_type].append({'chosen': [{'role': 'user', 'content': item['prompt']}, {'role': 'assistant', 'content': item['chosen']}],
                                       'rejected': [{'role': 'user', 'content': item['prompt']}, {'role': 'assistant', 'content': item['rejected']}]}
                                     )
                break

    return comps, rm_types

def process_hhh():
    print('Processing hhh')
    rm_types = ['harmless', 'helpful', 'honest', 'other']
    comps = {rm_type: [] for rm_type in rm_types}
    for rm_type in rm_types:
        ds = load_dataset('HuggingFaceH4/hhh_alignment', rm_type)['test']
        for item in ds:
            if item['targets']['labels'][0] == 1:
                chosen = item['targets']['choices'][0]
                rejected = item['targets']['choices'][1]
            else:
                chosen = item['targets']['choices'][1]
                rejected = item['targets']['choices'][0]
            comps[rm_type].append({
                'chosen': [{'role': 'user', 'content': item['input']}, {'role': 'assistant', 'content': chosen}],
                'rejected': [{'role': 'user', 'content': item['input']}, {'role': 'assistant', 'content': rejected}]
            })

    return comps, rm_types

def process_rpr():
    print('Processing rpr')
    rm_types = ['rpr']
    comps = {rm_type: [] for rm_type in rm_types}
    ds = load_from_disk('data/rpr_criteria_context')['test']
    for item in ds:
        comps['rpr'].append(item)

    return comps, rm_types

def process_chatbot_arena():
    print('Processing chatbot-arena')
    rm_types = ['chatbot-arena']
    comps = {rm_type: [] for rm_type in rm_types}
    ds = load_dataset('lmsys/chatbot_arena_conversations')['train']
    for item in ds:
        if item['turn'] != 1 or len(item['conversation_a']) != 2 or len(item['conversation_b']) != 2:
            continue
        if item['winner'] == 'model_a':
            comps['chatbot-arena'].append({
                'chosen': item['conversation_a'],
                'rejected': item['conversation_b']
            })
        elif item['winner'] == 'model_b':
            comps['chatbot-arena'].append({
                'chosen': item['conversation_b'],
                'rejected': item['conversation_a']
            })

    return comps, rm_types

# # mt-bench is multi-turn benchmark
# def process_mt_bench():
#     print('Processing mt-bench')
#     rm_types = ['mt-bench']
#     comps = {rm_type: [] for rm_type in rm_types}
#     ds = load_dataset('lmsys/mt_bench_human_judgments', split='human')
#     print(ds)
#     for item in ds:
#         if len(item['conversation_a']) != 2 or len(item['conversation_b']) != 2:
#             continue
#         if item['winner'] == 'model_a':
#             comps['mt-bench'].append({
#                 'chosen': item['conversation_a'],
#                 'rejected': item['conversation_b']
#             })
#         elif item['winner'] == 'model_b':
#             comps['mt-bench'].append({
#                 'chosen': item['conversation_b'],
#                 'rejected': item['conversation_a']
#             })

#     return comps, rm_types

# comps, rm_types = process_chatbot_arena()
# print(len(comps['chatbot-arena']))
# print(comps['chatbot-arena'][0])
# print()