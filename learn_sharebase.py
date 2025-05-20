from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Tuple
from accelerate import Accelerator
import evaluate
import numpy as np
import os
import torch
import torch.nn as nn
from datasets import load_dataset, concatenate_datasets, Dataset, load_from_disk
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    LlamaTokenizer,
    PreTrainedModel,
)
from transformers.trainer_pt_utils import nested_detach
import jsonlines
from trl import RewardTrainer
from transformers.utils import PaddingStrategy
torch.backends.cuda.matmul.allow_tf32 = True
# os.environ["HF_TOKEN"] = ''
# os.environ['CUDA_VISIBLE_DEVICES'] = '7'
import wandb
import data_utils.process as data_process

accelerator = Accelerator()

RPR_CATEGORY_LIST = ['rpr-clarity-and-conciseness',
            'rpr-creativity-and-originality',
            'rpr-cultural-sensitivity',
            'rpr-scientific-rigor',
            'rpr-user-friendliness',
            'rpr-narrative-and-storytelling-quality',
            'rpr-pedagogical-effectiveness',
            'rpr-linguistic-creativity',
            'rpr-factual-accuracy',
            'rpr-humor-and-entertainment-value']

# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """
    per_device_train_batch_size: Optional[int] = field(default=1) 
    per_device_eval_batch_size: Optional[int] = field(default=1)
    gradient_accumulation_steps: Optional[int] = field(default=8)
    learning_rate: Optional[float] = field(default=2e-3)
    num_train_epochs: Optional[int] = field(
        default=1,
        metadata={"help": "The number of training epochs for the reward model."},
    )
    optim: Optional[str] = field(
        default="adamw_torch",
        metadata={"help": "The optimizer to use."},
    )
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "The lr scheduler"},)
    max_length: Optional[int] = field(default=4096) 
    use_lora: Optional[bool] = field(default=False)
    base_model: Optional[str] = field(default='Skywork/Skywork-Reward-Llama-3.1-8B-v0.2')
    wandb_name: Optional[str] = field(default="mixture_BT",)
    log_dir: Optional[str] = field(default='./output_models')
    loss_type: Optional[str] = field(default='mixture_reward')
    use_smallset: Optional[bool] = field(default=False)
    freeze_pretrained: Optional[bool] = field(default=True)
    data_path: Optional[str] = field(default='llm-blender/Unified-Feedback')
    num_heads: Optional[int] = field(default=5)
    orthogonal_loss_weight: Optional[float] = field(default=0)
    norm_loss_weight: Optional[float] = field(default=0)
    corr_loss_weight: Optional[float] = field(default=0.0)
    load_balance_loss_weight: Optional[float] = field(default=0.0)
    use_router: Optional[bool] = field(default=True)
    sanity_check: Optional[bool] = field(default=False)
    manual_seed: Optional[int] = field(default=0)
    eval_strategy: Optional[str] = field(default='steps')
    save_strategy: Optional[str] = field(default='steps')


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
torch.manual_seed(script_args.manual_seed)

if accelerator.is_main_process:
    print('Arguments:')
    for arg in vars(script_args):
        print(format(arg, '<30'), format(str(getattr(script_args, arg)), '<'))   # str, arg_type

# if script_args.corr_loss_weight > 0:
#     assert script_args.per_device_train_batch_size > 1, "Correlation loss only works with batch size > 1"

model_name = script_args.base_model
tokenizer_name = model_name
data_path = script_args.data_path

token_patterns = {
    # Llama3 token IDs of "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    "llama3": [128009, 128006, 78191, 128007, 271],
    # Gemma2 token IDs of "<end_of_turn>\n<start_of_turn>model\n"
    "gemma2": [107, 108, 106, 2516, 108],
}


def find_token_for_gating(lst, model_family):
    """Find the last occurrence of a token_pattern in a list."""
    token_pattern = token_patterns[model_family]
    token_pattern_len = len(token_pattern)
    search_end = len(lst)
    for j in range(search_end - token_pattern_len, -1, -1):
        if lst[j : j + token_pattern_len] == token_pattern:
            return j
    raise ValueError("Token pattern not found in the list.")

def build_dataset_mix(ds, tokenizer, size=None):    
    # ds = ds.select(range(0, len(ds), 5))
    if size is not None:
        ds = ds.select(range(0, size))

    def formatting_func(example):
        kwargs = {"return_tensors": "pt"}
        # kwargs = {"padding": 'max_length', "truncation": True, "max_length": script_args.max_length, "return_tensors": "pt"}
        chosen_messages = example['chosen']
        rejected_messages = example['rejected']
        if isinstance(chosen_messages, List):
            prompt_plus_chosen_response = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
            prompt_plus_rejected_response = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
        else:
            prompt_plus_chosen_response = chosen_messages
            prompt_plus_rejected_response = rejected_messages
        tokens_chosen = tokenizer.encode_plus(prompt_plus_chosen_response, **kwargs)
        tokens_rejected = tokenizer.encode_plus(prompt_plus_rejected_response, **kwargs)

        prompt_template = tokenizer.apply_chat_template(chosen_messages[:-1], tokenize=False, add_generation_prompt=True)
        tokens_prompt = tokenizer.encode_plus(prompt_template, **kwargs)['input_ids'][0]

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


def build_dataset(data_path, tokenizer, split='train', size=None):
    try:
        ds = load_dataset(data_path, 'all', split=split)
    except:
        ds = load_dataset(data_path, split=split)
    
    # if split == 'val':
    ds = ds.filter(lambda example: example['conv_A_rating'] != example['conv_B_rating'], num_proc=30)

    if size is not None:
        ds = ds.select(range(0, size))

    if split != 'val' and script_args.use_smallset:
        ds = ds.select(range(0, len(ds), 10)) #############

    def formatting_func(example):
        kwargs = {"return_tensors": "pt"}
        # kwargs = {"padding": 'max_length', "truncation": True, "max_length": script_args.max_length, "return_tensors": "pt"}
        if example['conv_A_rating'] > example['conv_B_rating']:
            chosen_messages = example['conv_A']
            rejected_messages = example['conv_B']
            margin = example['conv_A_rating'] - example['conv_B_rating']
        else:
            chosen_messages = example['conv_B']
            rejected_messages = example['conv_A']
            margin = example['conv_B_rating'] - example['conv_A_rating']
        
        if 'summarize' in example['source']:
            chosen_messages[0]['content'] = 'Generate one-sentence summary for the following post: ' + chosen_messages[0]['content'].strip()
            rejected_messages[0]['content'] = 'Generate one-sentence summary for the following post: ' + rejected_messages[0]['content'].strip()
        
        prompt_plus_chosen_response = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
        prompt_plus_rejected_response = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
        tokens_chosen = tokenizer.encode_plus(prompt_plus_chosen_response, **kwargs)
        tokens_rejected = tokenizer.encode_plus(prompt_plus_rejected_response, **kwargs)

        # add label mask
        prompt_template = tokenizer.apply_chat_template(chosen_messages[:-1], tokenize=False, add_generation_prompt=True)
        tokens_prompt = tokenizer.encode_plus(prompt_template, **kwargs)['input_ids'][0]
        # label_chosen = tokens_chosen["input_ids"][0].clone()
        # label_chosen[:len(tokens_prompt)] = -100
        # label_rejected = tokens_rejected["input_ids"][0].clone()
        # label_rejected[:len(tokens_prompt)] = -100
        return {
            "input_ids_chosen": tokens_chosen["input_ids"][0], "attention_mask_chosen": tokens_chosen["attention_mask"][0],
            "input_ids_rejected": tokens_rejected["input_ids"][0], "attention_mask_rejected": tokens_rejected["attention_mask"][0],
            "margin": margin, 'prompt_length': len(tokens_prompt),
        }

    ds = ds.map(formatting_func, batched=False, num_proc=20)
    ds = ds.filter(lambda x: len(x["input_ids_chosen"]) <= script_args.max_length and len(x["input_ids_rejected"]) <= script_args.max_length, num_proc=30)
    remove_columns = []
    for col in ds.column_names:
        if 'input' not in col and 'attention' not in col and 'margin' not in col and 'length' not in col:
            remove_columns.append(col)
    ds = ds.remove_columns(remove_columns)

    ds.set_format(type="torch")
    return ds

def build_dataset_80k(data_path, tokenizer, split='train', size=None):
    ds = load_dataset(data_path, split=split)

    if size is not None:
        ds = ds.select(range(0, size))

    def formatting_func(example):
        kwargs = {"truncation": True, "max_length": script_args.max_length, "return_tensors": "pt"}
        prompt = example['chosen'][0]['content']

        chosen_messages = example['chosen']
        rejected_messages = example['rejected']

        prompt_plus_chosen_response = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
        prompt_plus_rejected_response = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
        tokens_chosen = tokenizer.encode_plus(prompt_plus_chosen_response, **kwargs)
        tokens_rejected = tokenizer.encode_plus(prompt_plus_rejected_response, **kwargs)
        # add label mask
        prompt_template = tokenizer.apply_chat_template([{"content": prompt, "role": "user" }], tokenize=False, add_generation_prompt=True)
        tokens_prompt = tokenizer.encode_plus(prompt_template, **kwargs)['input_ids'][0]
        label_chosen = tokens_chosen["input_ids"][0].clone()
        label_chosen[:len(tokens_prompt)] = -100
        label_rejected = tokens_rejected["input_ids"][0].clone()
        label_rejected[:len(tokens_prompt)] = -100
        return {
            "input_ids_chosen": tokens_chosen["input_ids"][0], "attention_mask_chosen": tokens_chosen["attention_mask"][0],
            "input_ids_rejected": tokens_rejected["input_ids"][0], "attention_mask_rejected": tokens_rejected["attention_mask"][0],
            "label_chosen": label_chosen,  'label_rejected': label_rejected, 'prompt_length': len(tokens_prompt),
        }

    ds = ds.map(formatting_func, batched=False, num_proc=10) 
    ds = ds.filter(lambda x: len(x["input_ids_chosen"]) <= script_args.max_length and len(x["input_ids_rejected"]) <= script_args.max_length, num_proc=30)
    ds.set_format(type="torch")
    return ds

def build_dataset_helpsteer(ds, tokenizer, size=None):
    if size is not None:
        ds = ds.shuffle(seed=42).select(range(0, size))

    def formatting_func(example):
        kwargs = {"return_tensors": "pt"}
        if isinstance(example['chosen'], list):
            prompt = example['chosen'][0]['content']
            chosen_messages = example['chosen']
            rejected_messages = example['rejected']
        else:
            prompt = example['prompt']
            chosen_messages = [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': example['chosen']}]
            rejected_messages = [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': example['rejected']}]

        prompt_plus_chosen_response = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
        prompt_plus_rejected_response = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
        tokens_chosen = tokenizer(prompt_plus_chosen_response, **kwargs)
        tokens_rejected = tokenizer(prompt_plus_rejected_response, **kwargs)
        prompt_len = find_token_for_gating(tokens_chosen["input_ids"][0].tolist(), 'llama3')
        # add label mask
        # prompt_template = tokenizer.apply_chat_template([{"content": prompt, "role": "user" }], tokenize=False, add_generation_prompt=True)
        # tokens_prompt = tokenizer.encode_plus(prompt_template, **kwargs)['input_ids'][0]
        label_chosen = tokens_chosen["input_ids"][0].clone()
        label_chosen[:prompt_len] = -100
        label_rejected = tokens_rejected["input_ids"][0].clone()
        label_rejected[:prompt_len] = -100
        return {
            "input_ids_chosen": tokens_chosen["input_ids"][0], "attention_mask_chosen": tokens_chosen["attention_mask"][0],
            "input_ids_rejected": tokens_rejected["input_ids"][0], "attention_mask_rejected": tokens_rejected["attention_mask"][0],
            "label_chosen": label_chosen,  'label_rejected': label_rejected, 'prompt_length': prompt_len,
        }

    ds = ds.map(formatting_func, batched=False, num_proc=10) 
    ds = ds.filter(lambda x: len(x["input_ids_chosen"]) <= script_args.max_length and len(x["input_ids_rejected"]) <= script_args.max_length, num_proc=10)
    ds.set_format(type="torch")
    return ds


def build_dataset_rpr(ds, tokenizer, size=None):
    
    if size is not None:
        ds = ds.select(range(0, size))
    
    ds = ds.filter(
        lambda x: x["attribute"] in RPR_CATEGORY_LIST,num_proc=10)

    def formatting_func(example):
        kwargs = {"return_tensors": "pt"}
        if isinstance(example['chosen'], list):
            prompt = example['chosen'][0]['content']
            chosen_messages = example['chosen']
            rejected_messages = example['rejected']
        else:
            prompt = example['prompt']
            chosen_messages = [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': example['chosen']}]
            rejected_messages = [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': example['rejected']}]

        prompt_plus_chosen_response = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
        prompt_plus_rejected_response = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
        tokens_chosen = tokenizer.encode_plus(prompt_plus_chosen_response, **kwargs)
        tokens_rejected = tokenizer.encode_plus(prompt_plus_rejected_response, **kwargs)
        # add label mask
        prompt_template = tokenizer.apply_chat_template([{"content": prompt, "role": "user" }], tokenize=False, add_generation_prompt=True)
        tokens_prompt = tokenizer.encode_plus(prompt_template, **kwargs)['input_ids'][0]
        label_chosen = tokens_chosen["input_ids"][0].clone()
        label_chosen[:len(tokens_prompt)] = -100
        label_rejected = tokens_rejected["input_ids"][0].clone()
        label_rejected[:len(tokens_prompt)] = -100
        return {
            "input_ids_chosen": tokens_chosen["input_ids"][0], "attention_mask_chosen": tokens_chosen["attention_mask"][0],
            "input_ids_rejected": tokens_rejected["input_ids"][0], "attention_mask_rejected": tokens_rejected["attention_mask"][0],
            "label_chosen": label_chosen,  'label_rejected': label_rejected, 'prompt_length': len(tokens_prompt),
        }

    ds = ds.map(formatting_func, batched=False, num_proc=10) 
    ds = ds.filter(lambda x: len(x["input_ids_chosen"]) <= script_args.max_length and len(x["input_ids_rejected"]) <= script_args.max_length, num_proc=10)
    ds.set_format(type="torch")
    return ds

# initialize wandb
# if accelerator.is_main_process:
#     wandb.init(project='MultiRewardLearning', name=script_args.wandb_name)

# Define the training args. Needs to be done before the model is loaded if you are using deepspeed.
model_name_split = model_name.split("/")[-1]
output_name = f"{script_args.log_dir}/{model_name_split}_{script_args.wandb_name}"

training_args = TrainingArguments(
    output_dir=os.path.join(output_name, 'logs'),
    learning_rate=script_args.learning_rate,
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    per_device_eval_batch_size=script_args.per_device_eval_batch_size,
    num_train_epochs=script_args.num_train_epochs,
    # weight_decay=script_args.weight_decay,
    eval_strategy=script_args.eval_strategy,
    eval_steps=100000,
    save_strategy=script_args.save_strategy,
    save_steps=200,
    save_total_limit=3,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    gradient_checkpointing=True, 
    remove_unused_columns=False,
    label_names=[],
    bf16=True,
    logging_strategy="steps",
    logging_steps=1,
    warmup_ratio=0.05,
    optim=script_args.optim,
    lr_scheduler_type=script_args.lr_scheduler_type,
    run_name=script_args.wandb_name,
    # max_grad_norm=5.0,
    report_to='none',
    gradient_checkpointing_kwargs={"use_reentrant": False},
    ddp_find_unused_parameters=False,
    # load_best_model_at_end=True,
)
# Load the value-head model and tokenizer.
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast = False)
tokenizer.model_max_length = script_args.max_length
# if 'gemma' not in model_name:
if 'Llama' in model_name:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
else:
    tokenizer.pad_token = tokenizer.eos_token

data_paths = data_path.split('-')
train_datasets = []
eval_datasets = []

for data_path in data_paths:
    if 'helpsteer2_per_attribute_pairwise_augmented' in data_path:
        dataset = load_from_disk('semi-reward-models/dataset/helpsteer2_per_attribute_pairwise_augmented')['train']
        dataset = build_dataset_helpsteer(dataset, tokenizer)
        dataset_split = dataset.train_test_split(test_size=0.05)
        train_dataset, eval_dataset = dataset_split['train'], dataset_split['test']
        # train_dataset = dataset
        # eval_dataset = dataset
    elif 'helpsteer2_per_attribute_pairwise' in data_path:
        dataset = load_from_disk('semi-reward-models/dataset/helpsteer2_per_attribute_pairwise')['train']
        dataset = build_dataset_helpsteer(dataset, tokenizer)
        dataset_split = dataset.train_test_split(test_size=0.01)
        train_dataset, eval_dataset = dataset_split['train'], dataset_split['test']
    elif 'ultrafeedback_per_attribute_pairwise_30k' in data_path:
        dataset = load_from_disk('semi-reward-models/dataset/ultrafeedback_per_attribute_pairwise_30k')['train']
        dataset = build_dataset_helpsteer(dataset, tokenizer)
        dataset_split = dataset.train_test_split(test_size=0.01)
        train_dataset, eval_dataset = dataset_split['train'], dataset_split['test']
    elif 'rpr_per_category_pairwise_add_criterion' in data_path:
        dataset = load_from_disk('semi-reward-models/dataset/rpr_per_category_pairwise_add_criterion')['train']
        dataset = build_dataset_rpr(dataset, tokenizer)
        dataset_split = dataset.train_test_split(test_size=0.01)
        train_dataset, eval_dataset = dataset_split['train'], dataset_split['test']
    elif 'rpr_per_category_pairwise' in data_path:
        dataset = load_from_disk('semi-reward-models/dataset/rpr_per_category_pairwise')['train']
        dataset = build_dataset_rpr(dataset, tokenizer)
        dataset_split = dataset.train_test_split(test_size=0.01)
        train_dataset, eval_dataset = dataset_split['train'], dataset_split['test']
    elif 'pku_alignment_safe_pairwise' in data_path:
        dataset = load_from_disk('semi-reward-models/dataset/pku_alignment_safe_pairwise')['train']
        dataset = build_dataset_helpsteer(dataset, tokenizer, size=5000)
        dataset_split = dataset.train_test_split(test_size=0.01)
        train_dataset, eval_dataset = dataset_split['train'], dataset_split['test']
    elif 'Unified' in data_path:
        train_dataset = build_dataset(data_path, tokenizer, split='train') 
        eval_dataset = build_dataset(data_path, tokenizer, split='val')
    elif '80K' in data_path:
        dataset = build_dataset_80k(data_path, tokenizer, split='train')
        dataset_split = dataset.train_test_split(test_size=0.002)
        train_dataset, eval_dataset = dataset_split['train'], dataset_split['test'] # .select(range(1))
    elif 'helpsteer' in data_path.lower():
        dataset = load_dataset('nvidia/HelpSteer2')
        dataset = data_process.load_coherence_complexity_ds(dataset['train'])
        dataset = build_dataset_helpsteer(dataset, tokenizer)
        dataset_split = dataset.train_test_split(test_size=0.005)
        train_dataset, eval_dataset = dataset_split['train'], dataset_split['test']
    elif 'hh-rlhf' in data_path:
        # harmless_ds = load_dataset("Anthropic/hh-rlhf", data_dir="harmless-base")['train']
        # helpful_ds = load_dataset("Anthropic/hh-rlhf", data_dir="helpful-base")['train']
        # dataset = concatenate_datasets([harmless_ds, helpful_ds]).shuffle(seed=42)
        # dataset = build_dataset_mix(dataset, tokenizer)

        dataset = data_process.load_hh_rlhf_ds_chat()
        if script_args.sanity_check:
            dataset = dataset.select(range(0, 100))
        dataset = build_dataset_mix(dataset, tokenizer)
        dataset_split = dataset.train_test_split(test_size=0.01)
        train_dataset, eval_dataset = dataset_split['train'], dataset_split['test']
    elif '700k' in data_path:
        dataset = load_dataset('hendrydong/preference_700K', split='train')
        dataset = build_dataset_mix(dataset, tokenizer)
        dataset_split = dataset.train_test_split(test_size=0.01)
        train_dataset, eval_dataset = dataset_split['train'], dataset_split['test']
    else:
        dataset = load_dataset(data_path, split='train')
        dataset = build_dataset_mix(dataset, tokenizer) 
        dataset_split = dataset.train_test_split(test_size=0.01)
        train_dataset, eval_dataset = dataset_split['train'], dataset_split['test']

    train_datasets.append(train_dataset)
    eval_datasets.append(eval_dataset)

train_dataset = concatenate_datasets(train_datasets)
eval_dataset = concatenate_datasets(eval_datasets)


#######################################################
print(len(train_dataset), len(eval_dataset))


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            # print(name)
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def freeze_trainable_parameters(model, exclude=[]):
    for name, param in model.named_parameters():
        if name not in exclude:
            param.requires_grad = False


# device = accelerator.local_process_index
device = int(os.environ['LOCAL_RANK'])
print(device)


model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=1, # device_map=device, 
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

class CombinedScoreHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 1):
        """
        parameters description:
        - input_dim: input feature dimension, usually the same as the output dimension of the backbone
        - hidden_dim: intermediate layer dimension, can be adjusted
        - output_dim: output dimension, usually 1 (regression score)
        """
        super().__init__()
        # learnable part: learnable_net
        self.learnable_net = nn.Linear(input_dim, output_dim, bias=False)
        # frozen prior_net: this part remains unchanged during training
        self.prior_net = nn.Linear(input_dim, output_dim, bias=False)
        # freeze the parameters of prior_net
        for param in self.prior_net.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward propagation:
        1. x passes through learnable_net and prior_net respectively
        2. return the sum of the two parts
        """
        y_theta = self.learnable_net(x)  # learnable part
        y_p = self.prior_net(x)          # frozen prior part
        return y_theta + y_p

##########################
if script_args.freeze_pretrained:
    mlp_layer = CombinedScoreHead(model.config.hidden_size, script_args.num_heads)
    mlp_layer.to(device)
    freeze_trainable_parameters(model)
    model.score = mlp_layer

model.resize_token_embeddings(len(tokenizer))
print_trainable_parameters(model)
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)
model.config.pad_token_id = tokenizer.pad_token_id

# Define the metric that we'll use for validation.
accuracy = evaluate.load('accuracy')

def compute_metrics(eval_pred):
    predictions = eval_pred.predictions
    predictions = np.argmax(predictions, axis=1)
    labels = np.zeros(predictions.shape)
    return accuracy.compute(predictions=predictions, references=labels)


@dataclass
class RewardDataCollatorWithPadding:
    tokenizer: AutoTokenizer
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        merged_features = []
        margins = []
        for feature in features:
            merged_features.append(
                {
                    "input_ids": feature["input_ids_chosen"],
                    "attention_mask": feature["attention_mask_chosen"],
                }
            )
            merged_features.append(
                {
                    "input_ids": feature["input_ids_rejected"],
                    "attention_mask": feature["attention_mask_rejected"],
                }
            )
            if 'margin' in feature.keys():
                margins.append(feature['margin'])
        batch = self.tokenizer.pad(
            merged_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "return_loss": True,
            'prompt_length': torch.tensor([feature['prompt_length'] for feature in features]),
        }
        return batch

class RewardTrainer_new(RewardTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], output_hidden_states=True)
        rewards, last_hidden_state = outputs.logits, outputs.hidden_states[-1][:, -1, :] # hidden states of the last layer, last token
        bsz = rewards.size(0)
        jidx = torch.arange(0, bsz, 2)
        kidx = jidx + 1
        rewards_j = rewards[jidx]
        rewards_k = rewards[kidx]
        ###################################
        if script_args.loss_type == 'origin':
            loss = - nn.functional.logsigmoid(rewards_j - rewards_k).mean()
            # if accelerator.is_main_process:
            #     wandb.log({'origin BT loss': loss})
        elif script_args.loss_type == 'margin':
            loss = -nn.functional.logsigmoid(rewards_j - rewards_k - torch.tensor(inputs["margin"], device=inputs["margin"][0].device).view(-1,1)).mean()
            if accelerator.is_main_process:
                wandb.log({'margin BT loss': loss})
        elif script_args.loss_type == 'labelsmooth':
            loss = - 0.9 * nn.functional.logsigmoid(rewards_j - rewards_k).mean() - 0.1 * nn.functional.logsigmoid(rewards_k - rewards_j).mean() 
            if accelerator.is_main_process:
                wandb.log({'labelsmooth BT loss': loss})
        elif script_args.loss_type == 'mixture_reward':
            if script_args.use_router:
                print(model)
                unwrap_model = self.accelerator.unwrap_model(self.model)
                router_weights = unwrap_model.router(last_hidden_state.bfloat16())
                # router_weights_mean = router_weights.mean(dim=0)
                # load_balance_loss = script_args.load_balance_loss_weight * ((router_weights - router_weights_mean) ** 2).mean()
                load_balance_loss = (router_weights.softmax(dim=-1)).var(dim=1).mean()
                router_weights_j = router_weights[jidx]
                router_weights_k = router_weights[kidx]

                BTloss = - nn.functional.logsigmoid(((rewards_j * router_weights_j.softmax(dim=-1) - rewards_k * router_weights_k.softmax(dim=-1))).sum(dim=1)).mean()
            else:
                router_weights = model.router
                load_balance_loss = (router_weights.softmax(dim=-1)).var(dim=0)
                BTloss = - torch.log((nn.functional.sigmoid(rewards_j - rewards_k) * model.router.softmax(dim=-1)).sum(dim=-1)).mean()
            
            orthogonal_loss = 0
            for i in range(script_args.num_heads):
                # norm_loss += script_args.norm_loss_weight * torch.abs(torch.linalg.vector_norm(model.score.weight[i]) - 1)
                for j in range(i+1, script_args.num_heads):
                    orthogonal_loss += torch.abs(model.score.weight[i].dot(model.score.weight[j])) # / (torch.norm(model.score.weight[i]) * torch.norm(model.score.weight[j]))

            norm_loss = (torch.abs(torch.linalg.vector_norm(model.score.weight, dim=1) - 1)).mean()
            
            corr_loss = 0
            # m = torch.stack([rewards[:, i] for i in range(script_args.num_heads)]) # data corr
            m = model.score.weight # score layer weight corr
            corr_matrix = torch.corrcoef(m)
            for i in range(script_args.num_heads):
                for j in range(i+1, script_args.num_heads):
                    corr_loss += torch.abs(corr_matrix[i, j])

            loss = BTloss + script_args.orthogonal_loss_weight * orthogonal_loss \
                    + script_args.norm_loss_weight * norm_loss + script_args.corr_loss_weight * corr_loss \
                    + script_args.load_balance_loss_weight * load_balance_loss
            
            # if accelerator.is_main_process:
            #     print({'BTloss': BTloss, 'orthogonal_loss': orthogonal_loss, 'norm_loss': norm_loss,
            #             'corr_loss': corr_loss, 'load_balance_loss': load_balance_loss, 'total_loss': loss.detach().cpu().item()})
            # print('BTloss:', BTloss, 'orthogonal_loss:', orthogonal_loss, 'norm_loss:', norm_loss, 'total_loss:', loss)
        elif script_args.loss_type == 'mixture_BT':
            if script_args.use_router:
                lengths = inputs['prompt_length']
                prompt_out = torch.stack([outputs.hidden_states[-1][jidx[idx], lengths[idx], :] for idx in range(len(jidx))])
                unwrap_model = self.accelerator.unwrap_model(self.model)
                router_weights = unwrap_model.router(prompt_out.bfloat16())
                BTloss = - torch.log((nn.functional.sigmoid(rewards_j - rewards_k) * router_weights.softmax(dim=-1)).sum(dim=-1)).mean()
            else:
                router_weights = model.router
                BTloss = - torch.log((nn.functional.sigmoid(rewards_j - rewards_k) * router_weights.softmax(dim=-1)).sum(dim=-1)).mean()
            
            load_balance_loss = 0
            if script_args.load_balance_loss_weight > 0:
                load_balance_loss = router_weights.softmax(dim=-1)
                load_balance_loss = (load_balance_loss * torch.log(load_balance_loss)).sum(dim=-1).mean()

            orthogonal_loss = 0
            if script_args.orthogonal_loss_weight > 0:
                for i in range(script_args.num_heads):
                    for j in range(i+1, script_args.num_heads):
                        orthogonal_loss += torch.abs(model.score.weight[i].dot(model.score.weight[j])) # / (torch.norm(model.score.weight[i]) * torch.norm(model.score.weight[j]))

            norm_loss = 0
            if script_args.norm_loss_weight > 0:
                norm_loss = (torch.abs(torch.linalg.vector_norm(model.score.weight, dim=1) - 1)).mean()

            corr_loss = 0
            if script_args.corr_loss_weight > 0:
            # m = torch.stack([rewards[:, i] for i in range(script_args.num_heads)]) # data corr
                m = model.score.weight # score layer weight corr
                corr_matrix = torch.corrcoef(m)
                for i in range(script_args.num_heads):
                    for j in range(i+1, script_args.num_heads):
                        corr_loss += torch.abs(corr_matrix[i, j])
                
            loss = BTloss + script_args.orthogonal_loss_weight * orthogonal_loss \
                    + script_args.norm_loss_weight * norm_loss + script_args.corr_loss_weight * corr_loss \
                    + script_args.load_balance_loss_weight * load_balance_loss
            
            # if accelerator.is_main_process:
            #     print({'BTloss': BTloss, 'orthogonal_loss': orthogonal_loss, 'norm_loss': norm_loss,
            #                'corr_loss': corr_loss, 'load_balance_loss': load_balance_loss, 'total_loss': loss})
        elif script_args.loss_type == 'multi_linear':
            loss = - nn.functional.logsigmoid(rewards_j - rewards_k).sum(dim=-1).mean()
        else:
            raise NotImplementedError

        if return_outputs:
            if script_args.num_heads > 1 and script_args.use_router and script_args.loss_type in ['mixture_reward', 'mixture_BT']:
                return loss, {"rewards_j": rewards_j, "rewards_k": rewards_k, "router_weights_j": router_weights, "router_weights_k": router_weights}
            else:
                return loss, {"rewards_j": rewards_j, "rewards_k": rewards_k}
        return loss

    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        with torch.no_grad():
            loss, logits_dict = self.compute_loss(model, inputs, return_outputs=True)

        if prediction_loss_only:
            return (loss, None, None)

        loss = loss.detach()
        logits = tuple(v for k, v in logits_dict.items() if (k not in ignore_keys) and ('router' not in k))
        logits = nested_detach(logits)
        # Stack accepted against rejected, mean over logits
        # and softmax to get preferences between accepted and rejected to sum to 1
        logits = torch.stack(logits)
        B, S, H = logits.shape
        if script_args.num_heads > 1 and script_args.loss_type in ['mixture_reward', 'mixture_BT']:
            if not script_args.use_router:
                logits = (model.router.softmax(dim=-1).unsqueeze(0).unsqueeze(0).expand(B, S, -1) * logits).sum(dim=2)
            else:
                router_weights = tuple(v for k, v in logits_dict.items() if 'router' in k)
                router_weights = nested_detach(router_weights)
                router_weights = torch.stack(router_weights)
                logits = (router_weights.softmax(dim=-1) * logits).sum(dim=2)
        else:
            logits = logits.mean(dim=2)
        logits = logits.softmax(dim=0).T

        labels = torch.zeros(logits.shape[0])
        labels = self._prepare_inputs(labels)

        return loss, logits, labels

# Train the model, woohoo.
trainer = RewardTrainer_new(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    data_collator=RewardDataCollatorWithPadding(tokenizer=tokenizer, max_length=script_args.max_length),
)

print_trainable_parameters(trainer.model)
print('training')
trainer.train()

trainer.save_model(output_name)

# if accelerator.is_main_process:
#     wandb.finish()
