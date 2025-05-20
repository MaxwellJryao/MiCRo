# main_script.py
import os
import torch
from glob import glob
from peft import LoraConfig, TaskType, get_peft_model
from datasets import concatenate_datasets
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from load_data import *
from eval.criteria import ATTRIBUTES_LIST
from eval.eval_reward_bench import eval_reward_bench
from eval.plot import save_embeddings_distribution_plot, COLORS
from collections import defaultdict
import logging
from dataclasses import dataclass, field, asdict
import evaluate
import numpy as np
import torch.nn as nn
from utils import *
from typing import Any, Dict, List, Optional, Union, Tuple
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    PreTrainedModel
)
import wandb
from trainers import *
from data_utils import *

# ---------------------------
# Configuration and Initialization
# ---------------------------

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
        default=1, metadata={
            "help": "The number of training epochs for the reward model."}, )
    optim: Optional[str] = field(
        default="adamw_hf",
        metadata={"help": "The optimizer to use."},
    )
    lr_scheduler_type: Optional[str] = field(
        default="cosine", metadata={
            "help": "The lr scheduler"},)
    max_length: Optional[int] = field(default=4096)
    use_lora: Optional[bool] = field(default=False)
    # base_model: Optional[str] =  field(default="mistralai/Mistral-7B-Instruct-v0.2")
    # base_model: Optional[str] = field(default='Skywork/Skywork-Reward-Llama-3.1-8B')
    # base_model: Optional[str] = field(default='Ray2333/GRM-Llama3.2-3B-rewardmodel-ft')
    base_model: Optional[str] = field(
        default='Ray2333/Gemma-2B-rewardmodel-baseline')
    # base_model: Optional[str] =  field(default="google/gemma-2b-it")
    # base_model: Optional[str] = field(default="meta-llama/Llama-2-7b-chat-hf")
    wandb_name: Optional[str] = field(default="BT",)
    log_dir: Optional[str] = field(default='./output_models')
    loss_type: Optional[str] = field(default='BT')
    use_smallset: Optional[bool] = field(default=False)
    freeze_pretrained: Optional[bool] = field(default=True)
    num_heads: Optional[int] = field(default=5)
    orthogonal_loss_weight: Optional[float] = field(default=0)
    norm_loss_weight: Optional[float] = field(default=0)
    corr_loss_weight: Optional[float] = field(default=0.5)
    load_balance_loss_weight: Optional[float] = field(default=0.5)
    use_router: Optional[bool] = field(default=False)
    category: Optional[str] = field(default=None)
    dataset_split: Optional[str] = field(default='train')
    device: Optional[str] = field(default='cuda:0')
    output_result_dir: Optional[str] = field(default='./test_results/Ray2333_GRM-Llama3.2-3B-rewardmodel-ft/samples_15')
    n_samples_per_attribute: Optional[int] = field(default=100)
    dataset_dir: Optional[str] = field(default='./dataset')
    labeling_threshold: Optional[float] = field(default=0.8)


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

log_dir = os.path.join(script_args.output_result_dir, 'logs')
os.makedirs(log_dir, exist_ok=True) 

logging.basicConfig(
    filename=os.path.join(log_dir, 'evaluation_log.txt'),
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)


def initialize_wandb(script_args):
    wandb.init(
        project='teacher_model',
        config=script_args,
        name=script_args.wandb_name
    )

training_args = TrainingArguments(
    output_dir=os.path.join(script_args.output_result_dir, 'logs'),
    learning_rate=script_args.learning_rate,
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    per_device_eval_batch_size=script_args.per_device_eval_batch_size,
    num_train_epochs=script_args.num_train_epochs,
    # weight_decay=script_args.weight_decay,
    eval_strategy="steps",
    eval_steps=20,
    save_strategy="steps",
    save_steps=20,
    save_total_limit=1,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    gradient_checkpointing=True,
    remove_unused_columns=False,
    label_names=[],
    bf16=True,
    logging_strategy="steps",
    logging_steps=10,
    warmup_ratio=0.05,
    optim=script_args.optim,
    lr_scheduler_type=script_args.lr_scheduler_type,
    run_name=script_args.wandb_name,
    # max_grad_norm=5.0,
    report_to='wandb',
    gradient_checkpointing_kwargs={"use_reentrant": False},
    ddp_find_unused_parameters=False,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
)

tokenizer = AutoTokenizer.from_pretrained(script_args.base_model)
tokenizer.model_max_length = script_args.max_length
if 'Llama' in script_args.base_model:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
else:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForSequenceClassification.from_pretrained(
    script_args.base_model,
    num_labels=1,
    torch_dtype=torch.bfloat16,
    cache_dir="/srv/local/hf",
    device_map="auto"
)
print(model.device)
# ---------------------------
# Load Datasets
# ---------------------------

datasets_name = ['helpsteer2','rpr','ultrafeedback','pku-safe'] #'ultrafeedback',
dataset_paths = {
    'helpsteer2': {
        'dataset_path': os.path.join(
            script_args.dataset_dir,
            'helpsteer2_per_attribute_pairwise'),
        'num_attribute': 5},
    'rlhf-hh': {
        'dataset_path': os.path.join(
            script_args.dataset_dir,
            'anthropic_rlhf_hh_pairwise'),
        'num_attribute': 2},
    'ultrafeedback': {
        'dataset_path': os.path.join(
            script_args.dataset_dir,
            'ultrafeedback_per_attribute_pairwise'),
        'num_attribute': 4},
    'rpr': {
        'dataset_path': os.path.join(
            script_args.dataset_dir,
            'rpr_per_category_pairwise_add_criterion'),
        'num_attribute': 10},
    'pku-safe': {
        'dataset_path': os.path.join(
            script_args.dataset_dir,
            'pku_alignment_safe_pairwise'),
        'num_attribute': 1},
    }
unlabeled_dataset_name = ["400K"]
unlabeled_dataset_paths = {
    "400K":{'dataset_path': os.path.join(script_args.dataset_dir,"400K_pairwise")},
}

total_attributes = 0
attributes = []
for dataset_name in datasets_name:
    total_attributes += dataset_paths[dataset_name]['num_attribute']
    attributes.extend(ATTRIBUTES_LIST[dataset_name])
    
attri_index = 0
ds_train_all = []
for dataset_name in tqdm(datasets_name,desc='Loading training datasts...'):
    ds = load_from_disk(dataset_paths[dataset_name]['dataset_path'])['train']
    ds = build_dataset(ds, tokenizer, script_args)
    if dataset_name == 'rpr':
        ds = process_train_dataset(ds, script_args, attri_index, total_attributes, attributes = ATTRIBUTES_LIST['rpr'])
    else:
        ds = process_train_dataset(ds, script_args, attri_index, total_attributes)
    attri_index += dataset_paths[dataset_name]['num_attribute']
    ds_train_all.append(ds)
train_dataset = concatenate_datasets(ds_train_all)

eval_ratio = 0.1
train_dataset = train_dataset.shuffle(seed=42)
eval_size = int(len(train_dataset) * eval_ratio)
eval_dataset = train_dataset.select(range(eval_size))
train_dataset = train_dataset.select(range(eval_size, len(train_dataset)))
print('Number of training data:', len(train_dataset))
print('Number of validation data:', len(eval_dataset))

# ds_test_attributes, ds_test_all = [],[]
# attri_index = 0
# for dataset_name in tqdm(datasets_name,desc='Loading test datasts...'):
#     print(dataset_name)
#     ds = load_from_disk(dataset_paths[dataset_name]['dataset_path'])['test']
#     ds = build_dataset(ds, tokenizer, script_args)
#     if dataset_name == 'ultrafeedback':
#         num_samples = 2000
#         if len(ds) > num_samples:
#             sampled_indices = random.sample(range(len(ds)), num_samples)
#             ds = ds.select(sampled_indices)
#     if dataset_name == 'rpr':
#         ds, test_attributes = process_test_dataset(ds, script_args, attri_index, total_attributes, attributes = ATTRIBUTES_LIST['rpr'])
#     else:
#         ds, test_attributes = process_test_dataset(ds, script_args, attri_index, total_attributes)
#     attri_index += dataset_paths[dataset_name]['num_attribute']
#     ds_test_all.append(ds)
#     ds_test_attributes.extend(test_attributes.tolist())
# test_dataset = concatenate_datasets(ds_test_all)
# test_df = pd.DataFrame(test_dataset)
# assert len(ds_test_attributes) == len(test_df)

ds_unlabeled_all = []
for dataset_name in unlabeled_dataset_name:
    ds = load_from_disk(unlabeled_dataset_paths[dataset_name]['dataset_path'])['train']
    ds = build_dataset(ds, tokenizer, script_args)
    ds.set_format(type="torch")
    ds_unlabeled_all.append(ds)
unlabeled_dataset = concatenate_datasets(ds_unlabeled_all)

peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    inference_mode=False,
    r=64,
    lora_alpha=128,
    lora_dropout=0.05,
)


# init_score_weight = model.score.weight.clone()
# init_router_weight = model.router.weight.clone()

# Define the metric that we'll use for validation.
accuracy = evaluate.load('accuracy')


def compute_metrics(eval_pred):
    predictions = eval_pred.predictions
    predictions = np.argmax(predictions, axis=1)
    labels = np.zeros(predictions.shape)
    return accuracy.compute(predictions=predictions, references=labels)


# Train the model
initialize_wandb(script_args)
print('Start Training:')
if script_args.freeze_pretrained or not script_args.use_lora:
    trainer = MultiRewardTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=RewardDataCollatorWithPadding(
            tokenizer=tokenizer,
            max_length=script_args.max_length),
        script_args=script_args 
    )
else:
    trainer = MultiRewardTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=RewardDataCollatorWithPadding(
            tokenizer=tokenizer,
            max_length=script_args.max_length),
        script_args=script_args,
        peft_config=peft_config,
    )

print_trainable_parameters(trainer.model)
# trainer.train()

# logging.info("Best checkpoint path:", trainer.state.best_model_checkpoint)
# logging.info("Best evaluation metric:", trainer.state.best_metric)

# ---------------------------
# Evaluation
# ---------------------------
checkpoint_path = os.path.join(script_args.output_result_dir, 'logs','checkpoint-20')
model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint_path,
    num_labels=total_attributes,
    torch_dtype=torch.bfloat16,
    cache_dir="/srv/local/hf",
    device_map="auto"

)

# all_results = {'config':asdict(script_args)}
# r_chosen, r_rejected = inference_rewards(trainer, test_dataset)
# num_heads = r_chosen.size(1)
# attribute_accuracy = {attr: [] for attr in attributes}
# test_df = pd.DataFrame(test_dataset)
# assert len(ds_test_attributes) == len(test_df)

# for head_idx in range(num_heads):
#     correct_count = defaultdict(int)
#     total_count = defaultdict(int)
    
#     margin = r_chosen[:, head_idx] - r_rejected[:, head_idx]
#     scores = torch.sigmoid(margin)
#     predictions = (scores > 0.5).long()
    
#     for idx, row in test_df.iterrows():
#         attr = ds_test_attributes[idx]
#         chosen_score = scores[idx].item()
#         rejected_score = 1 - chosen_score
#         correct = chosen_score > 0.5
#         correct_count[attr] += int(correct)
#         total_count[attr] += 1
    
#     for attr in attributes:
#         if total_count[attr] > 0:
#             accuracy = correct_count[attr] / total_count[attr]
#             attribute_accuracy[attr].append(accuracy)
#         else:
#             attribute_accuracy[attr].append(0.0)

# for attr in attributes:
#     logging.info(f"Attribute: {attr}")
#     for head_idx, acc in enumerate(attribute_accuracy[attr]):
#         logging.info(f"  Head {head_idx}: {acc:.4f}")


# all_results['test_results'] = attribute_accuracy
# with open(os.path.join(script_args.output_result_dir,'all_results.pkl'),'wb') as f:
#     pickle.dump(all_results,f)

# ---------------------------
# Generate pseudo labels
# ---------------------------


r_j, r_k = inference_rewards(trainer, eval_dataset)
temperature = torch.std(torch.cat([r_j,r_k],dim=0))
print(temperature)

thresholds = [script_args.labeling_threshold]*total_attributes
all_rewards,selected_index, pseudo_labels,pseudo_head_masks = generate_pseudo_labels(trainer, unlabeled_dataset,thresholds, temperature)

with open(os.path.join(script_args.output_result_dir,'rewards_scores.pkl'),'wb') as f:
    pickle.dump(all_rewards,f)
    
logging.info(torch.sum(pseudo_head_masks,dim=0))
pseudo_labeled_data_dict = {
        "selected_index": selected_index,
        "head_masks": pseudo_head_masks,
        "labels": pseudo_labels
    }

def add_labels_to_dataset(unlabeled_dataset, pseudo_labeled_data_dict):
    selected_index = pseudo_labeled_data_dict["selected_index"]
    pseudo_labels = pseudo_labeled_data_dict["labels"].tolist()
    head_masks = pseudo_labeled_data_dict["head_masks"].tolist()

    selected_samples = unlabeled_dataset.select(selected_index)
    selected_samples = selected_samples.add_column("head_masks", head_masks)
    selected_samples = selected_samples.add_column("labels", pseudo_labels)

    return selected_samples

pseudo_labeled_dataset = add_labels_to_dataset(unlabeled_dataset, pseudo_labeled_data_dict)
pseudo_labeled_dataset.save_to_disk(os.path.join(script_args.output_result_dir,'pseudo_labeled_dataset'))

# ---------------------------
# Evaluation Reward Bench
# ---------------------------
# reward bench evaluation
# Load reward bench embeddings and labels
logging.info("Loading reward bench embeddings and labels...")

# ds_reward_bench = []
# ds_reward_bench = load_from_disk('reward_bench_pairwise')['train']
# reward_bench_dataset = build_dataset(ds, tokenizer, script_args)
# reward_bench_dataset.set_format(type="torch")
# df_reward_bench = pd.DataFrame(ds_reward_bench)

# r_chosen, r_rejected = inference_rewards(trainer, reward_bench_dataset)

# # evaluate single head on reward bench
# all_results['reward_bench'] = {}
# df_results = pd.DataFrame(columns=['id', 'subset', 'correct'])
# for head_idx in range(num_heads):
#     correct_count = defaultdict(int)
#     total_count = defaultdict(int)
    
#     margin = r_chosen[:, head_idx] - r_rejected[:, head_idx]
#     scores = torch.sigmoid(margin)
#     predictions = (scores > 0.5).long()
    
#     for idx, row in df_reward_bench.iterrows():
#         chosen_score = scores[idx].item()
#         rejected_score = 1 - chosen_score
#         correct = 0.5 if chosen_score == rejected_score else float(
#                             chosen_score > rejected_score)
#         df_results._append({
#                             'id': idx,
#                             'correct': correct,
#                             'subset': row["subset"]
#                         }, ignore_index=True)
    
#     df_results.to_csv(os.path.join(script_args.output_result_dir,f"df_results_reward_bench_{head_idx}.csv"))
#     df_reward_bench_res_final = eval_reward_bench(os.path.join(script_args.output_result_dir,f"df_results_reward_bench_attributes_{head_index}.csv"), os.path.join(script_args.output_result_dir,f'reward_bench_eval_{head_index}.txt'))
#     all_results['reward_bench'][attributes[head_idx]]={col:df_reward_bench_res_final[col].values[0] for col in ['Chat', 'Chat Hard', 'Safety', 'Reasoning']}
            
# with open(os.path.join(script_args.output_result_dir,'all_results.pkl'),'wb') as f:
#     pickle.dump(all_results,f)
# wandb.finish()
# train_chosen_embeddings=pairwise_train_data_dict['chosen_embeddings']
# train_rejected_embeddings=pairwise_train_data_dict['rejected_embeddings']
# train_head_masks=pairwise_train_data_dict['head_mask']

# sample_indices = torch.randint(0, unlabeled_chosen_embeddings.shape[0], (50000,))
# # Saving plots for each type of embeddings
# save_embeddings_distribution_plot(trainer, train_chosen_embeddings,train_rejected_embeddings, "Labeled Embeddings", output_result_dir, attributes,train_head_masks=train_head_masks)
# save_embeddings_distribution_plot(trainer, unlabeled_chosen_embeddings[sample_indices],unlabeled_rejected_embeddings[sample_indices], "Unlabeled Embeddings", output_result_dir, attributes)