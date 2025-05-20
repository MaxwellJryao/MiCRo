import os
import torch
from glob import glob
from argparse import ArgumentParser
from semi_multi_reward import SemiRewardBTTrainer, PairwiseDataset
import json
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from load_data import *
from eval.criteria import ATTRIBUTES_LIST
from eval.eval_reward_bench import eval_reward_bench
from eval.plot import save_embeddings_distribution_plot, COLORS
from safetensors.torch import save_file
import logging

"""
Perform multi-objective linear regression on precomputed embeddings.
This script loads embeddings and labels, splits the data into training and validation sets,
trains Ridge regression models for each attribute across a range of regularization strengths (alphas),
selects the best alpha based on validation loss, and saves the resulting regression weights.
"""

# ---------------------------
# Argument Parsing
# ---------------------------
parser = ArgumentParser(description="Linear Probing on Precomputed Embeddings")
parser.add_argument(
    "--config_file",
    type=str,
    default='./config_BT.json',
    help="Path to config file",
)
args = parser.parse_args()

# ---------------------------
# Configuration and Setup
# ---------------------------

with open(args.config_file, 'r') as file:
    config = json.load(file)

# Extract names from paths
model_path = config['model_path']
model_name = model_path.split("/")[-1].replace('-', '_')
unlabeled_dataset_name = "hendrydong/preference_700K".split(
    "/")[-1].replace('-', '_')


def custom_collate_fn(batch):
    collated_batch = {key: torch.stack([item[key] for item in batch]) if isinstance(
        batch[0][key], torch.Tensor) else [item[key] for item in batch] for key in batch[0]}
    return collated_batch


output_dir = config.get("output_dir", "./")

logging.basicConfig(
    filename=os.path.join(output_dir, 'evaluation_log_STAGE2.txt'),
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# ---------------------------
# Loading Embeddings and Labels
# ---------------------------
# Paths for labeled and unlabeled embeddings
dataset_dir = config['dataset_dir']
helpsteer2_train_path = os.path.join(
    dataset_dir,
    f"SemiMultiRM_embeddings_{model_name}_helpsteer2_per_attribute_pairwise_train.safetensors")
helpsteer2_test_path = os.path.join(
    dataset_dir,
    f"SemiMultiRM_embeddings_{model_name}_helpsteer2_per_attribute_pairwise_test.safetensors")
hh_train_path = os.path.join(
    dataset_dir,
    f"SemiMultiRM_embeddings_{model_name}_anthropic_rlhf_hh_pairwise_train.safetensors")
hh_test_path = os.path.join(
    dataset_dir,
    f"SemiMultiRM_embeddings_{model_name}_anthropic_rlhf_hh_pairwise_test.safetensors")
ultrafeedback_train_path = os.path.join(
    dataset_dir,
    f"SemiMultiRM_embeddings_{model_name}_ultrafeedback_per_attribute_pairwise_train.safetensors")
ultrafeedback_test_path = os.path.join(
    dataset_dir,
    f"SemiMultiRM_embeddings_{model_name}_ultrafeedback_per_attribute_pairwise_test.safetensors")
rpr_train_path = os.path.join(
    dataset_dir,
    f"SemiMultiRM_embeddings_{model_name}_rpr_per_category_pairwise_add_criterion_train.safetensors")
rpr_test_path = os.path.join(
    dataset_dir,
    f"SemiMultiRM_embeddings_{model_name}_rpr_per_category_pairwise_add_criterion_test.safetensors")
unlabeled_path = os.path.join(
    dataset_dir,
    f"SemiMultiRM_embeddings_{model_name}_400K_pairwise_train.safetensors")
pseudo_labeled_dataset_path = os.path.join(
    dataset_dir,
    f"SemiMultiRM_embeddings_{model_name}_pseudo_labeled_dataset.safetensors")
reward_bench_path = os.path.join(
    './dataset', f"SemiMultiRM_embeddings_{model_name}_reward_bench.safetensors"
)
reward_bench_source_path = os.path.join(
    './dataset', f"SemiMultiRM_embeddings_{model_name}_reward_bench_source.pkl"
)
reward_bench_path_with_context = os.path.join(
    './dataset', f"SemiMultiRM_embeddings_{model_name}_reward_bench_augmented_context.safetensors"
)
reward_bench_source_path_with_context = os.path.join(
    './dataset', f"SemiMultiRM_embeddings_{model_name}_reward_bench_augmented_context_source.pkl"
)

dataset_paths = {
    'helpsteer2': {
        'train_path': helpsteer2_train_path,
        'test_path': helpsteer2_test_path,
        'num_attribute': 5},
    'rlhf-hh': {
        'train_path': hh_train_path,
        'test_path': hh_test_path,
        'num_attribute': 2},
    'ultrafeedback': {
        'train_path': ultrafeedback_train_path,
        'test_path': ultrafeedback_test_path,
        'num_attribute': 4},
    'rpr': {
        'train_path': rpr_train_path,
        'test_path': rpr_test_path,
        'num_attribute': 10},
}


# Load labeled embeddings and labels
datasets = config['datasets']
total_attributes = 0
attributes = []
for dataset in datasets:
    total_attributes += dataset_paths[dataset]['num_attribute']
    attributes.extend(ATTRIBUTES_LIST[dataset])
config["num_labels"] = total_attributes

logging.info("Loading labeled embeddings and labels...")
pairwise_train_data_dict1, pairwise_eval_data_dict, pairwise_test_data_dict = \
    load_mix_data(datasets, config, dataset_paths, total_attributes=total_attributes)

logging.info("Training Pairwise Data Size:",
             pairwise_train_data_dict1["chosen_embeddings"].shape[0])
logging.info(
    "Eval Pairwise Data Size:",
    pairwise_eval_data_dict["chosen_embeddings"].shape[0])
logging.info(
    "Test Pairwise Data Size:",
    pairwise_test_data_dict["chosen_embeddings"].shape[0])

logging.info("Loading pseudo labeled embeddings and labels...")
pairwise_train_data_dict2 = load_data(
    pseudo_labeled_dataset_path,
    keys=[
        "chosen_embeddings",
        "chosen_prompt_embeddings",
        "rejected_prompt_embeddings",
        "rejected_embeddings",
        "head_mask",
        "labels"])

pairwise_train_data_dict = {}
for key in pairwise_train_data_dict1:
    pairwise_train_data_dict[key] = torch.cat([pairwise_train_data_dict1[key], pairwise_train_data_dict2[key]])

test_pairwise_dataset = PairwiseDataset(pairwise_test_data_dict)
test_pairwise_loader = DataLoader(
    test_pairwise_dataset,
    batch_size=512,
    shuffle=False,
    collate_fn=custom_collate_fn)

# Load unlabeled embeddings (chosen and rejected)
logging.info("Loading unlabeled embeddings...")
unlabeled_data = load_data(
    unlabeled_path,
    keys=[
        "chosen_embeddings",
        "chosen_prompt_embeddings",
        "rejected_prompt_embeddings",
        "rejected_embeddings"])
unlabeled_chosen_embeddings = unlabeled_data["chosen_embeddings"]
unlabeled_rejected_embeddings = unlabeled_data["rejected_embeddings"]
unlabeled_chosen_prompt_embeddings = unlabeled_data["chosen_prompt_embeddings"]
unlabeled_rejected_prompt_embeddings = unlabeled_data["rejected_prompt_embeddings"]
unlabeled_texts = torch.stack(
    [unlabeled_chosen_embeddings, unlabeled_rejected_embeddings], dim=1)
unlabeled_prompts = torch.stack(
    [unlabeled_chosen_prompt_embeddings, unlabeled_rejected_prompt_embeddings], dim=1)
logging.info(f"Total unlabeled embeddings: {unlabeled_texts.shape[0]}")
logging.info(
    f"Total unlabeled prompt embeddings: {unlabeled_prompts.shape[0]}")

unlabeled_dict = {
    "chosen_embeddings": unlabeled_chosen_embeddings,
    "rejected_embeddings": unlabeled_rejected_embeddings,
    "chosen_prompt_embeddings": unlabeled_chosen_prompt_embeddings,
    "rejected_prompt_embeddings": unlabeled_rejected_prompt_embeddings
}

unlabel_dataset = PairwiseDataset(unlabeled_dict)
unlabel_loader = DataLoader(unlabel_dataset, batch_size=512, shuffle=False)


trainer = SemiRewardBTTrainer(
    labeled_dict=pairwise_train_data_dict,
    eval_labeled_dict=pairwise_eval_data_dict,
    config=config
)

# Set training parameters
epochs = config.get("epochs", 5)
warm_up_steps = config.get("warm_up_steps", 10)
threshold = config.get("threshold", None)
alpha = config.get("alpha", 1)
beta = config.get("beta", 1)
add_gate = config.get("add_gate", True)

thresholds = [threshold] * len(attributes)

# Train the model
trained_model = trainer.train_model(
    epochs=epochs,
    thresholds=thresholds,
    warm_up_steps=warm_up_steps,
    alpha=alpha,
    beta=beta,
    add_gate=add_gate
)


# Generate Pseudo labels
selected_index, pseudo_labels, pseudo_head_masks, thresholds, all_max_probs, all_outputs = trainer.generate_pseudo_labels(
    unlabel_loader, thresholds=thresholds, temperature=None, apply_calibration=False)

if len(selected_index)>0:
    print(torch.sum(torch.stack(pseudo_head_masks,dim=0),dim=0))

    pseudo_labeled_data_dict = {
        "chosen_embeddings": unlabeled_chosen_embeddings[selected_index, :],
        "rejected_embeddings": unlabeled_rejected_embeddings[selected_index, :],
        "chosen_prompt_embeddings": unlabeled_chosen_prompt_embeddings[selected_index, :],
        "rejected_prompt_embeddings": unlabeled_rejected_prompt_embeddings[selected_index, :],
        "head_mask": torch.stack(pseudo_head_masks, dim=0),
        "labels": torch.stack(pseudo_labels, dim=0).squeeze(-1)
    }

file_name = '_'.join(["SemiMultiRM", "embeddings", model_name, 'pseudo_labeled_dataset']) + ".safetensors"
        # Save as safetensors
save_file(pseudo_labeled_data_dict,
    os.path.join(dataset_dir, file_name)
)

logging.info(f"Saved embeddings to {dataset_dir}")

# ---------------------------
# Evaluation
# ---------------------------
output_result_dir = os.path.join(output_dir,'best_model_result_STAGE2')
os.makedirs(output_result_dir,exist_ok=True)
all_results = {'config':config}
acc, df_results = trainer.pair_eval_model(test_pairwise_loader, save_res=True,mode='multihead')
df_results.to_csv(os.path.join(output_result_dir,"df_results_test.csv"))
subset_accuracies = df_results.groupby('subset').apply(
    lambda group: (group['correct'].sum() / len(group)) * 100  # Calculate percentage accuracy
).reset_index().rename(columns={0: 'accuracy'})
logging.info(f"Accuracies:\n{acc}")
logging.info(f"Attribute Accuracies:\n{subset_accuracies}")

eval_accuracy, all_predictions, _ = trainer.eval_model(test_pairwise_loader)
logging.info(f"Final Evaluation Accuracy: {eval_accuracy * 100:.2f}%")
label_accuracies = {}
confusion_matrices = {}

all_results['test_results'] = {}
for attribute in attributes: 
    attribute_indices = df_results[df_results['subset'] == attribute].index

    if not attribute_indices.empty:
        label_predictions = all_predictions[attribute_indices, attributes.index(attribute)]
        label_true = torch.ones_like(label_predictions)  

        label_accuracy = (label_predictions == label_true).sum().item() / label_true.size(0)
        label_accuracies[f"{attribute}_accuracy"] = label_accuracy

        confusion_matrices[f"{attribute}_confusion_matrix"] = confusion_matrix(
            label_true.cpu().numpy(), label_predictions.cpu().numpy()
        )

        logging.info(f"Accuracy for {attribute}: {label_accuracy:.2f}")
        logging.info(f"Confusion Matrix for {attribute}:")
        logging.info(confusion_matrices[f"{attribute}_confusion_matrix"])
        all_results['test_results'][attribute]=label_accuracy
    else:
        print(f"No data for {attribute}")

# reward bench evaluation
# Load reward bench embeddings and labels
logging.info("Loading reward bench embeddings and labels...")
if 'rpr' not in datasets:
    reward_bench_data_dict = load_reward_bench_data(
            reward_bench_path, reward_bench_source_path
        )
else:
    reward_bench_data_dict = load_reward_bench_data(
        reward_bench_path_with_context, reward_bench_source_path_with_context
    )
reward_bench_pairwise_dataset = PairwiseDataset(reward_bench_data_dict)
reward_bench_loader = DataLoader(reward_bench_pairwise_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)


acc, df_results = trainer.pair_eval_model(reward_bench_loader, save_res=True,mode='multihead')
df_results.to_csv(os.path.join(output_result_dir,"df_results_reward_bench.csv"))
eval_reward_bench(os.path.join(output_result_dir,"df_results_reward_bench.csv"), os.path.join(output_result_dir,'reward_bench_eval.txt'))

# evaluate single head on reward bench
all_results['reward_bench'] = {}
for i in range(len(attributes)):
    attribute = attributes[i]
    weights = torch.zeros(len(attributes))
    weights[i] = 1
    acc, df_results = trainer.pair_eval_model(reward_bench_loader, save_res=True, weights = weights,mode='multihead')
    df_results.to_csv(os.path.join(output_result_dir,f"df_results_reward_bench_attributes_{i}.csv"))
    df_reward_bench_res_final = eval_reward_bench(os.path.join(output_result_dir,f"df_results_reward_bench_attributes_{i}.csv"), os.path.join(output_result_dir,f'reward_bench_eval_{i}.txt'))
    all_results['reward_bench'][attribute]={col:df_reward_bench_res_final[col].values[0] for col in ['Chat', 'Chat Hard', 'Safety', 'Reasoning']}

with open(os.path.join(output_result_dir,'all_results.pkl'),'wb') as f:
    pickle.dump(all_results,f)
    
train_chosen_embeddings=pairwise_train_data_dict['chosen_embeddings']
train_rejected_embeddings=pairwise_train_data_dict['rejected_embeddings']
train_head_masks=pairwise_train_data_dict['head_mask']

sample_indices = torch.randint(0, unlabeled_chosen_embeddings.shape[0], (50000,))
# Saving plots for each type of embeddings
save_embeddings_distribution_plot(trainer, train_chosen_embeddings,train_rejected_embeddings, "Labeled Embeddings", output_result_dir, attributes,train_head_masks=train_head_masks)
save_embeddings_distribution_plot(trainer, unlabeled_chosen_embeddings[sample_indices],unlabeled_rejected_embeddings[sample_indices], "Unlabeled Embeddings", output_result_dir, attributes)