import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import numpy as np
from torch.utils.data import random_split
import random
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, Subset
import wandb
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import pandas as pd
import os
from collections import defaultdict
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt


def initialize_wandb(config):
    wandb.init(
        project=config.get('project_name', 'mix_reward_model'),
        config=config,
        name=config.get('run_name', 'default_run_name')
    )


class GatingNetwork(nn.Module):
    """
    Gating Network: A simple MLP with softmax output and temperature scaling
    This network learns to combine multiple reward objectives based on the input context
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        temperature: float = 100,
        logit_scale: float = 1.0,
        hidden_dim: int = 1024,
        n_hidden: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.temperature = temperature
        # self.logit_scale = nn.Parameter(torch.ones(1) * logit_scale)
        self.dropout_prob = dropout
        layers = []
        for _ in range(n_hidden):
            layers.append(
                nn.Linear(
                    in_features,
                    hidden_dim,
                    dtype=torch.bfloat16))
            in_features = hidden_dim
        layers.append(
            nn.Linear(
                in_features,
                out_features,
                bias=bias,
                dtype=torch.bfloat16))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        # Apply the linear layers with ReLU and dropout
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
                if self.dropout_prob > 0:
                    x = F.dropout(
                        x, p=self.dropout_prob, training=self.training)
        # Apply softmax with temperature scaling
        x = F.softmax(x / self.temperature, dim=1)
        return x  # * self.logit_scale[0]


class PairwiseDataset(Dataset):
    def __init__(self, data_dict):
        self.data_dict = data_dict

    def __len__(self):
        key = list(self.data_dict.keys())[0]
        return len(self.data_dict[key])

    def __getitem__(self, idx):
        return {key: self.data_dict[key][idx] for key in self.data_dict}


class MultiLabelRewardModel(nn.Module):
    def __init__(
            self,
            input_size=768,
            hidden_size=512,
            num_labels=5,
            num_class=5,
            num_layers=1):
        super(MultiLabelRewardModel, self).__init__()
        self.num_labels = num_labels
        self.num_class = num_class
        self.num_layers = num_layers
        self.layer_norm = nn.LayerNorm(input_size)

        if num_layers == 1:
            self.layers = nn.Sequential(
                # nn.utils.weight_norm(nn.Linear(input_size, num_labels * num_class, dtype=torch.bfloat16, bias=False))
                nn.Linear(
                    input_size,
                    num_labels * num_class,
                    dtype=torch.bfloat16,
                    bias=False)
            )
        elif num_layers > 1:
            layers = [
                nn.Sequential(
                    nn.Linear(
                        input_size,
                        hidden_size,
                        dtype=torch.bfloat16,
                        bias=True),
                    nn.ReLU())]
            layers += [
                nn.Sequential(
                    nn.Linear(
                        hidden_size,
                        hidden_size,
                        dtype=torch.bfloat16,
                        bias=True),
                    nn.ReLU()) for _ in range(
                    num_layers - 2)]
            layers.append(
                nn.Linear(
                    hidden_size,
                    num_labels *
                    num_class,
                    dtype=torch.bfloat16,
                    bias=False))
            self.layers = nn.Sequential(*layers)
        else:
            raise ValueError("The number of layers must be at least 1.")
        # self.batch_norms = nn.ModuleList([nn.BatchNorm1d(num_class) for _ in range(num_labels)])

    def forward(self, x):
        # x = self.layer_norm(x)
        x = self.layers(x)
        x = x.view(-1, self.num_labels, self.num_class)
        return x


class SemiRewardTrainer():
    def __init__(self, config, labeled_dict, eval_labeled_dict=None):
        # Initialize labeled and unlabeled datasets
        self.set_seed(config.get('seed', 42))
        self.config = config
        self.config['use_wandb'] = self.config.get('use_wandb', True)

        # Initialize configurations
        for key, value in self.config.items():
            setattr(self, key, value)

        # Prepare datasets
        train_dataset = PairwiseDataset(labeled_dict)
        if eval_labeled_dict:
            eval_dataset = PairwiseDataset(eval_labeled_dict)
        else:
            train_ratio = config.get('train_ratio', 0.9)
            train_size = int(train_ratio * len(train_dataset))
            eval_size = len(train_dataset) - train_size
            train_dataset, eval_dataset = random_split(
                train_dataset, [train_size, eval_size])

        # Create DataLoaders for training, evaluation, and unlabeled data
        self.labeled_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True)
        self.eval_loader = DataLoader(
            eval_dataset,
            batch_size=self.batch_size,
            shuffle=False)

        self.model = MultiLabelRewardModel(
            input_size=config['input_size'],
            num_labels=self.num_labels,
            num_class=self.num_class,
            num_layers=self.num_layers).to(
            self.device)
        self.model = self.model.to(dtype=torch.bfloat16)
        parameters = list(self.model.parameters())
        self.optimizer = optim.AdamW(
            parameters,
            lr=self.learning_rate,
            weight_decay=1e-2)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=30)
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        if self.use_wandb:
            initialize_wandb(config)

    def eval_model(self, eval_loader, temperature=1):
        self.model.eval()
        all_predictions = []
        all_head_masks = []
        all_probabilities = []
        total_correct = 0
        total_samples = 0

        # Loop through the evaluation data
        with torch.no_grad():
            for batch_data in eval_loader:

                texts = batch_data['embeddings'].to(self.device)
                head_mask = batch_data['head_mask'].to(self.device)
                # Shape: (batch_size, num_labels, num_class)
                outputs = torch.sigmoid(self.model(texts) / temperature)

                labels = torch.ones_like(outputs).to(self.device)
                prob_outputs = torch.cat([1 - outputs, outputs], dim=2)
                # Shape: (batch_size, num_labels)
                predicted = torch.argmax(prob_outputs, dim=2)

                all_predictions.append(predicted.cpu())
                all_probabilities.append(outputs.cpu())

                if head_mask is not None:
                    total_correct += ((predicted == labels.squeeze(-1)
                                       ).float() * head_mask).sum().item()
                    total_samples += head_mask.squeeze(-1).sum().item()
                    all_head_masks.append(head_mask.cpu())
                else:
                    total_correct += (predicted ==
                                      labels.squeeze(-1)).sum().item()
                    total_samples += labels.numel()

        all_predictions = torch.cat(all_predictions, dim=0)
        all_head_masks = torch.cat(all_head_masks, dim=0)
        all_probabilities = torch.cat(all_probabilities, dim=0)

        overall_accuracy = total_correct / total_samples if total_samples > 0 else 0
        print(f"Overall Evaluation Accuracy: {overall_accuracy * 100:.2f}%")
        try:
            wandb.log({"evaluation_accuracy": overall_accuracy})
        except BaseException:
            pass

        try:
            label_accuracies = {}
            for i in range(self.num_labels):
                if len(all_head_masks) > 0:
                    tmp_indices = (all_head_masks[:, i] == 1).nonzero(
                        as_tuple=True)[0]
                else:
                    tmp_indices = torch.arange(all_predictions.shape[0])
                    label_true = torch.ones_like(
                        all_predictions[:, i], dtype=torch.long)
                label_predictions = all_predictions[tmp_indices, i]
                label_true = torch.ones_like(label_predictions)
                probabilities = all_probabilities[tmp_indices, i]

                plt.figure()
                plt.hist(probabilities.float().numpy(), bins=20,
                         range=(0, 1), alpha=0.75, color='blue')
                plt.title("Distribution of Predicted Probabilities")
                plt.xlabel("Predicted Probability")
                plt.ylabel("Frequency")
                plt.grid()
                wandb.log({f"head_{i}_probability_histogram": wandb.Image(plt)})
                plt.close()

                num_samples = label_predictions.size(0)
                flip_indices = torch.randperm(num_samples)[:num_samples // 3]

                label_true[flip_indices] = 1 - label_true[flip_indices]
                label_predictions[flip_indices] = 1 - \
                    label_predictions[flip_indices]
                probabilities[flip_indices] = 1 - probabilities[flip_indices]

                label_accuracy = (label_predictions == label_true).sum(
                ).item() / label_predictions.size(0)
                label_accuracies[f"sample_{i}_accuracy"] = label_accuracy

                try:
                    wandb.log({f"label_{i}_accuracy": label_accuracy})
                except BaseException:
                    pass

                prob_true, prob_pred = calibration_curve(
                    label_true.numpy(), probabilities.float().numpy(), n_bins=10)

                # Plot calibration curve
                plt.figure()
                plt.plot(
                    prob_pred,
                    prob_true,
                    marker='o',
                    label="Calibration Curve")
                plt.plot([0, 1], [0, 1], linestyle="--",
                         label="Perfectly Calibrated")
                plt.xlabel("Mean Predicted Probability")
                plt.ylabel("Fraction of Positives")
                plt.title(f"Calibration Curve - Head {i}")
                plt.legend()
                plt.grid()

                # Log calibration curve to WandB
                try:
                    wandb.log(
                        {f"head_{i}_calibration_curve": wandb.Image(plt)})
                except BaseException:
                    pass
                plt.close()

        except BaseException:
            pass

        return overall_accuracy, all_predictions

    def generate_pseudo_labels(
            self,
            thresholds=None,
            temperature=0.8,
            apply_calibration=False):
        pseudo_labels = []
        selected_index = []
        pseudo_head_masks = []  # To store the generated head masks for each pseudo-label
        self.model.eval()
        all_max_probs = []

        with torch.no_grad():
            index = 0
            for texts, _, _ in tqdm(
                    self.unlabeled_loader, desc="Generating pseudo labels"):
                text = texts[:, 0, :].to(self.device)
                outputs = self.model(text)

                probs = torch.sigmoid(outputs / temperature)

                # If no thresholds are provided, default to 0.5 for each label
                if thresholds is None:
                    thresholds = torch.tensor(
                        [[0.5] * probs.shape[2]] * probs.shape[1]).to(self.device)

                all_max_probs.append(
                    torch.max(torch.cat([probs, 1 - probs], dim=2), dim=2)[0])
                thresholds = torch.tensor(thresholds).view(
                    1, -1, 1).to(probs.device)

                valid_mask = (
                    (probs > thresholds) | (
                        (1 - probs) > thresholds)).float()
                valid_label_set = (probs > thresholds).float() * valid_mask
                valid_indices = torch.where(
                    valid_mask.any(dim=(1, 2)))[0].cpu()

                if valid_indices.any():
                    pseudo_labels.extend(
                        valid_label_set[valid_indices].detach().cpu())
                    selected_index.extend(
                        (torch.arange(
                            probs.shape[0])[valid_indices] +
                            index).tolist())
                    pseudo_head_masks.extend(
                        valid_mask[valid_indices].detach().cpu())

                index += len(texts)

                # threshold update
                # thresholds = 0.9 * thresholds[0,:,0] + (1-0.9)*torch.cat(all_max_probs,dim=0).mean(dim=0)

        return selected_index, pseudo_labels, pseudo_head_masks, thresholds

    def train_model(
            self,
            epochs=5,
            warm_up_steps=1,
            thresholds=None,
            alpha=0.1,
            beta=0.1,
            add_gate=True):
        best_acc = 0  # Initialize best accuracy for model saving
        step = 0
        self.combined_labeled_loader = self.labeled_loader

        for epoch in range(epochs):
            epoch_loss = 0

            for batch_idx, batch_data in enumerate(
                    tqdm(self.combined_labeled_loader, desc=f"Training Epoch {epoch + 1}")):
                self.model.train()
                self.optimizer.zero_grad()

                text = batch_data['embeddings'].to(self.device)
                head_mask = batch_data['head_mask'].to(
                    self.device)  # torch.Size([2048, 5, 1])
                # torch.Size([2048, 5])
                labels = batch_data['labels'].squeeze(-1).to(self.device)

                outputs = self.model(text)
                criterion = nn.BCEWithLogitsLoss(reduction='none')
                loss = criterion(
                    outputs.view(
                        (outputs.shape[0] * self.num_labels,
                         self.num_class)),
                    labels.view(
                        -1,
                        1))  # Shape: (batch_size, num_heads, 1)

                head_mask_flatten = head_mask.view(
                    (outputs.shape[0] * self.num_labels, self.num_class))
                if self.auxilary_loss:
                    text_flatten = outputs.view(
                        outputs.shape[0] * self.num_labels, self.num_class)
                    loss = (loss + 0.01 * text_flatten**2) * head_mask_flatten
                else:
                    loss = loss * head_mask_flatten

                loss = loss.sum() / head_mask.sum()
                if head_mask.sum() == 0:
                    loss = torch.tensor(0.0, device=self.device)

                predicted_labels = (outputs.sigmoid() > 0.5).float()
                correct_predictions = (
                    predicted_labels == labels.unsqueeze(-1)).float() * head_mask
                label_accuracy_total = correct_predictions.sum(
                ) / head_mask.sum() if head_mask.sum() > 0 else 0

                wandb.log({
                    "batch_total_loss": loss.item(),
                })

                # Backpropagation
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            # Logging for each epoch
            avg_epoch_loss = epoch_loss / len(self.labeled_loader)
            self.scheduler.step()
            step += 1

            print(
                f"Epoch [{epoch + 1}/{epochs}], Total Loss: {avg_epoch_loss:.4f}, ")

            wandb.log({
                "epoch": epoch + 1,
                "avg_epoch_loss": avg_epoch_loss,
            })

            # Evaluate model and save if the accuracy improves
            acc, _ = self.eval_model(self.eval_loader)
            if self.config.get("save_model", False):
                if best_acc < acc:
                    model_save_path = os.path.join(
                        self.config.get(
                            "output_dir",
                            "./"),
                        self.config.get(
                            "model_save_path",
                            "") +
                        "best_trained_model.pth")
                    torch.save(self.model.state_dict(), model_save_path)
                    print(f"Model saved to {model_save_path}")
                    best_acc = acc

        if self.config.get("save_model", False):
            model_save_path = os.path.join(
                self.config.get(
                    "output_dir",
                    "./"),
                self.config.get(
                    "model_save_path",
                    "") +
                "final_trained_model.pth")
            torch.save(self.model.state_dict(), model_save_path)
            print(f"Final model saved to {model_save_path}")

        return self.model

    @staticmethod
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def load_model(self, model_path):
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded successfully from {model_path}")


class SemiRewardBTTrainer():
    def __init__(self, config, labeled_dict, eval_labeled_dict=None):
        # Initialize labeled and unlabeled datasets
        self.set_seed(config.get('seed', 42))
        self.config = config
        self.config['use_wandb'] = self.config.get('use_wandb', True)

        # Initialize configurations
        for key, value in self.config.items():
            setattr(self, key, value)

        # Prepare datasets
        train_dataset = PairwiseDataset(labeled_dict)
        if eval_labeled_dict:
            eval_dataset = PairwiseDataset(eval_labeled_dict)
        else:
            train_ratio = config.get('train_ratio', 0.9)
            train_size = int(train_ratio * len(train_dataset))
            eval_size = len(train_dataset) - train_size
            train_dataset, eval_dataset = random_split(
                train_dataset, [train_size, eval_size])

        # Create DataLoaders for training, evaluation, and unlabeled data
        self.labeled_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True)
        self.eval_loader = DataLoader(
            eval_dataset,
            batch_size=self.batch_size,
            shuffle=False)

        self.model = MultiLabelRewardModel(
            input_size=config['input_size'],
            num_labels=self.num_labels,
            num_class=self.num_class,
            num_layers=self.num_layers).to(
            self.device)
        self.model = self.model.to(dtype=torch.bfloat16)
        parameters = list(self.model.parameters())
        self.optimizer = optim.AdamW(
            parameters,
            lr=self.learning_rate,
            weight_decay=1e-2)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=30)
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        self.rewards_std=1.0
        if self.use_wandb:
            initialize_wandb(config)

    def eval_model(self, eval_loader, temperature=1):
        self.model.eval()
        all_predictions = []
        all_head_masks = []
        all_probabilities = []
        all_outputs = []
        total_correct = 0
        total_samples = 0

        # Loop through the evaluation data
        with torch.no_grad():
            for batch_data in eval_loader:
                texts1 = batch_data['chosen_embeddings'].to(self.device)
                texts2 = batch_data['rejected_embeddings'].to(self.device)
                head_mask = batch_data['head_mask'].to(self.device)
                # Shape: (batch_size, num_labels, num_class)
                outputs = torch.sigmoid(
                    (self.model(texts1) - self.model(texts2)))
                labels = torch.ones_like(outputs).to(self.device)
                prob_outputs = torch.cat([1 - outputs, outputs], dim=2)
                # Shape: (batch_size, num_labels)
                predicted = torch.argmax(prob_outputs, dim=2)

                all_predictions.append(predicted.cpu())
                all_outputs.append(self.model(texts1).cpu())
                all_outputs.append(self.model(texts2).cpu())
                all_probabilities.append(
                    torch.sigmoid(
                        (self.model(texts1) -
                         self.model(texts2)) /
                        temperature).cpu())

                if head_mask is not None:
                    total_correct += ((predicted == labels.squeeze(-1)
                                       ).float() * head_mask).sum().item()
                    total_samples += head_mask.squeeze(-1).sum().item()
                    all_head_masks.append(head_mask.cpu())
                else:
                    total_correct += (predicted ==
                                      labels.squeeze(-1)).sum().item()
                    total_samples += labels.numel()

        all_predictions = torch.cat(all_predictions, dim=0)
        all_head_masks = torch.cat(all_head_masks, dim=0)
        all_probabilities = torch.cat(all_probabilities, dim=0)
        all_outputs = torch.cat(all_outputs,dim=0)
        rewards_std = torch.std(all_outputs,dim=0)

        overall_accuracy = total_correct / total_samples if total_samples > 0 else 0
        print(f"Overall Evaluation Accuracy: {overall_accuracy * 100:.2f}%")
        try:
            wandb.log({"evaluation_accuracy": overall_accuracy})
        except BaseException:
            pass

        try:
            label_accuracies = {}
            for i in range(self.num_labels):
                if len(all_head_masks) > 0:
                    tmp_indices = (all_head_masks[:, i] == 1).nonzero(
                        as_tuple=True)[0]
                else:
                    tmp_indices = torch.arange(all_predictions.shape[0])
                    label_true = torch.ones_like(
                        all_predictions[:, i], dtype=torch.long)
                label_predictions = all_predictions[tmp_indices, i]
                label_true = torch.ones_like(label_predictions)
                probabilities = all_probabilities[tmp_indices, i]

                plt.figure()
                plt.hist(probabilities.float().numpy(), bins=20,
                         range=(0, 1), alpha=0.75, color='blue')
                plt.title("Distribution of Predicted Probabilities")
                plt.xlabel("Predicted Probability")
                plt.ylabel("Frequency")
                plt.grid()
                wandb.log({f"head_{i}_probability_histogram": wandb.Image(plt)})
                plt.close()

                num_samples = label_predictions.size(0)
                flip_indices = torch.randperm(num_samples)[:num_samples // 3]

                label_true[flip_indices] = 1 - label_true[flip_indices]
                label_predictions[flip_indices] = 1 - \
                    label_predictions[flip_indices]
                probabilities[flip_indices] = 1 - probabilities[flip_indices]

                label_accuracy = (label_predictions == label_true).sum(
                ).item() / label_predictions.size(0)
                label_accuracies[f"sample_{i}_accuracy"] = label_accuracy

                try:
                    wandb.log({f"label_{i}_accuracy": label_accuracy})
                except BaseException:
                    pass

                prob_true, prob_pred = calibration_curve(
                    label_true.numpy(), probabilities.float().numpy(), n_bins=10)

                # Plot calibration curve
                plt.figure()
                plt.plot(
                    prob_pred,
                    prob_true,
                    marker='o',
                    label="Calibration Curve")
                plt.plot([0, 1], [0, 1], linestyle="--",
                         label="Perfectly Calibrated")
                plt.xlabel("Mean Predicted Probability")
                plt.ylabel("Fraction of Positives")
                plt.title(f"Calibration Curve - Head {i}")
                plt.legend()
                plt.grid()

                # Log calibration curve to WandB
                try:
                    wandb.log(
                        {f"head_{i}_calibration_curve": wandb.Image(plt)})
                except BaseException:
                    pass
                plt.close()

        except BaseException:
            pass

        return overall_accuracy, all_predictions, rewards_std

    def generate_pseudo_labels(
            self,
            unlabeled_loader,
            thresholds=None,
            temperature=1.0,
            apply_calibration=False):
        pseudo_labels = []
        selected_index = []
        pseudo_head_masks = []  # To store the generated head masks for each pseudo-label
        self.model.eval()
        all_max_probs = []
        all_outputs = []
        if not temperature:
            temperature = self.rewards_std.to(self.device)

        with torch.no_grad():
            index = 0
            for batch_data in tqdm(
                    unlabeled_loader, desc="Generating pseudo labels"):
                texts1 = batch_data['chosen_embeddings'].to(self.device)
                texts2 = batch_data['rejected_embeddings'].to(self.device)
                outputs1 = self.model(texts1)
                outputs2 = self.model(texts2)
                outputs = outputs1 - outputs2

                probs = torch.sigmoid(outputs / temperature)

                # If no thresholds are provided, default to 0.5 for each label
                if thresholds is None:
                    thresholds = torch.tensor(
                        [[0.5] * probs.shape[2]] * probs.shape[1]).to(self.device)

                all_max_probs.append(
                    torch.max(torch.cat([probs, 1 - probs], dim=2), dim=2)[0])
                all_outputs.append(outputs)
                thresholds = torch.tensor(thresholds).view(
                    1, -1, 1).to(probs.device)

                valid_mask = (
                    (probs > thresholds) | (
                        (1 - probs) > thresholds)).float()
                valid_label_set = (probs > thresholds).float() * valid_mask

                valid_mask_reduced = valid_mask.any(dim=2).any(dim=1)
                valid_indices = torch.where(valid_mask_reduced)[0].cpu()

                if valid_indices.any():
                    pseudo_labels.extend(
                        valid_label_set[valid_indices].detach().cpu())
                    selected_index.extend(
                        (torch.arange(
                            probs.shape[0])[valid_indices] +
                            index).tolist())
                    pseudo_head_masks.extend(
                        valid_mask[valid_indices].detach().cpu())

                index += len(texts1)

                # threshold update
                # thresholds = 0.9 * thresholds[0,:,0] + (1-0.9)*torch.cat(all_max_probs,dim=0).mean(dim=0)

        return selected_index, pseudo_labels, pseudo_head_masks, thresholds, all_max_probs, all_outputs

    def train_model(
            self,
            epochs=5,
            warm_up_steps=1,
            thresholds=None,
            alpha=0.1,
            beta=0.1,
            add_gate=True):
        best_acc = 0  # Initialize best accuracy for model saving
        step = 0
        self.combined_labeled_loader = self.labeled_loader

        for epoch in range(epochs):
            epoch_loss = 0
            labeled_loss_total = 0

            for batch_idx, batch_data in enumerate(
                    tqdm(self.combined_labeled_loader, desc=f"Training Epoch {epoch + 1}")):
                self.model.train()
                self.optimizer.zero_grad()

                text1 = batch_data['chosen_embeddings'].to(self.device)
                text2 = batch_data['rejected_embeddings'].to(self.device)
                head_mask = batch_data['head_mask'].to(
                    self.device)  # torch.Size([2048, 5, 1])
                # torch.Size([2048, 5])
                labels = batch_data['labels'].squeeze(-1).to(self.device)

                s_A = self.model(text1)
                s_B = self.model(text2)

                outputs = s_A - s_B
                criterion = nn.BCEWithLogitsLoss(reduction='none')
                loss = criterion(
                    outputs.view(
                        (outputs.shape[0] * self.num_labels,
                         self.num_class)),
                    labels.view(
                        -1,
                        1))  # Shape: (batch_size, num_heads, 1)

                head_mask_flatten = head_mask.view(
                    (outputs.shape[0] * self.num_labels, self.num_class))
                if self.auxilary_loss:
                    text1_flatten = s_A.view(
                        outputs.shape[0] * self.num_labels, self.num_class)
                    text2_flatten = s_B.view(
                        outputs.shape[0] * self.num_labels, self.num_class)
                    loss = (loss + 0.01 * (text1_flatten**2 +
                            text2_flatten**2)) * head_mask_flatten
                else:
                    loss = loss * head_mask_flatten

                loss = loss.sum() / head_mask.sum()
                if head_mask.sum() == 0:
                    loss = torch.tensor(0.0, device=self.device)

                predicted_labels = (outputs.sigmoid() > 0.5).float()
                correct_predictions = (
                    predicted_labels == labels.unsqueeze(-1)).float() * head_mask
                label_accuracy_total = correct_predictions.sum(
                ) / head_mask.sum() if head_mask.sum() > 0 else 0

                wandb.log({
                    "batch_total_loss": loss.item(),
                })

                # Backpropagation
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            # Logging for each epoch
            avg_epoch_loss = epoch_loss / len(self.labeled_loader)
            avg_labeled_loss = labeled_loss_total / len(self.labeled_loader)
            self.scheduler.step()
            step += 1

            print(
                f"Epoch [{epoch + 1}/{epochs}], Total Loss: {avg_epoch_loss:.4f}, "
                f"Labeled Loss: {avg_labeled_loss:.4f},")

            wandb.log({
                "epoch": epoch + 1,
                "avg_epoch_loss": avg_epoch_loss,
                "avg_labeled_loss": avg_labeled_loss,
            })

            # Evaluate model and save if the accuracy improves
            acc, _, rewards_std = self.eval_model(self.eval_loader)
            print(self.rewards_std)
            if self.config.get("save_model", False):
                if best_acc < acc:
                    model_save_path = os.path.join(
                        self.config.get(
                            "output_dir",
                            "./"),
                        self.config.get(
                            "model_save_path",
                            "") +
                        "best_trained_model.pth")
                    torch.save(self.model.state_dict(), model_save_path)
                    print(f"Model saved to {model_save_path}")
                    best_acc = acc
                    self.rewards_std = rewards_std

        if self.config.get("save_model", False):
            model_save_path = os.path.join(
                self.config.get(
                    "output_dir",
                    "./"),
                self.config.get(
                    "model_save_path",
                    "") +
                "final_trained_model.pth")
            torch.save(self.model.state_dict(), model_save_path)
            print(f"Final model saved to {model_save_path}")

        return self.model

    def pair_eval_model(
            self,
            chosen_reject_pair,
            weights=None,
            save_res=True,
            mode='multihead'):
        """
        Evaluate model performance on pairwise data (chosen vs. rejected).

        Parameters:
        - chosen_reject_pair: DataLoader or dataset containing pairs of embeddings (chosen, rejected).
        - weights: Dictionary of weights for each attribute to calculate the weighted score.

        Returns:
        - accuracy: Percentage of cases where chosen score > rejected score.
        """
        self.model.eval()
        correct_count = 0
        total_count = 0
        df_results = pd.DataFrame(columns=['id', 'subset', 'correct'])

        with torch.no_grad():
            for idx, batch in tqdm(enumerate(chosen_reject_pair)):
                chosen_embeddings, rejected_embeddings = batch["chosen_embeddings"].to(
                    self.device), batch["rejected_embeddings"].to(self.device)
                chosen_prompt_embeddings, rejected_prompt_embeddings = batch["chosen_prompt_embeddings"].to(
                    self.device), batch["rejected_prompt_embeddings"].to(self.device)
                batch_size = chosen_embeddings.shape[0]

                # Get model predictions for chosen and rejected
                # shape: (batch_size, num_labels, num_class)
                chosen_outputs = self.model(chosen_embeddings)
                # shape: (batch_size, num_labels, num_class)
                rejected_outputs = self.model(rejected_embeddings)

                if weights is None:
                    weights = (
                        torch.ones(
                            self.num_labels) /
                        self.num_labels).to(
                        self.device)
                else:
                    weights = weights.to(self.device)
                if mode == 'multihead':
                    chosen_score = torch.sigmoid((weights.float(
                    ) * (chosen_outputs - rejected_outputs).squeeze(-1)).sum(dim=1, keepdim=True))
                else:
                    chosen_score = (weights.float() * torch.sigmoid(chosen_outputs -
                                    rejected_outputs).squeeze(-1)).sum(dim=1, keepdim=True)
                rejected_score = 1 - chosen_score
                correct_count += (chosen_score > 0.5).sum().item()
                total_count += chosen_score.size(0)

                if save_res:
                    for i in range(batch_size):
                        correct = 0.5 if chosen_score[i] == rejected_score[i] else float(
                            chosen_score[i] > rejected_score[i])
                        row = {
                            'id': idx * batch_size + i,
                            'correct': correct
                        }
                        if "subset" in batch:
                            row['subset'] = batch['subset'][i]
                            df_results = df_results._append(
                                row, ignore_index=True)

        accuracy = correct_count / total_count if total_count > 0 else 0
        print(f"Pairwise Evaluation Accuracy: {accuracy * 100:.2f}%")
        try:
            wandb.log({"pairwise_evaluation_accuracy": accuracy})
        except BaseException:
            pass

        return accuracy, df_results

    @staticmethod
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def load_model(self, model_path):
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded successfully from {model_path}")


class GateNetworkTrainer():
    def __init__(self, config, labeled_dict, eval_labeled_dict=None):
        self.set_seed(config.get('seed', 42))
        self.config = config
        self.config['use_wandb'] = self.config.get('use_wandb', True)

        # Initialize configurations
        for key, value in self.config.items():
            setattr(self, key, value)

        # Prepare datasets
        train_dataset = PairwiseDataset(labeled_dict)
        if eval_labeled_dict:
            eval_dataset = PairwiseDataset(eval_labeled_dict)
        else:
            train_ratio = config.get('train_ratio', 0.9)
            train_size = int(train_ratio * len(train_dataset))
            eval_size = len(train_dataset) - train_size
            train_dataset, eval_dataset = random_split(
                train_dataset, [train_size, eval_size])

        # Create DataLoaders for training, evaluation, and unlabeled data
        self.labeled_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True)
        self.eval_loader = DataLoader(
            eval_dataset,
            batch_size=self.batch_size,
            shuffle=False)

        self.reward_model = MultiLabelRewardModel(
            input_size=config['input_size'],
            num_labels=self.num_labels,
            num_class=self.num_class,
            num_layers=self.num_layers).to(
            self.device)
        self.model = self.model.to(dtype=torch.bfloat16)
        self.model.load()

        self.gate_network = GatingNetwork(
            in_features=config['input_size'],
            out_features=self.num_labels,
            temperature=self.config.get(
                'temperature_gate',
                1)).to(
            self.device)
        parameters = list(self.gate_network.parameters())
        self.optimizer = optim.AdamW(
            parameters,
            lr=self.learning_rate,
            weight_decay=1e-2)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=30)
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        if self.use_wandb:
            initialize_wandb(config)

    def train():
        self.model.eval()

        for epoch in range(epochs):
            epoch_loss = 0
            labeled_loss_total = 0

            for batch_idx, batch_data in enumerate(
                    tqdm(self.labeled_loader, desc=f"Training Epoch {epoch + 1}")):
                self.gate_network.train()
                self.optimizer.zero_grad()
                binary_loss = 0.0

                text1 = batch_data['chosen_embeddings'].to(self.device)
                text2 = batch_data['rejected_embeddings'].to(self.device)
                prompt = batch_data['chosen_prompt_embeddings'].to(self.device)

                s_A = self.model(text1)
                s_B = self.model(text2)
                weights = self.gate_network(prompt).float()

                outputs = weights * \
                    (s_A - s_B).squeeze(-1).sum(dim=1, keepdim=True)
                weighted_outputs = torch.clamp(weighted_outputs, min=1e-10)
                binary_loss += -F.logsigmoid(weighted_outputs).mean()

            wandb.log({
                "batch_binary_loss": binary_loss.item()
            })

            binary_loss.backward()
            self.optimizer.step()
            epoch_loss += binary_loss.item()

        avg_epoch_loss = epoch_loss / len(self.labeled_loader)
        self.scheduler.step()
        step += 1

        print(
            f"Epoch [{epoch + 1}/{epochs}], Total Loss: {avg_epoch_loss:.4f},")

        wandb.log({
            "epoch": epoch + 1,
            "avg_epoch_loss": avg_epoch_loss,
        })

        acc, _ = self.eval_model(
            self.eval_loader, temperature=self.config.get(
                'temperature_unlabel', 1))
        if self.config.get("save_model", False):
            if best_acc < acc:
                gate_save_path = os.path.join(
                    self.config.get(
                        "output_dir",
                        "./"),
                    self.config.get(
                        "model_save_path",
                        "") +
                    "gate_weights.pth")
                torch.save(self.gate_network.state_dict(), gate_save_path)
                print(f"Model saved to {gate_save_path}")
                best_acc = acc

        if self.config.get("save_model", False):
            gate_save_path = os.path.join(
                self.config.get(
                    "output_dir",
                    "./"),
                self.config.get(
                    "model_save_path",
                    "") +
                "final_gate_weights.pth")
            torch.save(self.gate_network.state_dict(), gate_save_path)
            print(f"Final model saved to {gate_save_path}")

        return self.gate_network

    def eval(eval_loader, weights=None):
        self.model.eval()
        self.gate_network.eval()

        correct_count = 0
        total_count = 0
        df_results = pd.DataFrame(columns=['id', 'subset', 'correct'])

        with torch.no_grad():
            for idx, batch in tqdm(enumerate(eval_loader)):
                chosen_embeddings, rejected_embeddings = batch["chosen_embeddings"].to(
                    self.device), batch["rejected_embeddings"].to(self.device)
                prompt_embeddings = batch["chosen_prompt_embeddings"].to(
                    self.device)
                batch_size = chosen_embeddings.shape[0]

                # Get model predictions for chosen and rejected
                # shape: (batch_size, num_labels, num_class)
                chosen_outputs = self.model(chosen_embeddings)
                # shape: (batch_size, num_labels, num_class)
                rejected_outputs = self.model(rejected_embeddings)

                weights = self.gate_network(prompt_embeddings)
                chosen_score = torch.sigmoid((weights.float(
                ) * (chosen_outputs - rejected_outputs).squeeze(-1)).sum(dim=1, keepdim=True))

                # mixture model setting
                # chosen_score = (weights.float() * torch.sigmoid(chosen_outputs-rejected_outputs).squeeze(-1)).sum(dim=1, keepdim=True)

                rejected_score = 1 - chosen_score
                correct_count += (chosen_score > 0.5).sum().item()
                total_count += chosen_score.size(0)

                if save_res:
                    for i in range(batch_size):
                        correct = 0.5 if chosen_score[i] == rejected_score[i] else float(
                            chosen_score[i] > rejected_score[i])
                        row = {
                            'id': idx * batch_size + i,
                            'correct': correct
                        }
                        if "subset" in batch:
                            row['subset'] = batch['subset'][i]
                            df_results = df_results._append(
                                row, ignore_index=True)

        accuracy = correct_count / total_count if total_count > 0 else 0
        print(f"Pairwise Evaluation Accuracy: {accuracy * 100:.2f}%")
        try:
            wandb.log({"pairwise_evaluation_accuracy": accuracy})
        except BaseException:
            pass

        return accuracy, df_results

    def load_model(self, model_path):
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded successfully from {model_path}")

        if gate_model_path is not None:
            gate_checkpoint = torch.load(
                gate_model_path, map_location=self.device)
            if 'gate_state_dict' in gate_checkpoint:
                self.gate_network.load_state_dict(
                    gate_checkpoint['gate_state_dict'])
            else:
                self.gate_network.load_state_dict(gate_checkpoint)

            self.gate_network.to(self.device)
            self.gate_network.eval()
            print(f"Gating Network loaded successfully from {gate_model_path}")

    @staticmethod
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


class MixtureBTTrainer():
    def __init__(self, config, labeled_dict, eval_labeled_dict=None):
        # Initialize labeled and unlabeled datasets
        self.set_seed(config.get('seed', 42))
        self.config = config
        self.config['use_wandb'] = self.config.get('use_wandb', True)

        # Initialize configurations
        for key, value in self.config.items():
            setattr(self, key, value)

        # Prepare datasets
        train_dataset = PairwiseDataset(labeled_dict)
        if eval_labeled_dict:
            eval_dataset = PairwiseDataset(eval_labeled_dict)
        else:
            train_ratio = config.get('train_ratio', 0.9)
            train_size = int(train_ratio * len(train_dataset))
            eval_size = len(train_dataset) - train_size
            train_dataset, eval_dataset = random_split(
                train_dataset, [train_size, eval_size])

        # Create DataLoaders for training, evaluation, and unlabeled data
        self.labeled_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True)
        self.eval_loader = DataLoader(
            eval_dataset,
            batch_size=self.batch_size,
            shuffle=False)

        self.model = MultiLabelRewardModel(
            input_size=config['input_size'],
            num_labels=self.num_labels,
            num_class=self.num_class,
            num_layers=self.num_layers).to(
            self.device)
        self.model = self.model.to(dtype=torch.bfloat16)
        # self.weights = GatingNetwork(in_features=config['input_size'],out_features=self.K).to(
        #     self.device)
        # parameters = list(self.model.parameters()) + list(self.weights.parameters())
        self.weights = nn.Parameter(torch.rand(self.K, device=self.device))
        parameters = list(self.model.parameters()) + [self.weights]
        self.optimizer = optim.AdamW(
            parameters,
            lr=self.learning_rate,
            weight_decay=1e-2)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=30)
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        if self.use_wandb:
            initialize_wandb(config)

    def eval_model(self, eval_loader, temperature=1):
        self.model.eval()
        all_predictions = []
        all_head_masks = []
        all_probabilities = []
        total_correct = 0
        total_samples = 0

        # Loop through the evaluation data
        with torch.no_grad():
            for batch_data in eval_loader:
            
                texts1 = batch_data['chosen_embeddings'].to(self.device)
                texts2 = batch_data['rejected_embeddings'].to(self.device)
                head_mask = batch_data['head_mask'].to(self.device)
                prompt = batch_data['chosen_prompt_embeddings'].to(self.device)
                # Shape: (batch_size, num_labels, num_class)
                outputs = torch.sigmoid(
                    (self.model(texts1) - self.model(texts2)))
                
                # weights = self.weights(prompt)
                # weights_reshaped = weights.unsqueeze(-1)
                
                with torch.no_grad():
                    weights = self.weights / self.weights.sum()
                weights_reshaped = weights.view(1,self.K,1)
                
                weighted_outputs = (outputs * weights_reshaped).sum(dim=1)
                labels = torch.ones_like(weighted_outputs).to(self.device)
                prob_outputs = torch.cat([1 - weighted_outputs, weighted_outputs], dim=1)
                # Shape: (batch_size, num_labels)
                predicted = torch.argmax(prob_outputs, dim=1)
                total_correct += (predicted ==
                                  labels.squeeze(-1)).sum().item()
                total_samples += labels.numel()

                all_head_predictions = torch.argmax(torch.cat([1 - outputs, outputs], dim=2),dim=2)
                all_predictions.append(all_head_predictions.cpu())
                all_probabilities.append(
                    torch.sigmoid(
                        (self.model(texts1) -
                         self.model(texts2)) /
                        temperature).cpu())
                all_head_masks.append(head_mask.cpu())

        all_predictions = torch.cat(all_predictions, dim=0)
        all_head_masks = torch.cat(all_head_masks, dim=0)
        all_probabilities = torch.cat(all_probabilities, dim=0)

        overall_accuracy = total_correct / total_samples if total_samples > 0 else 0
        print(f"Overall Evaluation Accuracy: {overall_accuracy * 100:.2f}%")
        try:
            wandb.log({"evaluation_accuracy": overall_accuracy})
        except BaseException:
            pass

        # try:
        #     label_accuracies = {}
        #     for i in range(self.num_labels):
        #         if len(all_head_masks) > 0:
        #             tmp_indices = (all_head_masks[:, i] == 1).nonzero(
        #                 as_tuple=True)[0]
        #         else:
        #             tmp_indices = torch.arange(all_predictions.shape[0])
        #             label_true = torch.ones_like(
        #                 all_predictions[:, i], dtype=torch.long)
        #         label_predictions = all_predictions[tmp_indices, i]
        #         label_true = torch.ones_like(label_predictions)
        #         probabilities = all_probabilities[tmp_indices, i]

        #         plt.figure()
        #         plt.hist(probabilities.float().numpy(), bins=20,
        #                  range=(0, 1), alpha=0.75, color='blue')
        #         plt.title("Distribution of Predicted Probabilities")
        #         plt.xlabel("Predicted Probability")
        #         plt.ylabel("Frequency")
        #         plt.grid()
        #         wandb.log({f"head_{i}_probability_histogram": wandb.Image(plt)})
        #         plt.close()

        #         num_samples = label_predictions.size(0)
        #         flip_indices = torch.randperm(num_samples)[:num_samples // 3]

        #         label_true[flip_indices] = 1 - label_true[flip_indices]
        #         label_predictions[flip_indices] = 1 - \
        #             label_predictions[flip_indices]
        #         probabilities[flip_indices] = 1 - probabilities[flip_indices]

        #         label_accuracy = (label_predictions == label_true).sum(
        #         ).item() / label_predictions.size(0)
        #         label_accuracies[f"sample_{i}_accuracy"] = label_accuracy

        #         try:
        #             wandb.log({f"label_{i}_accuracy": label_accuracy})
        #         except BaseException:
        #             pass

        #         prob_true, prob_pred = calibration_curve(
        #             label_true.numpy(), probabilities.float().numpy(), n_bins=10)

        #         # Plot calibration curve
        #         plt.figure()
        #         plt.plot(
        #             prob_pred,
        #             prob_true,
        #             marker='o',
        #             label="Calibration Curve")
        #         plt.plot([0, 1], [0, 1], linestyle="--",
        #                  label="Perfectly Calibrated")
        #         plt.xlabel("Mean Predicted Probability")
        #         plt.ylabel("Fraction of Positives")
        #         plt.title(f"Calibration Curve - Head {i}")
        #         plt.legend()
        #         plt.grid()

        #         # Log calibration curve to WandB
        #         try:
        #             wandb.log(
        #                 {f"head_{i}_calibration_curve": wandb.Image(plt)})
        #         except BaseException:
        #             pass
        #         plt.close()

        except BaseException:
            pass

        return overall_accuracy, all_predictions

    def train_model(
            self,
            epochs=5,
            warm_up_steps=1,
            thresholds=None,
            alpha=0.1,
            beta=0.1,
            add_gate=True):
        best_acc = 0  # Initialize best accuracy for model saving
        step = 0
        self.combined_labeled_loader = self.labeled_loader

        for epoch in range(epochs):
            epoch_loss = 0
            labeled_loss_total = 0

            for batch_idx, batch_data in enumerate(
                    tqdm(self.combined_labeled_loader, desc=f"Training Epoch {epoch + 1}")):
                self.model.train()
                self.optimizer.zero_grad()
                

                text1 = batch_data['chosen_embeddings'].to(self.device)
                text2 = batch_data['rejected_embeddings'].to(self.device)
                prompt = batch_data['chosen_prompt_embeddings'].to(self.device)
                head_mask = batch_data['head_mask'].to(
                    self.device)  # torch.Size([2048, 5, 1])
                # torch.Size([2048, 5])
                labels = batch_data['labels'].squeeze(-1).to(self.device)

                s_A = self.model(text1)
                s_B = self.model(text2)

                outputs = s_A - s_B
                # weights = self.weights(prompt)
                # weights_reshaped = weights.unsqueeze(-1)
                # import ipdb
                # ipdb.set_trace()
                
                with torch.no_grad():
                    weights = self.weights / self.weights.sum()
                weights_reshaped = weights.view(1,self.K,1)
                
                loss = - \
                    torch.log((nn.functional.sigmoid(s_A - s_B) * weights_reshaped).sum(dim=-1)).mean() - 0.01*(weights*torch.log(weights)).sum()
                # weighted_outputs = (weights.float() * torch.sigmoid(outputs).squeeze(-1)).sum(dim=1, keepdim=True)
                # weighted_outputs = torch.clamp(weighted_outputs, min=1e-8)
                # weights_q_step =  weights.float() * torch.sigmoid(outputs).squeeze(-1) / weighted_outputs
                
                # weighted_outputs_m_step = (weights_q_step.float() * F.logsigmoid(outputs).squeeze(-1)).sum(dim=1, keepdim=True)     
                # weights = torch.clamp(weights, min=1e-8)
                # loss = -(weighted_outputs_m_step.mean() + (weights_q_step * torch.log(weights)).mean())
                if self.auxilary_loss:
                    text1_flatten = s_A.view(
                        outputs.shape[0] * self.num_labels, self.num_class)
                    text2_flatten = s_B.view(
                        outputs.shape[0] * self.num_labels, self.num_class)
                    loss = (loss + 0.01 * (text1_flatten**2 +
                            text2_flatten**2).mean())

                wandb.log({
                    "batch_total_loss": loss.item(),
                })

                # Backpropagation
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            # Logging for each epoch
            avg_epoch_loss = epoch_loss / len(self.labeled_loader)
            avg_labeled_loss = labeled_loss_total / len(self.labeled_loader)
            self.scheduler.step()
            step += 1

            print(
                f"Epoch [{epoch + 1}/{epochs}], Total Loss: {avg_epoch_loss:.4f}, "
                f"Labeled Loss: {avg_labeled_loss:.4f},")

            wandb.log({
                "epoch": epoch + 1,
                "avg_epoch_loss": avg_epoch_loss,
                "avg_labeled_loss": avg_labeled_loss,
            })

            # Evaluate model and save if the accuracy improves
            acc, _ = self.eval_model(self.eval_loader)
            if self.config.get("save_model", False):
                if best_acc < acc:
                    model_save_path = os.path.join(
                        self.config.get(
                            "output_dir",
                            "./"),
                        self.config.get(
                            "model_save_path",
                            "") +
                        "best_trained_model.pth")
                    torch.save(self.model.state_dict(), model_save_path)
                    print(f"Model saved to {model_save_path}")
                    best_acc = acc

        if self.config.get("save_model", False):
            model_save_path = os.path.join(
                self.config.get(
                    "output_dir",
                    "./"),
                self.config.get(
                    "model_save_path",
                    "") +
                "final_trained_model.pth")
            torch.save(self.model.state_dict(), model_save_path)
            print(f"Final model saved to {model_save_path}")

        return self.model

    def pair_eval_model(
            self,
            chosen_reject_pair,
            weights=None,
            save_res=True,
            mode='multihead'):
        """
        Evaluate model performance on pairwise data (chosen vs. rejected).

        Parameters:
        - chosen_reject_pair: DataLoader or dataset containing pairs of embeddings (chosen, rejected).
        - weights: Dictionary of weights for each attribute to calculate the weighted score.

        Returns:
        - accuracy: Percentage of cases where chosen score > rejected score.
        """
        self.model.eval()
        correct_count = 0
        total_count = 0
        df_results = pd.DataFrame(columns=['id', 'subset', 'correct'])
        if weights is None:
            self.weights = self.weights.to(self.device)
        else:
            weights = weights.to(self.device)

        with torch.no_grad():
            for idx, batch in tqdm(enumerate(chosen_reject_pair)):
                chosen_embeddings, rejected_embeddings = batch["chosen_embeddings"].to(
                    self.device), batch["rejected_embeddings"].to(self.device)
                chosen_prompt_embeddings, rejected_prompt_embeddings = batch["chosen_prompt_embeddings"].to(
                    self.device), batch["rejected_prompt_embeddings"].to(self.device)
                batch_size = chosen_embeddings.shape[0]
            

                # Get model predictions for chosen and rejected
                # shape: (batch_size, num_labels, num_class)
                chosen_outputs = self.model(chosen_embeddings)
                # shape: (batch_size, num_labels, num_class)
                rejected_outputs = self.model(rejected_embeddings)

                if weights is None:
                    # self.weights = self.weights.to(self.device)
                    # weights_dist = self.weights(chosen_prompt_embeddings)
                    weights_dist = self.weights.to(self.device)
                else:
                    weights_dist = weights.to(self.device)
                chosen_score = (weights_dist.float() * torch.sigmoid(chosen_outputs -
                                rejected_outputs).squeeze(-1)).sum(dim=1, keepdim=True)
                rejected_score = 1 - chosen_score
                correct_count += (chosen_score > 0.5).sum().item()
                total_count += chosen_score.size(0)

                if save_res:
                    for i in range(batch_size):
                        correct = 0.5 if chosen_score[i] == rejected_score[i] else float(
                            chosen_score[i] > rejected_score[i])
                        row = {
                            'id': idx * batch_size + i,
                            'correct': correct
                        }
                        if "subset" in batch:
                            row['subset'] = batch['subset'][i]
                            df_results = df_results._append(
                                row, ignore_index=True)

        accuracy = correct_count / total_count if total_count > 0 else 0
        print(f"Pairwise Evaluation Accuracy: {accuracy * 100:.2f}%")
        try:
            wandb.log({"pairwise_evaluation_accuracy": accuracy})
        except BaseException:
            pass

        return accuracy, df_results

    @staticmethod
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def load_model(self, model_path):
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded successfully from {model_path}")


class SemiRewardClassifierTrainer():
    def __init__(self, config, labeled_dict, eval_labeled_dict=None):
        # Initialize labeled and unlabeled datasets
        self.set_seed(config.get('seed', 42))
        self.config = config
        self.config['use_wandb'] = self.config.get('use_wandb', True)

        # Initialize configurations
        for key, value in self.config.items():
            setattr(self, key, value)

        # Prepare datasets
        train_dataset = PairwiseDataset(labeled_dict)
        if eval_labeled_dict:
            eval_dataset = PairwiseDataset(eval_labeled_dict)
        else:
            train_ratio = config.get('train_ratio', 0.9)
            train_size = int(train_ratio * len(train_dataset))
            eval_size = len(train_dataset) - train_size
            train_dataset, eval_dataset = random_split(
                train_dataset, [train_size, eval_size])

        # Create DataLoaders for training, evaluation, and unlabeled data
        self.labeled_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True)
        self.eval_loader = DataLoader(
            eval_dataset,
            batch_size=self.batch_size,
            shuffle=False)

        self.model = MultiLabelRewardModel(
            input_size=config['input_size'],
            num_labels=self.num_labels,
            num_class=self.num_class,
            num_layers=self.num_layers).to(
            self.device)
        self.model = self.model.to(dtype=torch.bfloat16)
        parameters = list(self.model.parameters())
        self.optimizer = optim.AdamW(
            parameters,
            lr=self.learning_rate,
            weight_decay=1e-2)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=30)
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        if self.use_wandb:
            initialize_wandb(config)

    def eval_model(self, eval_loader, temperature=1):
        self.model.eval()
        all_predictions = []
        all_head_masks = []
        all_probabilities = []
        total_correct = 0
        total_samples = 0

        # Loop through the evaluation data
        with torch.no_grad():
            for batch_data in eval_loader:
                texts1 = batch_data['chosen_embeddings'].to(self.device)
                texts2 = batch_data['rejected_embeddings'].to(self.device)
                head_mask = batch_data['head_mask'].to(self.device)
                # Shape: (batch_size, num_labels, num_class)
                outputs_pos = torch.sigmoid(self.model(texts1))
                outputs_neg = torch.sigmoid(self.model(texts2))
                labels = torch.ones_like(outputs).to(self.device)
                prob_outputs = torch.cat([1 - outputs, outputs], dim=2)
                # Shape: (batch_size, num_labels)
                predicted = torch.argmax(prob_outputs, dim=2)

                all_predictions.append(predicted.cpu())
                all_probabilities.append(
                    torch.sigmoid(
                        (self.model(texts1) -
                         self.model(texts2)) /
                        temperature).cpu())

                if head_mask is not None:
                    total_correct += ((predicted == labels.squeeze(-1)
                                       ).float() * head_mask).sum().item()
                    total_samples += head_mask.squeeze(-1).sum().item()
                    all_head_masks.append(head_mask.cpu())
                else:
                    total_correct += (predicted ==
                                      labels.squeeze(-1)).sum().item()
                    total_samples += labels.numel()

        all_predictions = torch.cat(all_predictions, dim=0)
        all_head_masks = torch.cat(all_head_masks, dim=0)
        all_probabilities = torch.cat(all_probabilities, dim=0)

        overall_accuracy = total_correct / total_samples if total_samples > 0 else 0
        print(f"Overall Evaluation Accuracy: {overall_accuracy * 100:.2f}%")
        try:
            wandb.log({"evaluation_accuracy": overall_accuracy})
        except BaseException:
            pass

        try:
            label_accuracies = {}
            for i in range(self.num_labels):
                if len(all_head_masks) > 0:
                    tmp_indices = (all_head_masks[:, i] == 1).nonzero(
                        as_tuple=True)[0]
                else:
                    tmp_indices = torch.arange(all_predictions.shape[0])
                    label_true = torch.ones_like(
                        all_predictions[:, i], dtype=torch.long)
                label_predictions = all_predictions[tmp_indices, i]
                label_true = torch.ones_like(label_predictions)
                probabilities = all_probabilities[tmp_indices, i]

                plt.figure()
                plt.hist(probabilities.float().numpy(), bins=20,
                         range=(0, 1), alpha=0.75, color='blue')
                plt.title("Distribution of Predicted Probabilities")
                plt.xlabel("Predicted Probability")
                plt.ylabel("Frequency")
                plt.grid()
                wandb.log({f"head_{i}_probability_histogram": wandb.Image(plt)})
                plt.close()

                num_samples = label_predictions.size(0)
                flip_indices = torch.randperm(num_samples)[:num_samples // 3]

                label_true[flip_indices] = 1 - label_true[flip_indices]
                label_predictions[flip_indices] = 1 - \
                    label_predictions[flip_indices]
                probabilities[flip_indices] = 1 - probabilities[flip_indices]

                label_accuracy = (label_predictions == label_true).sum(
                ).item() / label_predictions.size(0)
                label_accuracies[f"sample_{i}_accuracy"] = label_accuracy

                try:
                    wandb.log({f"label_{i}_accuracy": label_accuracy})
                except BaseException:
                    pass

                prob_true, prob_pred = calibration_curve(
                    label_true.numpy(), probabilities.float().numpy(), n_bins=10)

                # Plot calibration curve
                plt.figure()
                plt.plot(
                    prob_pred,
                    prob_true,
                    marker='o',
                    label="Calibration Curve")
                plt.plot([0, 1], [0, 1], linestyle="--",
                         label="Perfectly Calibrated")
                plt.xlabel("Mean Predicted Probability")
                plt.ylabel("Fraction of Positives")
                plt.title(f"Calibration Curve - Head {i}")
                plt.legend()
                plt.grid()

                # Log calibration curve to WandB
                try:
                    wandb.log(
                        {f"head_{i}_calibration_curve": wandb.Image(plt)})
                except BaseException:
                    pass
                plt.close()

        except BaseException:
            pass

        return overall_accuracy, all_predictions

    def generate_pseudo_labels(
            self,
            unlabeled_loader,
            thresholds=None,
            temperature=1.0,
            apply_calibration=False):
        pseudo_labels = []
        selected_index = []
        pseudo_head_masks = []  # To store the generated head masks for each pseudo-label
        self.model.eval()
        all_max_probs = []
        all_outputs = []

        with torch.no_grad():
            index = 0
            for batch_data in tqdm(
                    unlabeled_loader, desc="Generating pseudo labels"):
                texts1 = batch_data['chosen_embeddings'].to(self.device)
                texts2 = batch_data['rejected_embeddings'].to(self.device)
                outputs1 = self.model(texts1)
                outputs2 = self.model(texts2)
                outputs = outputs1 - outputs2

                probs = torch.sigmoid(outputs / temperature)

                # If no thresholds are provided, default to 0.5 for each label
                if thresholds is None:
                    thresholds = torch.tensor(
                        [[0.5] * probs.shape[2]] * probs.shape[1]).to(self.device)

                all_max_probs.append(
                    torch.max(torch.cat([probs, 1 - probs], dim=2), dim=2)[0])
                all_outputs.append(outputs)
                thresholds = torch.tensor(thresholds).view(
                    1, -1, 1).to(probs.device)

                valid_mask = (
                    (probs > thresholds) | (
                        (1 - probs) > thresholds)).float()
                valid_label_set = (probs > thresholds).float() * valid_mask

                valid_mask_reduced = valid_mask.any(dim=2).any(dim=1)
                valid_indices = torch.where(valid_mask_reduced)[0].cpu()

                if valid_indices.any():
                    pseudo_labels.extend(
                        valid_label_set[valid_indices].detach().cpu())
                    selected_index.extend(
                        (torch.arange(
                            probs.shape[0])[valid_indices] +
                            index).tolist())
                    pseudo_head_masks.extend(
                        valid_mask[valid_indices].detach().cpu())

                index += len(texts1)

                # threshold update
                # thresholds = 0.9 * thresholds[0,:,0] + (1-0.9)*torch.cat(all_max_probs,dim=0).mean(dim=0)

        return selected_index, pseudo_labels, pseudo_head_masks, thresholds, all_max_probs, all_outputs

    def train_model(
            self,
            epochs=5,
            warm_up_steps=1,
            thresholds=None,
            alpha=0.1,
            beta=0.1,
            add_gate=True):
        best_acc = 0  # Initialize best accuracy for model saving
        step = 0
        self.combined_labeled_loader = self.labeled_loader

        for epoch in range(epochs):
            epoch_loss = 0
            labeled_loss_total = 0

            for batch_idx, batch_data in enumerate(
                    tqdm(self.combined_labeled_loader, desc=f"Training Epoch {epoch + 1}")):
                self.model.train()
                self.optimizer.zero_grad()

                text1 = batch_data['chosen_embeddings'].to(self.device)
                text2 = batch_data['rejected_embeddings'].to(self.device)
                head_mask = batch_data['head_mask'].to(
                    self.device)  # torch.Size([2048, 5, 1])
                # torch.Size([2048, 5])
                labels = batch_data['labels'].squeeze(-1).to(self.device)

                s_A = self.model(text1)
                s_B = self.model(text2)

                outputs = s_A - s_B
                criterion = nn.BCEWithLogitsLoss(reduction='none')
                loss = criterion(
                    outputs.view(
                        (outputs.shape[0] * self.num_labels,
                         self.num_class)),
                    labels.view(
                        -1,
                        1))  # Shape: (batch_size, num_heads, 1)

                head_mask_flatten = head_mask.view(
                    (outputs.shape[0] * self.num_labels, self.num_class))
                if self.auxilary_loss:
                    text1_flatten = s_A.view(
                        outputs.shape[0] * self.num_labels, self.num_class)
                    text2_flatten = s_B.view(
                        outputs.shape[0] * self.num_labels, self.num_class)
                    loss = (loss + 0.01 * (text1_flatten**2 +
                            text2_flatten**2)) * head_mask_flatten
                else:
                    loss = loss * head_mask_flatten

                loss = loss.sum() / head_mask.sum()
                if head_mask.sum() == 0:
                    loss = torch.tensor(0.0, device=self.device)

                predicted_labels = (outputs.sigmoid() > 0.5).float()
                correct_predictions = (
                    predicted_labels == labels.unsqueeze(-1)).float() * head_mask
                label_accuracy_total = correct_predictions.sum(
                ) / head_mask.sum() if head_mask.sum() > 0 else 0

                wandb.log({
                    "batch_total_loss": loss.item(),
                })

                # Backpropagation
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            # Logging for each epoch
            avg_epoch_loss = epoch_loss / len(self.labeled_loader)
            avg_labeled_loss = labeled_loss_total / len(self.labeled_loader)
            self.scheduler.step()
            step += 1

            print(
                f"Epoch [{epoch + 1}/{epochs}], Total Loss: {avg_epoch_loss:.4f}, "
                f"Labeled Loss: {avg_labeled_loss:.4f},")

            wandb.log({
                "epoch": epoch + 1,
                "avg_epoch_loss": avg_epoch_loss,
                "avg_labeled_loss": avg_labeled_loss,
            })

            # Evaluate model and save if the accuracy improves
            acc, _ = self.eval_model(self.eval_loader)
            if self.config.get("save_model", False):
                if best_acc < acc:
                    model_save_path = os.path.join(
                        self.config.get(
                            "output_dir",
                            "./"),
                        self.config.get(
                            "model_save_path",
                            "") +
                        "best_trained_model.pth")
                    torch.save(self.model.state_dict(), model_save_path)
                    print(f"Model saved to {model_save_path}")
                    best_acc = acc

        if self.config.get("save_model", False):
            model_save_path = os.path.join(
                self.config.get(
                    "output_dir",
                    "./"),
                self.config.get(
                    "model_save_path",
                    "") +
                "final_trained_model.pth")
            torch.save(self.model.state_dict(), model_save_path)
            print(f"Final model saved to {model_save_path}")

        return self.model

    def pair_eval_model(
            self,
            chosen_reject_pair,
            weights=None,
            save_res=True,
            mode='multihead'):
        """
        Evaluate model performance on pairwise data (chosen vs. rejected).

        Parameters:
        - chosen_reject_pair: DataLoader or dataset containing pairs of embeddings (chosen, rejected).
        - weights: Dictionary of weights for each attribute to calculate the weighted score.

        Returns:
        - accuracy: Percentage of cases where chosen score > rejected score.
        """
        self.model.eval()
        correct_count = 0
        total_count = 0
        df_results = pd.DataFrame(columns=['id', 'subset', 'correct'])

        with torch.no_grad():
            for idx, batch in tqdm(enumerate(chosen_reject_pair)):
                chosen_embeddings, rejected_embeddings = batch["chosen_embeddings"].to(
                    self.device), batch["rejected_embeddings"].to(self.device)
                chosen_prompt_embeddings, rejected_prompt_embeddings = batch["chosen_prompt_embeddings"].to(
                    self.device), batch["rejected_prompt_embeddings"].to(self.device)
                batch_size = chosen_embeddings.shape[0]

                # Get model predictions for chosen and rejected
                # shape: (batch_size, num_labels, num_class)
                chosen_outputs = self.model(chosen_embeddings)
                # shape: (batch_size, num_labels, num_class)
                rejected_outputs = self.model(rejected_embeddings)

                if weights is None:
                    weights = (
                        torch.ones(
                            self.num_labels) /
                        self.num_labels).to(
                        self.device)
                else:
                    weights = weights.to(self.device)
                if mode == 'multihead':
                    chosen_score = torch.sigmoid((weights.float(
                    ) * (chosen_outputs - rejected_outputs).squeeze(-1)).sum(dim=1, keepdim=True))
                else:
                    chosen_score = (weights.float() * torch.sigmoid(chosen_outputs -
                                    rejected_outputs).squeeze(-1)).sum(dim=1, keepdim=True)
                rejected_score = 1 - chosen_score
                correct_count += (chosen_score > 0.5).sum().item()
                total_count += chosen_score.size(0)

                if save_res:
                    for i in range(batch_size):
                        correct = 0.5 if chosen_score[i] == rejected_score[i] else float(
                            chosen_score[i] > rejected_score[i])
                        row = {
                            'id': idx * batch_size + i,
                            'correct': correct
                        }
                        if "subset" in batch:
                            row['subset'] = batch['subset'][i]
                            df_results = df_results._append(
                                row, ignore_index=True)

        accuracy = correct_count / total_count if total_count > 0 else 0
        print(f"Pairwise Evaluation Accuracy: {accuracy * 100:.2f}%")
        try:
            wandb.log({"pairwise_evaluation_accuracy": accuracy})
        except BaseException:
            pass

        return accuracy, df_results

    @staticmethod
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def load_model(self, model_path, gate_model_path=None):
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded successfully from {model_path}")


