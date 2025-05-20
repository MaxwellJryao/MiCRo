from trl import RewardTrainer
import torch
from typing import Any, Dict, List, Optional, Union, Tuple
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
from dataclasses import dataclass, field
from transformers.utils import PaddingStrategy
import torch.nn.functional as F
import torch.nn as nn
import wandb


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
            if "head_masks" in feature:
                merged_features.append(
                    {
                        "input_ids": feature["input_ids_chosen"],
                        "attention_mask": feature["attention_mask_chosen"],
                        "head_masks":feature["head_masks"]
                    }
                )
                merged_features.append(
                    {
                        "input_ids": feature["input_ids_rejected"],
                        "attention_mask": feature["attention_mask_rejected"],
                        "head_masks":feature["head_masks"]
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
                    "head_masks":batch["head_masks"],
                    "return_loss": True,
                    'prompt_length': torch.tensor([feature['prompt_length'] for feature in features]),
                }
            else:
                merged_features.append(
                    {
                        "input_ids": feature["input_ids_chosen"],
                        "attention_mask": feature["attention_mask_chosen"]
                    }
                )
                merged_features.append(
                    {
                        "input_ids": feature["input_ids_rejected"],
                        "attention_mask": feature["attention_mask_rejected"]
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

class MultiRewardTrainer(RewardTrainer):
    def __init__(self, *args, script_args=None, **kwargs):
        super().__init__(*args, **kwargs)  
        self.script_args = script_args

    def compute_loss(self, model, inputs, return_outputs=False):
        script_args = self.script_args

        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_hidden_states=True)
        rewards, last_hidden_state = outputs.logits, outputs.hidden_states[-1][:, -1, :]
        bsz = rewards.size(0)

        jidx = torch.arange(0, bsz, 2)
        kidx = jidx + 1
        rewards_j = rewards[jidx] # (bsz,num_heads)
        rewards_k = rewards[kidx]

        diff = rewards_j - rewards_k
        diff_flat = diff.view(-1)
        if "head_masks" in inputs:
            head_mask_flatten = inputs["head_masks"][jidx].view(-1)
        else:
            head_mask_flatten = torch.ones_like(rewards_j).view(-1)

        if script_args.loss_type == 'BT':
            loss = - (nn.functional.logsigmoid(diff_flat) * head_mask_flatten).mean()
            wandb.log({'origin BT loss': loss})
        else:
            raise NotImplementedError

        if return_outputs:
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
        logits = tuple(v for k, v in logits_dict.items() if k not in ignore_keys)
        logits = nested_detach(logits)
        logits = torch.stack(logits,dim=-1).softmax(dim=2) #(batch_size, num_heads, 2)

        bsz = logits.size(0) * 2
        jidx = torch.arange(0, bsz, 2)
        head_mask = inputs['head_masks'][jidx]
        assert logits.shape[0]==head_mask.shape[0], "Predicted logits & head masks dimension mismatch!"

        head_mask = head_mask.bool()
        filtered_logits = logits[head_mask]  

        labels = torch.zeros(logits.shape[0])
        filtered_labels = labels[head_mask].view(1, -1)
        filtered_labels = self._prepare_inputs(filtered_labels)

        return loss, filtered_logits, filtered_labels

class MultiRewardBinaryTrainer(RewardTrainer):
    def __init__(self, *args, script_args=None, **kwargs):
        super().__init__(*args, **kwargs)  
        self.script_args = script_args

    def compute_loss(self, model, inputs, return_outputs=False):
        script_args = self.script_args

        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_hidden_states=True)
        rewards, last_hidden_state = outputs.logits, outputs.hidden_states[-1][:, -1, :]
        bsz = rewards.size(0)

        jidx = torch.arange(0, bsz, 2)
        kidx = jidx + 1
        rewards_j = rewards[jidx] # (bsz,num_heads)
        rewards_k = rewards[kidx]

        diff = rewards_j - rewards_k
        diff_flat = diff.view(-1)
        if "head_masks" in inputs:
            head_mask_flatten = inputs["head_masks"][jidx].view(-1)
        else:
            head_mask_flatten = torch.ones_like(rewards_j).view(-1)

        if script_args.loss_type == 'binary':
            loss = - (nn.functional.logsigmoid(diff_flat) * head_mask_flatten).mean()
            wandb.log({'origin binary classification loss': loss})
        else:
            raise NotImplementedError

        if return_outputs:
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

        
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_hidden_states=True
        )
        # hidden states of the last layer, last token
        rewards, last_hidden_state = outputs.logits, outputs.hidden_states[-1][:, -1, :]
        bsz = rewards.size(0)
        jidx = torch.arange(0, bsz, 2)
        kidx = jidx + 1
        rewards_j = rewards[jidx]
        rewards_k = rewards[kidx]

        ###################################
        # Combine rewards and labels
        rewards_combined = torch.cat([rewards_j, rewards_k], dim=0)
        labels = torch.cat([
            torch.ones(rewards_j.size(0), device=rewards_j.device),  # Labels for rewards_j
            torch.zeros(rewards_k.size(0), device=rewards_k.device)  # Labels for rewards_k
        ])

        # Compute the raw loss
        raw_loss = F.binary_cross_entropy_with_logits(
            rewards_combined.squeeze(),  # Ensure correct shape
            labels,
            reduction="none"  # Compute loss for each element without reducing
        )

        ###################################
        # Apply the head mask
        if "head_masks" in inputs:
            head_mask_flatten_j = inputs["head_masks"][jidx].view(-1)  # jidx 对应的 mask
            head_mask_flatten_k = inputs["head_masks"][kidx].view(-1)  # kidx 对应的 mask
        else:
            head_mask_flatten_j = torch.ones_like(rewards_j.view(-1))  # 默认全 1
            head_mask_flatten_k = torch.ones_like(rewards_k.view(-1))  # 默认全 1

        # Combine head masks
        head_masks_combined = torch.cat([head_mask_flatten_j, head_mask_flatten_k], dim=0)

        # Apply the head mask to the raw loss
        masked_loss = raw_loss * head_masks_combined

        # Reduce the loss (e.g., take the mean)
        final_loss = masked_loss.mean()

        ###################################
        return final_loss, {
            "rewards_j": rewards_j,
            "rewards_k": rewards_k,
            "raw_loss": raw_loss,
            "masked_loss": masked_loss,
            "head_masks_combined": head_masks_combined
        }