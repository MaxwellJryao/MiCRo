import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns

COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',  
    '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5', 
    '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5'  
]

def save_individual_attribute_plots(train_chosen, 
                                    train_rejected, unlabeled_chosen, unlabeled_rejected, train_head_masks,
                                    title, output_dir, attributes):

    train_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']  
    unlabeled_colors = ['#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5', 
    '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5' ]  
    
    for i, attribute in enumerate(attributes):
        fig, ax = plt.subplots(figsize=(15, 12))
        valid_indices = torch.where(train_head_masks[:,i]==1)[0]
        sns.kdeplot(
            train_chosen[valid_indices, i].squeeze(), 
            ax=ax,
            label=f'Train Chosen - {attribute}',
            color=train_colors[0],
            linewidth=2.5,
            fill=True
        )
        valid_indices = torch.where(train_head_masks[:,i]==1)[0]
        sns.kdeplot(
            train_rejected[valid_indices, i].squeeze(), 
            ax=ax,
            label=f'Train Rejected - {attribute}',
            color=train_colors[1],
            linewidth=2.5,
            fill=True
        )

        sns.kdeplot(
            unlabeled_chosen[:, i].squeeze(), 
            ax=ax,
            label=f'Unlabeled Chosen - {attribute}',
            color=unlabeled_colors[0],
            linewidth=2.5,
            fill=True
        )
        sns.kdeplot(
            unlabeled_rejected[:, i].squeeze(), 
            ax=ax,
            label=f'Unlabeled Rejected - {attribute}',
            color=unlabeled_colors[1],
            linewidth=2.5,
            fill=True
        )

        plt.title(f'Distribution for {attribute} - {title}', fontsize=16)
        plt.xlabel('Output Value', fontsize=14)
        plt.ylabel('Density', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
        plt.legend(loc='upper right', fontsize=10)

        output_path = os.path.join(output_dir, f"{title.replace(' ', '_').lower()}_{attribute}_distribution.png")
        plt.savefig(output_path, dpi=300)
        plt.close()
        
def save_embeddings_distribution_plot(trainer, chosen_embeddings, reject_embeddings, title, output_dir, attributes, train_head_masks=None):
    chosen_embeddings = chosen_embeddings.to(trainer.device)
    reject_embeddings = reject_embeddings.to(trainer.device)
    outputs = (trainer.model(chosen_embeddings)-trainer.model(reject_embeddings)).to(torch.float32).detach().cpu().numpy()

    assert len(attributes) == outputs.shape[1], "Attributes list length must match the output dimensions."

    fig, ax = plt.subplots()
    for i in range(outputs.shape[1]):
        if 'Unlabeled' not in title:
            valid_indices = torch.where(train_head_masks[:,i]==1)[0]
            sns.kdeplot(
                outputs[valid_indices, i].squeeze(), 
                ax=ax,
                label=attributes[i],
                color=COLORS[i],
                linewidth=2.5,
                fill=True 
            )
        else:
            sns.kdeplot(
            outputs[:, i].squeeze(), 
            ax=ax,
            label=attributes[i],
            color=COLORS[i],
            linewidth=2.5,
            fill=True 
        )

    plt.title(f'Distribution of Model Outputs for {title}', fontsize=16)
    plt.xlabel('Output Value', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)  
    plt.legend(loc='upper right', fontsize=12)  


    output_path = os.path.join(output_dir, f"{title.replace(' ', '_').lower()}_distribution.png")
    plt.savefig(output_path, dpi=300)  
    plt.close()

