import torch
from transformers import T5ForConditionalGeneration, AutoModelForQuestionAnswering
import numpy as np
from tqdm import tqdm

def estimate_rank(matrix, threshold=0.99):
    """
    Estimate the effective rank of a matrix using SVD.
    
    Args:
    matrix (torch.Tensor): The input matrix.
    threshold (float): The cumulative energy threshold (default: 0.99).
    
    Returns:
    int: The estimated effective rank.
    """
    U, S, V = torch.svd(matrix)
    cumulative_energy = torch.cumsum(S, dim=0) / torch.sum(S)
    return torch.sum(cumulative_energy < threshold).item() + 1

def analyze_model_ranks(model, layer_types=None):
    """
    Analyze the ranks of weight matrices in the model.
    
    Args:
    model (torch.nn.Module): The input model.
    layer_types (list): Types of layers to analyze (default: None, analyze all linear layers).
    
    Returns:
    dict: A dictionary containing rank information for each analyzed layer.
    """
    rank_info = {}
    
    if layer_types is None:
        layer_types = [torch.nn.Linear]
    
    for name, module in tqdm(model.named_modules(), desc="Analyzing layers"):
        if any(isinstance(module, layer_type) for layer_type in layer_types):
            weight = module.weight.data
            if weight.dim() == 2:
                estimated_rank = estimate_rank(weight)
                rank_info[name] = {
                    'shape': weight.shape,
                    'estimated_rank': estimated_rank,
                    'full_rank': min(weight.shape),
                    'rank_ratio': estimated_rank / min(weight.shape)
                }
    
    return rank_info

def suggest_lora_rank(rank_info, percentile=50):
    """
    Suggest a LoRA rank based on the analyzed rank information.
    
    Args:
    rank_info (dict): The rank information dictionary.
    percentile (int): The percentile to use for suggestion (default: 50, median).
    
    Returns:
    int: The suggested LoRA rank.
    """
    estimated_ranks = [info['estimated_rank'] for info in rank_info.values()]
    suggested_rank = int(np.percentile(estimated_ranks, percentile))
    return max(1, min(suggested_rank, 64))  # Clamp between 1 and 64

# Load the t5-small model
model = T5ForConditionalGeneration.from_pretrained('t5-small')
#model=AutoModelForQuestionAnswering.from_pretrained('Intel/dynamic_tinybert')
# Analyze the model's weight matrices
rank_info = analyze_model_ranks(model)

# Print rank information
for layer_name, info in rank_info.items():
    print(f"Layer: {layer_name}")
    print(f"  Shape: {info['shape']}")
    print(f"  Estimated Rank: {info['estimated_rank']}")
    print(f"  Full Rank: {info['full_rank']}")
    print(f"  Rank Ratio: {info['rank_ratio']:.2f}")
    print()

# Suggest LoRA rank
suggested_rank = suggest_lora_rank(rank_info)
print(f"Suggested LoRA rank: {suggested_rank}")

# Calculate average rank ratio
avg_rank_ratio = np.mean([info['rank_ratio'] for info in rank_info.values()])
print(f"Average rank ratio: {avg_rank_ratio:.2f}")

# Iterate through each parameter in the model
for name, param in model.named_parameters():
    if param.requires_grad and param.dim() == 2:  # Check if the parameter is a 2D weight matrix
        print(f"Layer: {name}")
        print(f"  Shape: {param.shape}")
        print(f"  Size: {param.numel()}")  # Total number of elements (size) in the matrix
        print()