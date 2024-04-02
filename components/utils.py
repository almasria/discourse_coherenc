import statistics
from typing import List
import torch


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask) -> torch.Tensor:
    """
    Mean pooling to get sentence embeddings
        Args:
            model_output (torch.Tensor): Model output
            attention_mask (torch.Tensor): Attention mask


        Returns:
            torch.Tensor: Sentence embeddings
    """
    token_embeddings = model_output[
        0
    ]  # First element of model_output contains all token embeddings
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


# Compute the mean, median, or mode of a list of numbers
def compute_statistic(numbers: List[float], statistic: str = "mean") -> float:
    """
    Calculate the mean, median, or mode of a list of numbers
        Args:
            numbers (List): List of numbers
            statistic (str): Statistic to calculate

        Returns:
            float: Calculated statistic
    """
    if statistic == "mean":
        return statistics.mean(numbers)
    elif statistic == "median":
        return statistics.median(numbers)
    elif statistic == "mode":
        return statistics.mode(numbers)
    else:
        raise ValueError(
            "Invalid statistic parameter. Expected 'mean', 'median', or 'mode'."
        )
