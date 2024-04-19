import statistics
from typing import List, Dict
import torch
import numpy as np
import torch.nn.functional as F


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(
    model_output: torch.Tensor = None, attention_mask: torch.Tensor = None
) -> torch.Tensor:
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
def compute_statistics(numbers: List[float], statistic: str = "mean") -> float:
    """
    Calculate the mean, median, or mode of a list of numbers
        Args:
            numbers (List): List of numbers
            statistic (str): Statistic to calculate

        Returns:
            float: Calculated statistic
    """
    if statistic == "mean":
        return {statistic: statistics.mean(numbers)}
    elif statistic == "median":
        return {statistic: statistics.median(numbers)}
    elif statistic == "mode":
        return {statistic: statistics.mode(numbers)}
    elif statistic == "all":
        return {
            "mean": statistics.mean(numbers),
            "median": statistics.median(numbers),
            "mode": statistics.mode(numbers),
            "variance": statistics.variance(numbers),
            "stdev": statistics.stdev(numbers),
        }
    else:
        raise ValueError(
            "Invalid statistic parameter. Expected 'mean', 'median', or 'mode'."
        )


def cos_sim(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Compute the cosine similarity between two vectors
        Args:
            v1 (np.ndarray): The first vector
            v2 (np.ndarray): The second vector

        Returns:
            float: Cosine similarity
    """
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def normalize_vectors(vectors: torch.Tensor) -> torch.Tensor:
    """
    Normalize a list of vectors
        Args:
            vectors (torch.Tensor): Input vectors

        Returns:
            torch.Tensor: Normalized vectors
    """
    return F.normalize(vectors, p=2, dim=1)


def cls_pooling(
    outputs: torch.Tensor, inputs: Dict, strategy: str = "cls"
) -> np.ndarray:
    if strategy == "cls":
        outputs = outputs[:, 0]
    elif strategy == "mean":
        outputs = torch.sum(
            outputs * inputs["attention_mask"][:, :, None], dim=1
        ) / torch.sum(inputs["attention_mask"])
    else:
        raise NotImplementedError
    return outputs.detach().cpu()


def euclidean_distance(vector1, vector2):
    """
    Compute the Euclidean distance between two vectors.

    Args:
        vector1 (np.ndarray): First vector.
        vector2 (np.ndarray): Second vector.

    Returns:
        float: Euclidean distance between the two vectors.
    """
    return np.linalg.norm(vector1 - vector2)


 
def manhattan_distance(vector1, vector2):
    """
    Compute the Manhattan distance between two vectors.

    Args:
        vector1 (np.ndarray): First vector.
        vector2 (np.ndarray): Second vector.

    Returns:
        float: Manhattan distance between the two vectors.
    """
    return np.sum(np.abs(vector1 - vector2))