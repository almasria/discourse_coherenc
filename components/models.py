from transformers import AutoTokenizer, AutoModel
import torch
from typing import List
from utils import mean_pooling, normalize_vectors


class EmbeddingModel:
    """
    Class to load a sentence embedding model from the HuggingFace model hub

    Args:
        model_name (str): Model name to load from the HuggingFace model hub

    Returns:
        None
    """

    allowed_models = ["all-mpnet-base-v2", "all-MiniLM-L6-v2"]

    def __init__(self, model_name: str = "all-mpnet-base-v2") -> None:
        """
        Initialize the model and tokenizer

        Args:
            model_name (str): Model name to load from the HuggingFace model hub

        Returns:
            None
        """

        if model_name in self.allowed_models:
            self.tokenizer = AutoTokenizer.from_pretrained(
                "sentence-transformers/" + model_name
            )
            self.model = AutoModel.from_pretrained(
                "sentence-transformers/" + model_name
            )
        else:
            raise Exception("Model not supported!")

    def embed(
        self,
        sentences: List[str] = None,
        normalize_embeddings: bool = True,
        return_tensors: bool = False,
    ) -> torch.Tensor:
        """
        Embed a list of sentences

            Args:
                sentences (List[str]): List of sentences to embed
                normalzied (bool): Normalize embeddings
                return_tensors (bool): Return embeddings as tensors

            Returns:
                torch.Tensor: Embeddings for the input sentences

        """
        if sentences is not None:
            # Tokenize sentences
            encoded_input = self.tokenizer(
                sentences, padding=True, truncation=True, return_tensors="pt"
            )
            # Compute token embeddings
            with torch.no_grad():
                model_output = self.model(**encoded_input)
            # Perform pooling
            sentence_embeddings = mean_pooling(
                model_output, encoded_input["attention_mask"]
            )

            if normalize_embeddings:
                # Normalize embeddings
                sentence_embeddings = normalize_vectors(sentence_embeddings)

            if not return_tensors:
                sentence_embeddings = sentence_embeddings.numpy()

            return sentence_embeddings
