from typing import List, Union
import numpy as np
import torch
from bert_score import BERTScorer
import seaborn as sns
import matplotlib.pylab as plt
import pandas as pd


class QuerySession:
    """
    """
    def __init__(self, queries: List[str] = None, normalize_embeddings: bool = False, embedding_model: str = None) -> None:
        """
        """
        if queries is not None:
            self.queries: List[str] = queries[:]
            self._embed_queries(queries=queries, embedding_model=embedding_model, normalzied=normalize_embeddings)
               

        self.coherence_score = 0.0

    def add_query(self, query: str) -> None:
        """
        """
        self.queries.append(query)

    def _embed_queries(self, queries: List[str], embedding_model: str = None, normalzied = False) -> None:
        """
        """
        if embedding_model is not None:
            self.embeddings = embedding_model.embed(sentences=queries, normalzied=normalzied)[:]

    def _compute_coh_score(self, embedding_model: str = None) -> None:
        """
        """
        if embedding_model is not None:
            model = EmbeddingModel(model_name=embedding_model)
            self.embeddings = model.embed(sentences=self.queries).detach().numpy()
            self.coh_score = np.corrcoef(self.embeddings)[0, 1]


