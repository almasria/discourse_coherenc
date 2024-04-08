from typing import List, Union
import numpy as np
from utils import compute_statistics, cos_sim
from torch import nn
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline


class QuerySession:
    """ """

    def __init__(
        self,
        queries: List[str] = None,        
        embedding_model: nn.Sequential = None,
        context_window_size: int = 1,
        statistics: str = "mean",
        normalize_embeddings: bool = False
    ) -> None:
        """
        Initialize the QuerySession object
            Args:
                queries (List[str]): List of queries
                normalize_embeddings (bool): Use normalized embeddings
                embedding_model (str): Sentence embedding model
                context_window_size (int): Context window size
                statistics (str): Statistic to use when computing global coherence score

            Returns:
                None
        """
        if queries is not None:
            self.queries = queries[:]
            self.session_size = len(queries)
            self.embedding_model = embedding_model
            self.normalized_session = normalize_embeddings
            self.embeddings = self._embed_queries(
                queries=self.queries,
                embedding_model=self.embedding_model,
                normalize_embeddings=self.normalized_session,
            )
            self.global_coherence_score, self.local_coherence_scores = (
                self.compute_global_coherence(
                    context_window=context_window_size, statistics=statistics
                )
            )
            self.statistics = statistics
            self.context_window_size = context_window_size

    def _embed_queries(
        self,
        queries: List[str],
        embedding_model: str = None,
        normalize_embeddings: bool = False,
    ) -> np.ndarray:
        """
        Embed the list of queries
            Args:
                queries (List[str]): List of queries as string
                embedding_model (str): Sentence embedding model
                normalize_embeddings (bool): Use normalized embeddings

            Returns:
                np.ndarray: Embedded queries
        """
        if embedding_model is not None:
            return embedding_model.embed(
                sentences=queries, normalize_embeddings=normalize_embeddings
            )

    def _add_query(self, query: str = None) -> None:
        """
        Add a query to the session
            Args:
                query (str): Query to add to the session as a string

            Returns:
                None
        """
        if query is not None:
            self.queries.append(query)
            self.session_size += 1
            query_embedding = self.embedding_model.embed(
                sentences=[query], normalize_embeddings=self.normalized_session
            )
            self.embeddings = np.concatenate([self.embeddings, query_embedding])
            context_vector = self._compute_context_vector(
                position=self.session_size - 1, context_window=self.context_window_size
            )
            local_coherence_score = self._compute_local_coherence(
                current_vector=query_embedding[0], neighbor_vector=context_vector
            )

            self.local_coherence_scores.append(local_coherence_score)

            self.global_coherence_score = compute_statistic(
                numbers=self.local_coherence_scores, statistic=self.statistic
            )

    def add_queries(self, queries: Union[str, List[str]]) -> None:
        """
        Add queries to the session
            Args:
                queries (Union[str, List[str]]): Query or list of queries to add to the session

            Returns:
                None
        """
        if isinstance(queries, str):
            self._add_query(query=queries)
        elif isinstance(queries, list):
            for query in queries:
                self._add_query(query=query)
        else:
            raise ValueError("Invalid input type. Expected str or list of str.")

    def _compute_context_vector(
        self, position: int = 1, context_window: int = 1
    ) -> np.ndarray:
        """
        Compute the context vector for a given query in the session
            Args:
                position (int): Position of the query in the session
                context_window (int): Context window size
            Returns:
                np.ndarray: Context vector as a numpy array
        """
        if context_window >= self.session_size:
            raise ValueError("Context window is too big")

        if position < context_window:
            context_vector = np.sum(self.embeddings[:position], axis=0)
            context_vector = context_vector / np.linalg.norm(context_vector)

        else:
            context_vector = np.sum(
                self.embeddings[position - context_window : position], axis=0
            )
            context_vector = context_vector / np.linalg.norm(context_vector)

        return context_vector

    def _compute_local_coherence(
        self, current_vector: np.ndarray = None, neighbor_vector: np.ndarray = None
    ) -> float:
        """
        Compute the local coherence between a query and its context vector
            Args:
                current_vector (np.ndarray): Current query embedding
                neighbor_vector (np.ndarray): Context vector

            Returns:
                float: Local coherence score
        """
        if current_vector is not None and neighbor_vector is not None:
            return cos_sim(current_vector, neighbor_vector)

    def compute_global_coherence(
        self, context_window: int = 1, statistics: str = "mean"
    ) -> float:
        """
        Compute the global coherence score for the session
            Args:
                context_window (int): Context window size
                statistics (str): Statistic to use when computing the global coherence score

            Returns:
                float: Global coherence score
        """
        if self.session_size > 1:

            local_coherence_scores = []

            for i, embedding in enumerate(self.embeddings):
                if i == 0:
                    continue
                context_vector = self._compute_context_vector(
                    position=i, context_window=context_window
                )

                if self.embedding_model == "mxbai-embed-large-v1":
                    embedding = self.embedding_model._transform_query(
                        self.queries[i]
                    )
                    embedding = self.embedding_model.embed(
                        sentences=[embedding], normalize_embeddings=self.normalized_session
                    )

                local_coherence_scores.append(
                    self._compute_local_coherence(
                        current_vector=embedding, neighbor_vector=context_vector
                    )
                )

        return (
            [compute_statistics(numbers=local_coherence_scores, statistic=statistics)],
            local_coherence_scores,
        )

    def plot_local_coherence(self, smooth: bool=True, annotated: bool = True) -> None:
        """
        Plot the local coherence scores for the session
            Args:
                None

            Returns:
                None
        """

        X = list(range(1, self.session_size))
        Y = self.local_coherence_scores

        if smooth:
            X_Y_Spline = make_interp_spline(X, Y)

            X = np.linspace(min(X),max(X), 500)
            Y = X_Y_Spline(X)

            # X_new = np.linspace(min(X), max(X), 300)
            # spl = make_interp_spline(X, Y, k=3)
            # Y_smooth = spl(X_new)
            # plt.plot(X_new, Y_smooth)


        plt.plot(X, Y)

        # # Add text annotations to the plot
        # if annotated:              
        #     X_ = self.session_size
        #     Y_ = self.local_coherence_scores     
        #     for i in range(1, X_):
        #         plt.text(X_[i], Y_[i], f"{Y_[i]:.2f}", ha='left', va='center', bbox=dict(facecolor='white', edgecolor='black'))
        plt.xlabel("Query Position")
        plt.ylabel("Local Coherence Score")
        plt.title("Local Coherence Scores")
        plt.show()


    def print_queries(self) -> None:
        """
        Print the queries in the session
            Args:
                None

            Returns:
                None
        """
        for i,query in enumerate(self.queries):
            print(i, query)


        