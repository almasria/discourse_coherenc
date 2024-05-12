from models import EmbeddingModel
from session import QuerySession
from typing import List
import json


class DataProcessor:
    """
    Class to process data from json file

    Attributes:
        file_path (str): Path to the json file

    Returns:
        None
    """

    def __init__(self, file_path: str = None) -> None:
        """
        Initializes the DataProcessor object

        Args:
            file_path (str): Path to the json file

        Returns:
            None
        """
        self.file_path = file_path

    def _read_json_file(self) -> dict:
        """
        Reads the json file

        Args:
            None

        Returns:
            (dict): A dictionary containing the turns of the dialogue
        """
        with open(self.file_path, "r", encoding="utf8") as file:
            data = json.load(file)
        return data

    def export_sessions(
        self,
        decontextualized: bool = True,
        embedding_model: str = "all-mpnet-base-v2",
        context_window_size: int = 1,
        statistics: str = "mean",
        normalize_embeddings: bool = False,
        metric: str = "cosine"
    ) -> List[QuerySession]:
        """
        Export sessions from the json file

        Args:
            decontextualized (bool): Use decontextualized queries
            embedding_model (str): The name of the embedding model
            context_window_size (int): Context window size
            statistics (str): Statistics to use when computing global coherence score
            normalize_embeddings (bool): Use normalized embeddings
        
        Returns:
            List[QuerySession]: List of QuerySession objects
        """
        data = self._read_json_file()
        # sessions = []
        query_field = "oracle_query" if decontextualized else "query"
        embedding_model = EmbeddingModel(model_name=embedding_model)
        for i, session in enumerate(data):
            queries = [query[query_field] for query in session["turns"]]
            session_obj = QuerySession(
                queries=queries,
                embedding_model=embedding_model,
                context_window_size=context_window_size,
                statistics=statistics,
                normalize_embeddings=normalize_embeddings,
                metric=metric
            )
            # sessions.append(session_obj)
            # if i > 10:
            #     break

            yield session_obj

        # return sessions

    def export_queries(self):
        """
        Export queries from the json file
        
        Args:
            None

        Returns:
            List[List[str]]: List of list of queries
        """
        data = self._read_json_file()
        sessions = []  
        
        for i, session in enumerate(data):
            queries = [query["oracle_query"] for query in session["turns"]]
            sessions.append(queries)
        return sessions