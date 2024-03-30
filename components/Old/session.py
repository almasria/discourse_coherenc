from typing import List
import numpy as np
import torch
from bert_score import BERTScorer
import seaborn as sns
import matplotlib.pylab as plt
import pandas as pd


class Discourse:
    def __init__(self, session: List[str]) -> None:
        self.session = session[:]
        self.scorer = BERTScorer(
            lang="en",
            rescale_with_baseline=False,
            model_type="microsoft/deberta-xlarge-mnli",
        )

        self._compute_topic_coherence()

    def _compute_topic_coherence(self):

        session_size = len(self.session)
        similarities = torch.zeros((session_size, session_size), dtype=torch.float32)
        flow = torch.zeros((session_size - 1), dtype=torch.float32)

        if self.session is not None:
            for i, entry_i in enumerate(self.session):
                for j, entry_j in enumerate(self.session):
                    f1 = self.scorer.score([entry_i], [entry_j])[2].item()
                    similarities[i, j] = f1
                    if j == i + 1:
                        flow[i] = f1

        self.topic_coherence_matrix = similarities
        self.flow_coherence_list = flow
        self.flow_coherence_value = flow.mean()
        similarities_mean = similarities.clone()
        similarities_mean.diagonal(dim1=-1, dim2=-2).zero_()
        self.topic_coherence_value = similarities_mean.mean()

    def add_entry(self, entry: str):
        if entry and self.topic_coherence_matrix is not None:
            new_row = torch.zeros((1, len(self.session)))
            new_column = torch.zeros((len(self.session) + 1, 1))
            for i, entry_i in enumerate(self.session):
                f1 = self.scorer.score([entry_i], [entry])[2].item()
                new_row[0, i] = f1
                new_column[i, 0] = f1
            new_column[-1] = 1.0
            self.topic_coherence_matrix = torch.cat(
                (self.topic_coherence_matrix, new_row), 0
            )
            self.topic_coherence_matrix = torch.cat(
                (self.topic_coherence_matrix, new_column), 1
            )

            similarities_mean = self.topic_coherence_matrix.clone()
            similarities_mean.diagonal(dim1=-1, dim2=-2).zero_()
            self.topic_coherence_value = similarities_mean.mean()
            self.flow_coherence_list = torch.cat(
                (
                    self.flow_coherence_list,
                    torch.tensor([self.topic_coherence_matrix[-2, -1].item()]),
                ),
                0,
            )
            self.flow_coherence_value = self.flow_coherence_list.mean()

    def plot_topic_coherence(self):
        ax = sns.heatmap(self.topic_coherence_matrix, linewidth=0.5, cmap="mako", annot=True, vmin=0.3, vmax=1)
        plt.show()  

    def plot_flow_coherence(self):
        ax = sns.lineplot(data=pd.DataFrame(self.flow_coherence_list.numpy()))
        plt.show()    

