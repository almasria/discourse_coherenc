# from typing import Self
from embedding_model import EmbeddingModel
from similarity_metric import SimilarityMetric
import numpy as np



class Discourse:
    '''
    A class representing a discourse

    Attributes:

    '''

    def __init__(self, sentences: list[str]=None) -> None:
        '''
        A constructor for the Discourse class

        Parameters:
            sentences (List[str]): A list of sentences that make up the discourse
        '''
        self.sentences = sentences  
        self.coherence_scores = {}

    def add_sentence(self, sentence: str=None):
        '''
        A method to add a sentence to the discourse

        Parameters:

        '''
        if sentence is None:
            return self
        if self.sentences is not None:
            self.sentences.append(sentence)
            return self
        self.sentences.append(sentence)
        return self
    
    def coherence_score(self, metric):
        '''
        '''

        computed_metrics = metric.compute(predictions=self.sentences[1:], references=self.sentences[:-1], lang="en")


        return np.mean(computed_metrics['f1'])
                       

    def update_score(self):
        pass

    def plot_coherence(self):
        pass
        
