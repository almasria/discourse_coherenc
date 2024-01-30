from typing import List
from transformers import AutoTokenizer, AutoModel, BertModel
import torch

class EmbeddingModel:
    '''
    A class to wrap the embedding model to be used

    Attributes:

    '''

    allowed_models = {"bert": ["bert-base-uncased", "bert-base-uncased"]} 


    def __init__(self, model_name: str="bert") -> None:
        '''

        '''  

        if model_name in self.allowed_models.keys():
            if model_name == "bert":
                self.tokenizer = AutoTokenizer.from_pretrained(self.allowed_models[model_name][0], )
                self.model = BertModel.from_pretrained(self.allowed_models[model_name][1])            
        else:
            raise Exception("Model not supported!")        

        
    def encode(self, sentences: List[str]=None):

        if sentences == None:            
            return
        
        # Tokenize sentences
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # Return the last hidden
        return model_output.last_hidden_state