import numpy as np
from angle_emb import AnglE, Prompts
import spacy


angle = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1', pooling_strategy='cls').cuda()
angle.set_prompt(prompt=Prompts.C)


nlp = spacy.load("en_core_web_sm")

def cos(v1, v2):
    return np.dot(v1.T, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def get_nouns(sent, model=nlp):
    docs = nlp(sent)
    nouns = [doc.text for doc in docs if doc.pos_ in ['NOUN', 'PROPN']]
    return " ".join(nouns)

def embed(sent):
    return angle.encode({"text": sent})[0]

def similarity(sent1 , sent2, model=angle, type='cos'):
    if type=='cos':
        emb1 = embed(sent1)
        emb2 = embed(sent2)
        return cos(emb1, emb2)
    elif type=='rouge':
        
