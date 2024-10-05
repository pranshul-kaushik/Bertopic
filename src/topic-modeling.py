import os

import matplotlib.pyplot as plt
import numpy as np
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, LangChain, MaximalMarginalRelevance
from cuml.cluster import HDBSCAN
from cuml.manifold import UMAP
from cuml.preprocessing import normalize
from langchain_text_splitters.character import RecursiveCharacterTextSplitter

n_gram_range = (3,6)
mmr_diversity = 1
mmr_top_n_words = 30
keybert_top_n_words = 50
random_state = 42


representation_model = [
    KeyBERTInspired(top_n_words=keybert_top_n_words, random_state=random_state), 
    MaximalMarginalRelevance(diversity=mmr_diversity, top_n_words = mmr_top_n_words), 
]

chunk_size = 1000

splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0, add_start_index=True)
splitted_docs = splitter.create_documents(corpus)
splitted_docs = list(map(lambda x: x.page_content, splitted_docs))

