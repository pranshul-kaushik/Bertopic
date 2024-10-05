import os

import matplotlib.pyplot as plt
import numpy as np
import spaces
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from cuml.cluster import HDBSCAN
from cuml.manifold import UMAP
from cuml.preprocessing import normalize


@spaces.GPU()
def topic_modeling(
    docs,
    embeddings,
    embedding_model,
    n_gram_range=(3, 6),
    mmr_diversity=1,
    mmr_top_n_words=30,
    keybert_top_n_words=50,
    random_state=42,
    min_cluster_size=15,
):
    """
    Perform topic modeling on a list of documents and their embeddings.

    Parameters
    ----------
    docs : List of str
        The list of documents to be topic modeled.
    embeddings : List of numpy.ndarray
        The list of embeddings of the given documents.
    embedding_model : SentenceTransformer
        The embedding model used to generate the embeddings.
    n_gram_range : Tuple of int, optional
        The range of n-grams to be considered. Defaults to (3, 6).
    mmr_diversity : float, optional
        The diversity value of the MMR model. Defaults to 1.
    mmr_top_n_words : int, optional
        The number of top words to be considered in the MMR model. Defaults to 30.
    keybert_top_n_words : int, optional
        The number of top words to be considered in the KeyBERT model. Defaults to 50.
    random_state : int, optional
        The random seed for reproducibility. Defaults to 42.
    min_cluster_size : int, optional
        The minimum size of a cluster to be considered as a topic. Defaults to 15.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The datamap of the topic modeling.
    topic_info_df : pandas.DataFrame
        The topic information dataframe.
    """
    representation_model = [
        KeyBERTInspired(top_n_words=keybert_top_n_words, random_state=random_state),
        MaximalMarginalRelevance(diversity=mmr_diversity, top_n_words=mmr_top_n_words),
    ]

    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        low_memory=False,
        random_state=random_state,
    )

    topic_model = BERTopic(
        embedding_model=embedding_model,
        representation_model=representation_model,
        n_gram_range=n_gram_range,
        hdbscan_model=hdbscan_model,
        umap_model=umap_model,
        verbose=True,
    ).fit(docs, embeddings=embeddings)

    fig = topic_model.visualize_document_datamap(docs=docs)

    topic_info_df = topic_model.get_topic_info()

    return fig, topic_info_df
