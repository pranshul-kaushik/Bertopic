import spaces
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from src.utils.constants import EMBEDDING_MODEL_NAME

embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)


@spaces.GPU()
def embed_splitted_docs(splitted_docs):
    """
    Encode the given list of documents using the specified embedding model.

    Parameters
    ----------
    splitted_docs : List of str
        The list of documents to be embedded.

    Returns
    -------
    embeddings : List of numpy.ndarray
        The embeddings of the given documents.
    """
    embeddings = embedding_model.encode(splitted_docs, show_progress_bar=True)
    return embeddings


def split_corpus(corpus, chunk_size=1000):
    """
    Split a given corpus into chunks of a given size.

    Parameters
    ----------
    corpus : List of str
        The corpus to be split.
    chunk_size : int, default=1000
        The size of the chunks to be split from the corpus.

    Returns
    -------
    List of str
        The list of chunks (splitted documents) from the corpus.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=0, add_start_index=True
    )
    splitted_docs = splitter.create_documents(corpus)
    splitted_docs = list(map(lambda x: x.page_content, splitted_docs))

    return splitted_docs
