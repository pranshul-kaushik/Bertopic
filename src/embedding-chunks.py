import spaces
from sentence_transformers import SentenceTransformer

embedding_model_name = "BAAI/bge-small-en"
embedding_model = SentenceTransformer(embedding_model_name)

@spaces.GPU()
def embed_splitted_docs(splitted_docs):
    embeddings = embedding_model.encode(splitted_docs, show_progress_bar=True)
    return embeddings