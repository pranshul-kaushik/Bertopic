import gradio as gr
import matplotlib.pyplot as plt
import pandas as pd
import spaces
from sentence_transformers import SentenceTransformer

from src.scripts.nlp_processing import embed_splitted_docs, split_corpus
from src.scripts.topic_modeling import topic_modeling
from src.utils.constants import EMBEDDING_MODEL_NAME
from src.utils.utils import extract_corpus

embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)


@spaces.GPU()
def test():
    embeddings = embedding_model.encode(
        ["Test1", "Test2", "Test3"], show_progress_bar=True
    )
    print(":" * 10 + " TEST " + "*" * 10)
    print(embeddings)


def greet(fileobj):
    test()

    # Read the file
    corpus = extract_corpus(fileobj)

    # Split the corpus
    splitted_docs = split_corpus(corpus)

    # Embed the splitted documents
    embeddings = embed_splitted_docs(splitted_docs)

    # Topic modeling
    fig, df = topic_modeling(splitted_docs, embeddings)

    # Save the figure
    return (fig, df)


demo = gr.Interface(fn=greet, inputs="file", outputs=[gr.Image(), gr.Dataframe()])
demo.launch()
