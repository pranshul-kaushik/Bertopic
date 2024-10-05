import gradio as gr

from src.scripts.nlp_processing import embed_splitted_docs, split_corpus
from src.scripts.topic_modeling import topic_modeling
from src.utils.utils import extract_corpus


def greet(fileobj):
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
