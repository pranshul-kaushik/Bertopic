import gradio as gr

from src.scripts.nlp_processing import embed_splitted_docs, split_corpus
from src.scripts.topic_modeling import topic_modeling
from src.utils.huggingface_utlis import save_dataset_to_hf_hub
from src.utils.utils import extract_corpus


def greet(fileobj):
    # Read the file
    corpus, filename = extract_corpus(fileobj)

    # Split the corpus
    splitted_docs = split_corpus(corpus)

    # Embed the splitted documents
    embeddings = embed_splitted_docs(splitted_docs)

    # Topic modeling
    df = topic_modeling(filename, splitted_docs, embeddings)

    # Save
    save_dataset_to_hf_hub(df, corpus, splitted_docs, filename)

    # Save the figure
    return df


demo = gr.Interface(fn=greet, inputs="file", outputs=[gr.Dataframe()])
demo.launch()
