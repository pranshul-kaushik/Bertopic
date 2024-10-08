import os
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict

from src.utils.utils import get_timestamp
from utils.constants import DATASET_REPO_ID, EMBEDDING_MODEL_NAME, MODEL_REPO_ID

HF_TOKEN = os.environ.get("HF_TOKEN", None)


def save_model_to_hf_hub(topic_model, filename):
    topic_model.push_to_hf_hub(
        repo_id=MODEL_REPO_ID,
        commit_message=f"{get_timestamp()} - {filename}",
        token=HF_TOKEN,
        private=True,
        serialization="safetensors",
        save_embedding_model=EMBEDDING_MODEL_NAME,
        save_ctfidf=True,
    )


def save_dataset_to_hf_hub(topic_model, corpus, docs, filename):
    raw_df = pd.DataFrame({"text": corpus})

    intrim_df = pd.DataFrame({"text": docs})

    topic_info_df = topic_model.get_topic_info()

    dataset = DatasetDict(
        {
            "input": Dataset.from_pandas(raw_df),
            "processed": Dataset.from_pandas(intrim_df),
            "output": Dataset.from_pandas(topic_info_df),
        }
    )

    dataset.push_to_hub(
        DATASET_REPO_ID + f"{Path(filename).stem}-{get_timestamp()}",
        private=True,
        token=HF_TOKEN,
    )

    return topic_info_df
