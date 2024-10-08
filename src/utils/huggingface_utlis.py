import os
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict

from src.utils.constants import DATASET_REPO_ID, EMBEDDING_MODEL_NAME, MODEL_REPO_ID
from src.utils.utils import get_timestamp

HF_TOKEN = os.environ.get("HF_TOKEN", None)


def save_dataset_to_hf_hub(topic_info_df, corpus, docs, filename):
    raw_df = pd.DataFrame({"text": corpus})

    intrim_df = pd.DataFrame({"text": docs})

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
