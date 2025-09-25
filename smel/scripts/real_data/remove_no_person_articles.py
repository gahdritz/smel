import argparse
from itertools import combinations
import json
import math
import os
import pickle
import random
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
import torch
import transformers
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from urllib.parse import urlparse

from constants import (
    DOMAINS,
    URL_TO_NAME,
)


PICKLE_DIR = "pickles"

os.makedirs(PICKLE_DIR, exist_ok=True)

torch.manual_seed(42)
random.seed(43)

model_name = "meta-llama/Llama-3.3-70B-Instruct"
tokenizer_name = "meta-llama/Llama-3.2-1B-Instruct"

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")

# Load the model and tokenizer
pipeline = transformers.pipeline(
    "text-generation",
    model=model_name,
    #model_kwargs={"quantization_config": quantization_config, "torch_dtype": torch.float16},
    model_kwargs={"torch_dtype": torch.bfloat16, "attn_implementation": "flash_attention_2"},
    tokenizer=tokenizer,
    device_map="auto",
)

# Remove article pairs that do not feature the same person

data_name = "Matched_Articles_Deduplicated_5000samples_score0.7"
df = pd.read_csv(f"Updated_{data_name}_notruncate.csv") 
verifications = []

for idx, row in df.iterrows():
    main_name = row['main_name']

    messages = [
        {"role": "system", "content": "You are an assistant that analyzes descriptions of names. Specifically, given a piece of text, if it is mentioned that no person is mentioned in both articles, or that one person who is mentioned in both could not be determined, then you should output '0', else if only a single name, and nothing else, is the full description then you should output '1'."},
        {"role": "user", "content": f"Description: {main_name}"}
    ]

    outputs = pipeline(
        [messages],
        temperature=1.0,
        batch_size=64,
        max_new_tokens=5,
    )

    verify = [o[0]["generated_text"][-1]["content"] for o in outputs]
    verifications.append(verify[0])

df["verifications"] = verifications
df = df[df["verifications"] == '1']
df.to_csv(f'Removed_Updated_{data_name}_notruncate.csv', index=False)

# # Save the DataFrame as a Pickle file
df.to_pickle(f'Removed_Updated_{data_name}_notruncate.pkl')