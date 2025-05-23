import argparse
from itertools import combinations
import json
import math
import os
import pickle
import random

import matplotlib.pyplot as plt
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


PICKLE_DIR = "../pickles"

os.makedirs(PICKLE_DIR, exist_ok=True)

torch.manual_seed(42)
random.seed(43)

model_name = "meta-llama/Llama-3.3-70B-Instruct"

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
)

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")

pipeline = transformers.pipeline(
    "text-generation",
    model=model_name,
    #model_kwargs={"quantization_config": quantization_config, "torch_dtype": torch.float16},
    model_kwargs={"torch_dtype": torch.bfloat16, "attn_implementation": "flash_attention_2"},
    tokenizer=tokenizer,
    device_map="auto",
)

with open(os.path.join(PICKLE_DIR, "disaster.pickle"), "rb") as fp:
    entity_names = pickle.load(fp)

FACTS_TO_GENERATE = 8

BATCH_SIZE = 10000
pipeline_batch_size = 32

entity_fact_lists = []
all_messages = []
for i, entity_name in enumerate(entity_names):
    if(i % 10 == 0):
        print(i)

    messages = [
#       {"role": "system", "content": f"You are an assistant that makes up facts. Given the name of a fictional U.S. government agency, write at least {FACTS_TO_GENERATE} unique facts about the agency, one per line. Keep the facts short, specific, and completely distinct from each other. Keep it realistic; all of the facts should be plausible."},
#       {"role": "system", "content": f"You are an assistant that makes up facts. Given the name of a fictional \"true crime,\" write at least {FACTS_TO_GENERATE} unique facts about the crime, one per line. Keep the facts short, specific, and completely distinct from each other. Keep it realistic---the facts should be physically plausible."},
       {"role": "system", "content": f"You are an assistant that makes up facts. Given the name of a fictional natural disaster, write at least {FACTS_TO_GENERATE} unique facts about the disaster, one per line. Keep the facts short, specific, and completely distinct from each other. Keep it realistic---the facts should be physically plausible."},
#       {"role": "user", "content": f"Agency: {entity_name}"},
#       {"role": "user", "content": f"True crime: {entity_name}"},
       {"role": "user", "content": f"Disaster: {entity_name}"},
     ]

    all_messages.append(messages)
                
outputs = pipeline(
    all_messages,
    temperature=1.0,
    max_new_tokens=1024,
    batch_size=pipeline_batch_size,
)
output_texts = [o[0]["generated_text"] for o in outputs]

fact_lists = [o[-1]["content"].split('\n') for o in output_texts]

output_path = os.path.join(PICKLE_DIR, f"disaster_fact_lists.pickle")
with open(output_path, "wb") as fp:
    pickle.dump(fact_lists, fp, protocol=pickle.HIGHEST_PROTOCOL)
