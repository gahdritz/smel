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
tokenizer_name = "meta-llama/Llama-3.2-1B-Instruct"

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
)

# Load the model and tokenizer
pipeline = transformers.pipeline(
    "text-generation",
    model=model_name,
    #model_kwargs={"quantization_config": quantization_config, "torch_dtype": torch.float16},
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

with open(os.path.join(PICKLE_DIR, "agencies.pickle"), "rb") as fp:
    agency_names = pickle.load(fp)

FACTS_TO_GENERATE = 10

agency_fact_lists = []
for i, agency_name in enumerate(agency_names):
    if(i % 10 == 0):
        print(i)

    fact_list = []
    while(len(fact_list) < FACTS_TO_GENERATE):    
        messages = [
            {"role": "system", "content": f"You are a bot that makes up facts. Given the name of a fictional U.S. government agency, write at least {FACTS_TO_GENERATE} unique facts about the agency, one per line. Keep the facts short, specific, and completely distinct from each other."},
            {"role": "user", "content": f"Agency: {agency_name}"}
        ]
                
        outputs = pipeline(
            messages,
            temperature=1.0,
            max_new_tokens=256,
        )
        output_text = outputs[0]["generated_text"]

        fact_list.extend(output_text[-1]["content"].split('\n')) 

    agency_fact_lists.append(fact_list)

output_path = os.path.join(PICKLE_DIR, "agency_fact_lists.pickle")
with open(output_path, "wb") as fp:
    pickle.dump(agency_fact_lists, fp, protocol=pickle.HIGHEST_PROTOCOL)
