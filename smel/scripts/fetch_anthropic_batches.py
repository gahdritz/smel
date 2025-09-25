import argparse
from itertools import combinations
import json
import os
import pickle
import random

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
import torch
import transformers
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from smel.utils.constants import (
    DOMAINS,
    ENTITIES,
    URL_TO_NAME,
    filter_combinations,
)


MODEL = "claude-3-7-sonnet-20250219"

client = anthropic.Anthropic()

for e in ENTITIES:
    OPENAI_BATCH_DIR = f"openai_batches/{MODEL}"
    RUN_NAME = f"{e}_summaries"
    PICKLE_DIR = f"fact_leakage/pickles"
    PROMPT_DIR = f"{PICKLE_DIR}/{MODEL}_{RUN_NAME}_prompts/"
    BATCH_PATH = f"{PICKLE_DIR}/{MODEL}_{RUN_NAME}_prompts_batch.pickle"
    OUTPUT_DIR = f"{PICKLE_DIR}/{MODEL}_{RUN_NAME}/"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(BATCH_PATH, "rb") as fp:
        batch = pickle.load(fp)

    batch_id = batch.id

    current_batch = client.messages.batches.retrieve(batch_id)
    
    result_dict = {}
    if(current_batch.request_counts.processing == 0):
        assert(current_batch.request_counts.canceled == 0)
        assert(current_batch.request_counts.errored == 0)
        assert(current_batch.request_counts.expired == 0)

        for result in client.messages.batches.results(batch_id):
            result_id = result.custom_id
            result_text = result.result.message.content[0].text
            result_dict[result_id] = result_text
    else:
        print(f"{batch_id} not done yet!")
        continue

    for f in os.listdir(PROMPT_DIR):
        transformed_filename = f.split('_claude')[0].replace('context_', '').replace('filtered_', '')
       
        files = {k:v for k,v in result_dict.items() if transformed_filename in k}

        responses = []
        for i in range(len(files)):
            responses.append(result_dict[f"{transformed_filename}_{i}"])

        with open(os.path.join(OUTPUT_DIR, f), "wb") as fp:
            pickle.dump(list(zip(responses, [[] for _ in responses])), fp, protocol=pickle.HIGHEST_PROTOCOL)


