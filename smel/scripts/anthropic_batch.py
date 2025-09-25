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
)

from smel.utils.constants import (
    DOMAINS,
    ENTITIES,
    URL_TO_NAME,
)
from smel.utils.utils import (
    filter_combinations,
)


MODEL = "claude-3-7-sonnet-20250219"

client = anthropic.Anthropic()

for e in ENTITIES:
    OPENAI_BATCH_DIR = f"openai_batches/{MODEL}"
    RUN_NAME = f"{e}_summaries"
    PICKLE_DIR = f"pickles"
    PROMPT_DIR = f"{PICKLE_DIR}/{MODEL}_{RUN_NAME}_prompts/"
    OUTPUT_PATH = f"{PICKLE_DIR}/{MODEL}_{RUN_NAME}_prompts_batch.pickle"

    requests = []
    for f in os.listdir(PROMPT_DIR):
        with open(os.path.join(PROMPT_DIR, f), "rb") as fp:
            msgs = pickle.load(fp)

        for i, chat in enumerate(msgs):
            system=None
            if(chat[0]["role"] == "system"):
                system = chat[0]["content"]
                chat = chat[1:]

            custom_id = f"{f.split('_claude')[0].replace('context_', '').replace('filtered_', '')}_{i}"
            request = Request(
                custom_id=custom_id,
                params=MessageCreateParamsNonStreaming(
                    model=MODEL,
                    max_tokens=1024,
                    messages=chat,
                    system=system,
                ),
            )

            requests.append(request)
    
    message_batch = client.messages.batches.create(
        requests=requests,
    )

    print(message_batch)

    with open(OUTPUT_PATH, "wb") as fp:
        pickle.dump(message_batch, fp, protocol=pickle.HIGHEST_PROTOCOL)

