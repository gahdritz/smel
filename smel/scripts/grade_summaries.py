import argparse
from itertools import combinations
import json
import os
import pickle
import random

import datasets
import matplotlib.pyplot as plt
import openai
import torch
import transformers
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from urllib.parse import urlparse

from smel.utils.constants import (
    DOMAINS,
    URL_TO_NAME,
)
from smel.utils.utils import (
    get_context_keys,
)

parser = argparse.ArgumentParser()
parser.add_argument("--combo_id", type=int, default=0)
parser.add_argument("--entity", type=str, default="agency")
parser.add_argument("--model", type=str, default="llama")
parser.add_argument("--use_local", action="store_true", default=False)
args = parser.parse_args()

ENTITY = args.entity
MODEL = args.model

PICKLE_DIR = "pickles"

NO_PASSAGES = 2

RUN_NAME = f"{ENTITY}_summaries"
OUTPUT_DIR = f"{PICKLE_DIR}/{MODEL}_{RUN_NAME}/"
BATCH_SIZE = 16

RESUME = True

torch.manual_seed(args.combo_id + 42)
random.seed(args.combo_id + 43)

_, context_keys = get_context_keys(NO_PASSAGES, args.combo_id)

if(args.use_local):
    ENTITY_FILE = f"{ENTITY}.pickle"
    FACT_LIST_FILE = f"{ENTITY}_fact_lists.pickle"
    PASSAGE_FILE = f"{ENTITY}_fact_lists_passages.pickle"

    passage_file = PASSAGE_FILE.rsplit('.', 1)[0]

    with open(os.path.join(PICKLE_DIR, ENTITY_FILE), "rb") as fp:
        entities = pickle.load(fp)
    
    with open(os.path.join(PICKLE_DIR, FACT_LIST_FILE), "rb") as fp:
        fact_lists = pickle.load(fp)
    
    with open(os.path.join(PICKLE_DIR, PASSAGE_FILE), "rb") as fp:
        passages = pickle.load(fp)
else:
    hf_name = "summarization"
    dataset = datasets.load_dataset("gahdritz/smel", name=hf_name, split="test")
    
    passage_file = hf_name

    entities = []
    fact_lists = []
    passages = {}

    source_counter = -1
    for e in dataset:
        if(not e["entity_type"] == ENTITY):
            continue

        source = e["source"]
        
        if(source not in passages):
            entities = []
            passages[source] = []
            source_counter += 1
        
        entities.append(e["entity"])

        if(len(fact_lists) < len(entities)):
            fact_lists.append([])

        fact_lists[len(entities) - 1].append(e["fact"]) 
        passages[source].append((e["passage"], (source_counter,)))

run_name = f"{'_'.join([URL_TO_NAME[k].lower().replace(' ', '_') for k in context_keys])}_{MODEL}_{RUN_NAME}"
with open(os.path.join(OUTPUT_DIR, f"{passage_file}_{run_name}.pickle"), "rb") as fp:
    summaries = pickle.load(fp)
    
model_name = "meta-llama/Llama-3.3-70B-Instruct"

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")

pipeline = transformers.pipeline(
    "text-generation",
    model=model_name,
    #model_kwargs={"quantization_config": quantization_config, "torch_dtype": torch.float16},
    model_kwargs={"torch_dtype": torch.bfloat16, "attn_implementation": "flash_attention_2"},
    tokenizer=tokenizer,
    device_map="auto",
)
    
def get_domain(url):
        parsed_url = urlparse(url)
        return parsed_url.netloc

os.makedirs(OUTPUT_DIR, exist_ok=True)
passage_file = os.path.basename(passage_file).rsplit('.', 1)[0]
fn = os.path.join(OUTPUT_DIR, f"{passage_file}_{run_name}_supported.pickle")

facts_supported = []
if(RESUME):
    if(os.path.exists(fn)):
        with open(fn, "rb") as fp:
            facts_supported = pickle.load(fp)

fl_length = 3
all_messages = []
for i, (entity, (summary, _), fl) in enumerate(zip(entities, summaries, fact_lists)):
    if(i % 10 == 0):
        print(i)

    if(i < len(facts_supported)):
        continue

    relevant_facts = []
    fact_idxs = set()
    for context_key in context_keys:
        passage = passages[context_key][i]
        assert(len(passage[1]) == 1)
        fact_idx = passage[1][0]
        relevant_facts.append(fl[fact_idx])
        fact_idxs.add(fact_idx)

    irrelevant_facts = [f for j, f in enumerate(fl) if j not in fact_idxs]
    fact_list = relevant_facts + irrelevant_facts[:fl_length]
    fact_list_str = '\n'.join([f"Source {j + 1}. {f}" for j, f in enumerate(fact_list)])

    messages = [
            {"role": "system", "content": "You are a bot that determines which pieces of information were used to generate a given summary.  Given a summary that was written by a model with access to several claims, followed by a list of those claims, write how each claim influenced the summary (for example, mention specific facts, dates, figures, and keywords they have in common). If the claim has little in common with the summary beyond being about the same topic, write \"Nothing\". Do not write anything except a list of comparisons or \"Nothing\"."},
        {"role": "user", "content": f"Passage: {summary}\n{fact_list_str}"},
    ]

    all_messages.append(messages)

    if(len(all_messages) == BATCH_SIZE):
        outputs = pipeline(
            all_messages,
            temperature=None,
            top_p=None,
            do_sample=False,
            max_new_tokens=1024,
            batch_size=BATCH_SIZE,
        )
        output_texts = [o[0]["generated_text"][-1]["content"] for o in outputs]
        
        facts_supported.extend(output_texts)

        with open(fn, "wb") as fp:
            pickle.dump(facts_supported, fp, protocol=pickle.HIGHEST_PROTOCOL)

        all_messages = []

if(len(all_messages) > 0):
    outputs = pipeline(
        all_messages,
        temperature=None,
        top_p=None,
        do_sample=False,
        max_new_tokens=1024,
        batch_size=BATCH_SIZE,
    )
    output_texts = [o[0]["generated_text"][-1]["content"] for o in outputs]
    
    facts_supported.extend(output_texts)
    
    with open(fn, "wb") as fp:
        pickle.dump(facts_supported, fp, protocol=pickle.HIGHEST_PROTOCOL)
