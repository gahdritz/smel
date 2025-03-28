import argparse
from itertools import combinations
import json
import os
import pickle
import random

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

from constants import (
    DOMAINS,
    URL_TO_NAME,
)

PICKLE_DIR = "../pickles/"
AGENCY_FILE = "agencies.pickle"
FACT_LIST_FILE = "agency_fact_lists.pickle"
PASSAGE_FILE = "agency_fact_lists_passages.pickle"

NO_PASSAGES = 2

C4_JSONL = None
C4_JSONL = "../scratch/c4-0000.json"
NO_C4_DOCUMENTS = 15 - NO_PASSAGES 
#NO_C4_DOCUMENTS = 0
MODEL = "openai_gpt-4o"
RUN_NAME = "agency_summaries"
OUTPUT_DIR = f"pickles/{MODEL}_{RUN_NAME}/"

RESUME = True

os.makedirs(OUTPUT_DIR, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--combo_id", type=int, default=0)
args = parser.parse_args()

torch.manual_seed(args.combo_id + 42)
random.seed(args.combo_id + 43)

domain_combinations = list(combinations([url for _, url in DOMAINS], NO_PASSAGES))
context_keys = list(domain_combinations[args.combo_id])

run_name = f"{'_'.join([URL_TO_NAME[k].lower().replace(' ', '_') for k in context_keys])}_{MODEL}_{RUN_NAME}"
fact_list_file = os.path.basename(FACT_LIST_FILE).rsplit('.', 1)[0]
with open(os.path.join(OUTPUT_DIR, f"{fact_list_file}_{run_name}.pickle"), "rb") as fp:
    summaries = pickle.load(fp)
    
model_name = "meta-llama/Llama-3.3-70B-Instruct"

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
)

pipeline = transformers.pipeline(
    "text-generation",
    model=model_name,
    #model_kwargs={"quantization_config": quantization_config, "torch_dtype": torch.float16},
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)
    
with open(os.path.join(PICKLE_DIR, AGENCY_FILE), "rb") as fp:
    agencies = pickle.load(fp)

with open(os.path.join(PICKLE_DIR, FACT_LIST_FILE), "rb") as fp:
    fact_lists = pickle.load(fp)

with open(os.path.join(PICKLE_DIR, PASSAGE_FILE), "rb") as fp:
    passages = pickle.load(fp)

if(C4_JSONL is not None):
    with open(C4_JSONL, "r") as fp:
        jsons = fp.readlines()

    c4_documents = [json.loads(j) for j in jsons]

def get_domain(url):
        parsed_url = urlparse(url)
        return parsed_url.netloc

fact_list_file = os.path.basename(FACT_LIST_FILE).rsplit('.', 1)[0]
fn = os.path.join(OUTPUT_DIR, f"{fact_list_file}_{run_name}_supported.pickle")

facts_supported = []
if(RESUME):
    if(os.path.exists(fn)):
        with open(fn, "rb") as fp:
            facts_supported = pickle.load(fp)

fl_length = 3
for i, (agency, (summary, _), fl) in enumerate(zip(agencies, summaries, fact_lists)):
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
            {"role": "system", "content": "You are a bot that determines which pieces of information were used to generate a given summary.  Given a summary of a government agency that was written by a model with access to several claims, followed by a list of those claims, write how each claim influenced the summary (for example, mention specific facts, dates, figures, and keywords they have in common). If the claim has little in common with the summary beyond being about the same government agency, write \"Nothing\". Do not write anything except a list of comparisons or \"Nothing\"."},
        {"role": "user", "content": f"Passage: {summary}\n{fact_list_str}"},
    ]
 
    outputs = pipeline(
        messages,
#        temperature=0.,
        temperature=None,
        top_p=None,
        do_sample=False,
        max_new_tokens=1024,
    )
    output_text = outputs[0]["generated_text"][-1]["content"]

    facts_supported.append(output_text)

    with open(fn, "wb") as fp:
        pickle.dump(facts_supported, fp, protocol=pickle.HIGHEST_PROTOCOL)
