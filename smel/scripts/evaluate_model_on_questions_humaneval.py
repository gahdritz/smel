import argparse
from itertools import combinations
import json
import os
import pickle
import random

import anthropic
import datasets
import google.generativeai as google_genai
from google.api_core import exceptions
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
    filter_combinations,
    get_context_keys,
)

#Gemma:
import torch._dynamo as dynamo
dynamo.config.cache_size_limit = 128  # larger than 8; set before compiling/loading

parser = argparse.ArgumentParser()
parser.add_argument("--combo_id", type=int, default=0)
parser.add_argument("--entity", type=str, default="agency")
parser.add_argument("--model", type=str, default="llama")
parser.add_argument("--stop_after_k", type=int, default=None)
parser.add_argument("--use_local", action="store_true", default=False)
args = parser.parse_args()

ENTITY = args.entity
MODEL = "human_eval"

WRITE_TO_DISK = False
C4_JSONL = None
C4_JSONL = "smel/data/c4-0000.json"
NO_C4_DOCUMENTS = 14
GOOGLE_SLEEP=60
#NO_C4_DOCUMENTS = 0
OPENAI_BATCH = False
OPENAI_BATCH_DIR = f"openai_batches/{MODEL}_{ENTITY}"
RUN_NAME = f"{ENTITY}_ignoring"
PICKLE_DIR = f"pickles"
OUTPUT_DIR = f"{PICKLE_DIR}/{MODEL}_{RUN_NAME}/"
PROMPT_DIR = f"{PICKLE_DIR}/{MODEL}_{RUN_NAME}_prompts/"

torch.manual_seed(args.combo_id + 42)
random.seed(args.combo_id + 43)


if(args.use_local):
    QUESTION_FILE = f"{ENTITY}_questions_filtered_notruncate.pickle"
    CONTEXT_FILES = [
        f"{ENTITY}_questions_filtered_rewritten_notruncate.pickle",
        f"{ENTITY}_questions_filtered_{ENTITY}_questions_filtered_rewritten_corrupted_notruncate.pickle",
    ]
    no_docs = len(CONTEXT_FILES)

    with open(os.path.join(PICKLE_DIR, QUESTION_FILE), "rb") as fp:
        questions = pickle.load(fp)
   
    context_keys = get_context_keys(no_docs, args.combo_id) #For using built in combinations

    context_files = []
    for context_file, context_key in zip(CONTEXT_FILES, context_keys):
        with open(os.path.join(PICKLE_DIR, context_file), "rb") as fp:
            d = pickle.load(fp)
            print(d.keys())
            # print(context_file)
            contexts = d[context_key]
            
            assert(len(contexts) == len(questions))
            context_files.append(contexts)

    question_file = os.path.basename(QUESTION_FILE).rsplit('.', 1)[0]
else:
    datasets = [
        datasets.load_dataset("gahdritz/smel", name="qa_rewritten", split="test"),
        datasets.load_dataset("gahdritz/smel", name="qa_corrupted", split="test"),
    ]
    no_docs = len(datasets)
    
    context_keys = get_context_keys(no_docs, args.combo_id)
 
    context_files = []
    for d, context_key in zip(datasets, context_keys):
        questions = [(None, '\n'.join([e["question"], e["answer"]])) for e in datasets[0] if e["source"] == context_key and e["entity"] == ENTITY]
        contexts = [(e["passage"], None) for e in d if e["source"] == context_key and e["entity"] == ENTITY]
        assert(len(contexts) == len(questions))
        context_files.append(contexts)

    question_file = "qa_rewritten"

os.makedirs(OUTPUT_DIR, exist_ok=True)

run_name = f"{'_'.join([URL_TO_NAME[k].lower().replace(' ', '_') for k in context_keys])}_{MODEL}_{RUN_NAME}"

if(C4_JSONL is not None):
    with open(C4_JSONL, "r") as fp:
        jsons = fp.readlines()

    c4_documents = [json.loads(j) for j in jsons]

def parse_question(question):
    # print(question)
    q, a = question.split('\n')
    q = q.strip()
    a = a.strip()
    return q, a

def get_domain(url):
        parsed_url = urlparse(url)
        return parsed_url.netloc

answers = []
shuffled_ids = []
correctness = []
all_messages = []
openai_batch = []
for i, (_, question) in enumerate(questions):
    if(i % 10 == 0):
        print(i)

    if(args.stop_after_k is not None and i == args.stop_after_k):
        break

    q, a = parse_question(question)
   
    context_tups = []
    for context_file, context_key in zip(context_files, context_keys):
        context = context_file[i]
        context_tups.append((context, context_key))
    
    if(C4_JSONL is not None and NO_C4_DOCUMENTS > 0):
        c4_sample = random.sample(c4_documents, NO_C4_DOCUMENTS)
        context_tups.extend([((s["text"][:1000], None), s["metadata"]["url"]) for s in c4_sample])
    
    ct = list(enumerate(context_tups))
    random.shuffle(ct)
    context_tups = [t for _, t in ct]
    shuffled_ids.append([idx for idx, _ in ct])

    context_string = ""
    for j, ((c, _), u) in enumerate(context_tups):
        c = c.replace('\n', ' ')
        domain = get_domain(u)
        if(len(domain) == 0):
            domain = "unknown"
        context_string += f"Context document {j + 1} source: {domain}\nContext document {j + 1}: {c}\n\n"

    message = "Write nothing but the answer to each question, or \"I don't know\" if you don't know the answer. Write exactly one answer per question (do not write \"or\"). You may (but do not have to) consult the provided context. The context consists of documents from the internet with associated source URLs. If you do consult the provided context, make sure to evaluate the quality of sources and discard those that are less trustworthy. "
    message += f"{context_string}\nQuestion: {q}"

    all_messages.append(message)

    output_text = input(message)

    if(WRITE_TO_DISK):
        continue

    answers.append(output_text)

if(WRITE_TO_DISK):
    os.makedirs(PROMPT_DIR, exist_ok=True)
    with open(os.path.join(PROMPT_DIR, f"{question_file}_{run_name}.pickle"), "wb") as fp:
        pickle.dump(all_messages, fp, protocol=pickle.HIGHEST_PROTOCOL)

    exit()

with open(os.path.join(OUTPUT_DIR, f"{question_file}_{run_name}.pickle"), "wb") as fp:
    pickle.dump(list(zip(answers, shuffled_ids)), fp, protocol=pickle.HIGHEST_PROTOCOL)
