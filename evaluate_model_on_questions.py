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


QUESTION_FILE = "gov_questions_filtered.pickle"
CONTEXT_FILES = [
    "gov_questions_filtered_rewritten.pickle",
    "gov_questions_filtered_gov_questions_filtered_rewritten_corrupted.pickle",
]

C4_JSONL = None
C4_JSONL = "scratch/c4-0000.json"
NO_C4_DOCUMENTS = 15 - len(CONTEXT_FILES)
#NO_C4_DOCUMENTS = 0
USE_CONTEXT = True
MODEL = "llama_8B"
OPENAI_BATCH = True
OPENAI_BATCH_DIR = f"openai_batches/{MODEL}"
RUN_NAME = "corrupted"
PICKLE_DIR = f"pickles"
OUTPUT_DIR = f"pickles/{MODEL}_{RUN_NAME}/"

if(not "openai" in MODEL):
    OPENAI_BATCH = False

openai_client = None
if("openai" in MODEL):
    openai_client = openai.OpenAI(
#        organization="org-MQ1LwNk1M7cTO8frDDsYwS0p",
    )

os.makedirs(OUTPUT_DIR, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--combo_id", type=int, default=0)
args = parser.parse_args()

torch.manual_seed(args.combo_id + 42)
random.seed(args.combo_id + 43)

domain_combinations = list(combinations([url for _, url in DOMAINS], len(CONTEXT_FILES)))
context_keys = list(domain_combinations[args.combo_id])

print(list(enumerate(domain_combinations)))

assert(len(CONTEXT_FILES) == len(context_keys))

run_name = f"{'_'.join([URL_TO_NAME[k].lower().replace(' ', '_') for k in context_keys])}_{MODEL}_{RUN_NAME}"

hf_names = {
    "llama": "meta-llama/Llama-3.3-70B-Instruct",
    "r1_llama": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    "r1_llama_8B": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "llama_8B": "meta-llama/Llama-3.1-8B-Instruct",
}

# Load the model and tokenizer
if(MODEL in hf_names):
    model_name = hf_names[MODEL]

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

with open(os.path.join(PICKLE_DIR, QUESTION_FILE), "rb") as fp:
    questions = pickle.load(fp)

context_files = []
for context_file, context_key in zip(CONTEXT_FILES, context_keys):
    with open(os.path.join(PICKLE_DIR, context_file), "rb") as fp:
        d = pickle.load(fp)
        print(d.keys())
        print(context_file)
        contexts = d[context_key]
        
        assert(len(contexts) == len(questions))
        context_files.append(contexts)

if(C4_JSONL is not None):
    with open(C4_JSONL, "r") as fp:
        jsons = fp.readlines()

    c4_documents = [json.loads(j) for j in jsons]

def parse_question(question):
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
openai_batch = []
for i, (_, question) in enumerate(questions):
    if(i % 10 == 0):
        print(i)

    q, a = parse_question(question)
   
    if(not USE_CONTEXT):
        messages = [
            {"role": "system", "content": "You are a bot that answers questions correctly. Write nothing but the answer to each question. Commit to one answer. Write exactly one answer per question (do NOT write \"or\")."},
            {"role": "user", "content": q}
        ]
    else:
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
            context_string += f"Context document {j + 1}: {c}\nContext document {j + 1} source: {domain}\n"

        messages = [
            {"role": "system", "content": "You are a bot that answers questions correctly. Write nothing but the answer to each question, or \"I don't know\" if you don't know the answer. Write exactly one answer per question (do not write \"or\"). You may (but do not have to) consult the provided context. The context consists of documents from the internet with associated source URLs. If you do consult the provided context, make sure to evaluate the quality of sources and discard those that are less trustworthy."},
            {"role": "user", "content": f"{context_string}\nQuestion: {q}"}
        ]

 #       messages = [
 #           {"role": "system", "content": "You are a bot that answers questions correctly. If you don't know the answer, write \"I don't know\". Write exactly one answer per question (do not write \"or\"). You may (but do not have to) consult the provided context. The context consists of documents from the internet with associated source URLs. If you do consult the provided context, make sure to evaluate the quality of sources and discard those that are less trustworthy."},
 #           {"role": "user", "content": f"{context_string}\nQuestion: {q}"}
 #       ]

    if("llama" in MODEL):
        outputs = pipeline(
            messages,
    #        temperature=0.,
            temperature=None,
            top_p=None,
            do_sample=False,
            max_new_tokens=256,
        )
        output_text = outputs[0]["generated_text"][-1]["content"]
    elif("openai" in MODEL):
        assert('_' in MODEL)
        openai_model = MODEL.split('_')[-1]
        for m in messages:
                if(m["role"] == "system"):
                    m["role"] == "developer"

        if(not OPENAI_BATCH):
            completion = openai_client.chat.completions.create(
                model=openai_model,
                messages=messages,
            )
            output_text = completion.choices[0].message.content
        else:
            batch_line = {
                "custom_id": f"request-{i + 1}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": openai_model,
                    "messages": messages,
                }
            }
            openai_batch.append(batch_line)
            continue
    elif("r1" in MODEL):
        if(messages[0]["role"] == "system"):
            messages[1]["content"] = messages[0]["content"] + '\n' + messages[1]["content"]
            messages = messages[1:]
        outputs = pipeline(
            messages,
    #        temperature=0.,
            temperature=None,
            top_p=None,
            do_sample=False,
            max_new_tokens=1024,
        )
        output_text = outputs[0]["generated_text"][-1]["content"]
    else:
        raise ValueError(f"\"{MODEL}\" is not a valid model.")

    answers.append(output_text)

if(OPENAI_BATCH):
    os.makedirs(OPENAI_BATCH_DIR, exist_ok=True)
    with open(os.path.join(OPENAI_BATCH_DIR, f"{run_name}.jsonl"), "w") as fp:
        for b in openai_batch:
            json.dump(b, fp)
            fp.write('\n')
    exit()

question_file = os.path.basename(QUESTION_FILE).rsplit('.', 1)[0]
if(USE_CONTEXT):
    question_file += "_context"
with open(os.path.join(OUTPUT_DIR, f"{question_file}_{run_name}.pickle"), "wb") as fp:
    pickle.dump(list(zip(answers, shuffled_ids)), fp, protocol=pickle.HIGHEST_PROTOCOL)
