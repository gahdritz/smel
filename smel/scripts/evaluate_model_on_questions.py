import argparse
from itertools import combinations
import json
import os
import pickle
import random

import anthropic
import datasets
from google import genai as google_genai
from google.api_core import exceptions
import openai
import torch
import transformers
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
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


parser = argparse.ArgumentParser()
parser.add_argument("--combo_id", type=int, default=0)
parser.add_argument("--entity", type=str, default="agency")
parser.add_argument("--args.model", type=str, default="llama")
parser.add_argument("--stop_after_k", type=int, default=None)
parser.add_argument("--use_local", action="store_true", default=False)
parser.add_argument("--args.write_to_disk", action="store_true", default=False)
parser.add_argument("--c4_jsonl", type=str, default="smel/data/c4-0000.json")
parser.add_argument("--no_c4_documents", type=int, default=14)
parser.add_argument("--args.openai_batch", action="store_true", default=False)
parser.add_argument("--args.run_suffix", type=str, default="")
parser.add_argument("--args.openai_batch_dir", type=str, default="args.openai_batches")
parser.add_argument("--args.pickle_dir", type=str, default="pickles")
args = parser.parse_args()

GOOGLE_SLEEP=60
RUN_NAME = f"{args.entity}{args.run_suffix}"

args.openai_batch_dir = f"{args.args.openai_batch_dir}/{args.model}{args.run_suffix}"

OUTPUT_DIR = f"{args.pickle_dir}/{args.model}_{RUN_NAME}/"
PROMPT_DIR = f"{args.pickle_dir}/{args.model}_{RUN_NAME}_prompts/"

torch.manual_seed(args.combo_id + 42)
random.seed(args.combo_id + 43)


if(args.use_local):
    GEN_FILE = f"{args.entity}_gens.pickle"
    CONTEXT_FILES = [
        f"{args.entity}_gens_rewritten.pickle",
        #f"{args.entity}_gens_rewritten_corrupted.pickle",
    ]
    no_docs = len(CONTEXT_FILES)

    with open(os.path.join(args.pickle_dir, GEN_FILE), "rb") as fp:
        gens = pickle.load(fp)
   
    _, context_keys = get_context_keys(no_docs, args.combo_id)

    context_files = []
    for context_file, context_key in zip(CONTEXT_FILES, context_keys):
        with open(os.path.join(args.pickle_dir, context_file), "rb") as fp:
            d = pickle.load(fp)
            print(d.keys())
            print(context_file)
            contexts = d[context_key]
            
            assert(len(contexts) == len(gens))
            context_files.append(contexts)

    gen_file = os.path.basename(GEN_FILE).rsplit('.', 1)[0]
else:
    datasets = [
        datasets.load_dataset("gahdritz/smel", name="qa_rewritten", split="test"),
        datasets.load_dataset("gahdritz/smel", name="qa_corrupted", split="test"),
    ]
    no_docs = len(datasets)
    
    _, context_keys = get_context_keys(no_docs, args.combo_id)
 
    context_files = []
    for d, context_key in zip(datasets, context_keys):
        gens = [(None, '\n'.join([e["question"], e["answer"]])) for e in datasets[0] if e["source"] == context_key and e["entity"] == args.entity]
        contexts = [(e["passage"], None) for e in d if e["source"] == context_key and e["entity"] == args.entity]
        assert(len(contexts) == len(gens))
        context_files.append(contexts)

    gen_file = "qa_rewritten"

if(not "openai" in args.model):
    args.openai_batch = False

openai_client = None
if("openai" in args.model):
    openai_client = openai.OpenAI()

gemini_client = None
if("gemini" in args.model):
    google_api_key = os.environ["GEMINI_API_KEY"]
    assert(len(google_api_key) > 0)
    gemini_client = google_genai.Client(api_key=google_api_key)

deepseek_client = None
if("deepseek" in args.model):
    key = os.environ["DEEPSEEK_API_KEY"]
    deepseek_client = openai.OpenAI(
        api_key=key,
        base_url="https://api.deepseek.com",
    )

claude_client = None
if("claude" in args.model):
    claude_client = anthropic.Anthropic()

os.makedirs(OUTPUT_DIR, exist_ok=True)

run_name = f"{'_'.join([URL_TO_NAME[k].lower().replace(' ', '_') for k in context_keys])}_{args.model}_{RUN_NAME}"

hf_names = {
    "llama": "meta-llama/Llama-3.3-70B-Instruct",
    "r1_llama": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    "r1_llama_8B": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "llama_8B": "meta-llama/Llama-3.1-8B-Instruct",
    "s1": "simplescaling/s1.1-32B",
    "gemma": "google/gemma-3-27b-it",
    "gemma_4B": "google/gemma-3-4b-it",
}

# Load the args.model and tokenizer
if(args.model in hf_names):
    model_name = hf_names[args.model]

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        padding_side="left",
        pad_to_multiple_of=8, # Gemma likes this
    )
    
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_name,
        model_kwargs={"torch_dtype": torch.bfloat16, "attn_implementation": "flash_attention_2"},
        tokenizer=tokenizer,
        device_map="auto",
    )

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
all_messages = []
args.openai_batch = []
for i, (_, _, question) in enumerate(gens):
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
    for j, ((c, _, _), u) in enumerate(context_tups):
        c = c.replace('\n', ' ')
        domain = get_domain(u)
        if(len(domain) == 0):
            domain = "unknown"
        context_string += f"Context document {j + 1} source: {domain}\nContext document {j + 1}: {c}\n\n"

    messages = [
        {"role": "system", "content": "You are an assistant that answers questions correctly. Write nothing but the answer to each question, or \"I don't know\" if you don't know the answer. Write exactly one answer per question (do not write \"or\"). You may (but do not have to) consult the provided context. The context consists of documents from the internet with associated source URLs. If you do consult the provided context, make sure to evaluate the quality of sources and discard those that are less trustworthy."},
        {"role": "user", "content": f"{context_string}\nQuestion: {q}"}
    ]

    all_messages.append(messages)

    if(args.write_to_disk):
        continue

    if("llama" in args.model):
        outputs = pipeline(
            messages,
    #        temperature=0.,
            temperature=None,
            top_p=None,
            do_sample=False,
            max_new_tokens=256,
        )
        output_text = outputs[0]["generated_text"][-1]["content"]
    elif("gemma" in args.model):
        while True:
            try:
                outputs = pipeline(
                    messages,
            #        temperature=0.,
                    temperature=None,
                    top_p=None,
                    do_sample=False,
                    max_new_tokens=256,
                )
                break
            except RuntimeError:
                print("Gemma error!")
                continue
        
        output_text = outputs[0]["generated_text"][-1]["content"]
    elif("gemini" in args.model):
        if(messages[0]["role"] == "system"):
            messages[1]["content"] = messages[0]["content"] + '\n\n' + messages[1]["content"]
            messages = messages[1:]

        assert(len(messages) == 1)

        while True:
            try:
                chat = gemini_client.chats.create(args.model=args.model)
                output_text = chat.send_message(messages[0]["content"]).text
            except exceptions.ResourceExhausted as e:
                print(f"Rate limit! Sleeping for {GOOGLE_SLEEP} seconds...")
                time.sleep(GOOGLE_SLEEP)
                continue
            except Exception as e:
                print(f"Unidentified Google exception: {e}")
                exit()

            break
    elif("openai" in args.model):
        assert('_' in args.model)
        openai_args.model = args.model.split('_')[-1]
        for m in messages:
                if(m["role"] == "system"):
                    m["role"] = "developer"

        if(not args.openai_batch):
            completion = openai_client.chat.completions.create(
                args.model=openai_args.model,
                messages=messages,
            )
            output_text = completion.choices[0].message.content
        else:
            batch_line = {
                "custom_id": f"request-{i + 1}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "args.model": openai_args.model,
                    "messages": messages,
                }
            }
            args.openai_batch.append(batch_line)
            continue
    elif("deepseek" in args.model):
        completion = deepseek_client.chat.completions.create(
            args.model=args.model,
            messages=messages,
            stream=False,
        )
        #print(len(completion.choices[0].message.reasoning_content))
        output_text = completion.choices[0].message.content
    elif("claude" in args.model):
        system=None
        if(messages[0]["role"] == "system"):
            system = messages[0]["content"]
            messages = messages[1:]

        message = claude_client.messages.create(
            args.model=args.model,
            max_tokens=128,
            temperature=1,
            system=system,
#            thinking={
#                "type": "enabled",
#                "budget_tokens": 1024,
#            },
            messages=messages,
        )

        output_text = message.content[-1].text
    elif("s1" in args.model):
        if(messages[0]["role"] == "system"):
            messages[1]["content"] = messages[0]["content"] + '\n\n' + messages[1]["content"]
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
        raise ValueError(f"\"{args.model}\" is not a valid model.")

    answers.append(output_text)

if(args.write_to_disk):
    os.makedirs(PROMPT_DIR, exist_ok=True)
    with open(os.path.join(PROMPT_DIR, f"{gen_file}_{run_name}.pickle"), "wb") as fp:
        pickle.dump(all_messages, fp, protocol=pickle.HIGHEST_PROTOCOL)

    exit()

if(args.openai_batch):
    os.makedirs(args.openai_batch_DIR, exist_ok=True)
    with open(os.path.join(args.openai_batch_DIR, f"{gen_file}_{run_name}.jsonl"), "w") as fp:
        for b in args.openai_batch:
            json.dump(b, fp)
            fp.write('\n')
    exit()

with open(os.path.join(OUTPUT_DIR, f"{gen_file}_{run_name}.pickle"), "wb") as fp:
    pickle.dump(list(zip(answers, shuffled_ids)), fp, protocol=pickle.HIGHEST_PROTOCOL)
