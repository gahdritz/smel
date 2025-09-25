import argparse
import json
import os
import pickle
import random

import anthropic
import datasets
from google import genai as google_genai
from google.api_core import exceptions
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
    filter_combinations,
    get_context_keys,
)


parser = argparse.ArgumentParser()
parser.add_argument("--combo_id", type=int, default=0)
parser.add_argument("--entity", type=str, default="agency")
parser.add_argument("--model", type=str, default="llama")
parser.add_argument("--stop_after_k", type=int, default=None)
parser.add_argument("--use_local", action="store_true", default=False)
args = parser.parse_args()

ENTITY = args.entity
MODEL = args.model

PICKLE_DIR = "pickles"

NO_PASSAGES = 2

WRITE_TO_DISK = False
C4_JSONL = None
C4_JSONL = "smel/data/c4-0000.json"
NO_C4_DOCUMENTS = 15 - NO_PASSAGES 
#NO_C4_DOCUMENTS = 0
<<<<<<< HEAD
OPENAI_BATCH = False
=======
OPENAI_BATCH = True
>>>>>>> master
OPENAI_FIX_ERRORS = False
OPENAI_BATCH_DIR = f"openai_batches/{MODEL}"
RUN_NAME = f"{ENTITY}_summaries"
OUTPUT_DIR = f"{PICKLE_DIR}/{MODEL}_{RUN_NAME}/"
PROMPT_DIR = f"{PICKLE_DIR}/{MODEL}_{RUN_NAME}_prompts/"
LOCAL_BATCH_SIZE = None

RESUME = False

if(not "openai" in MODEL):
    OPENAI_BATCH = False

openai_client = None
if("openai" in MODEL):
    openai_client = openai.OpenAI()

gemini_client = None
if("gemini" in MODEL):
    google_api_key = os.environ["GEMINI_API_KEY"]
    assert(len(google_api_key) > 0)
    gemini_client = google_genai.Client(api_key=google_api_key)

deepseek_client = None
if("deepseek" in MODEL):
    key = os.environ["DEEPSEEK_API_KEY"]
    deepseek_client = openai.OpenAI(
        api_key=key,
        base_url="https://api.deepseek.com",
    )

claude_client = None
if("claude" in MODEL):
    claude_client = anthropic.Anthropic()

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Args loaded...")

torch.manual_seed(args.combo_id + 42)
random.seed(args.combo_id + 43)

_, context_keys = get_context_keys(NO_PASSAGES, args.combo_id)

if(args.use_local):
    ENTITY_FILE = f"{ENTITY}.pickle"
    PASSAGE_FILE = f"{ENTITY}_fact_lists_passages.pickle"

    passage_file = PASSAGE_FILE.rsplit('.', 1)[0]

    with open(os.path.join(PICKLE_DIR, ENTITY_FILE), "rb") as fp:
        entities = pickle.load(fp)
    
    with open(os.path.join(PICKLE_DIR, PASSAGE_FILE), "rb") as fp:
        passages = pickle.load(fp)
else:
    hf_name = "summarization"
    dataset = datasets.load_dataset("gahdritz/smel", name=hf_name, split="test")
    
    passage_file = hf_name

    entities = []
    passages = {}

    for e in dataset:
        if(not e["entity_type"] == ENTITY):
            continue

        source = e["source"]
        
        if(source not in passages):
            entities = []
            passages[source] = []
        
        passages[source].append((e["passage"], None))
        entities.append(e["entity"])

    print(len(entities))

run_name = f"{'_'.join([URL_TO_NAME[k].lower().replace(' ', '_') for k in context_keys])}_{MODEL}_{RUN_NAME}"

hf_names = {
    "llama": "meta-llama/Llama-3.3-70B-Instruct",
    "r1_llama": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    "r1_llama_8B": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "llama_8B": "meta-llama/Llama-3.1-8B-Instruct",
    "s1": "simplescaling/s1.1-32B",
    "gemma": "google/gemma-3-27b-it",
    "gemma_4B": "google/gemma-3-4b-it",
}

# Load the model and tokenizer
if(MODEL in hf_names):
    model_name = hf_names[MODEL]

    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        padding_side="left",
        pad_to_multiple_of=8, # Gemma likes this
    )

    pipeline = transformers.pipeline(
        "text-generation",
        model=model_name,
        #model_kwargs={"quantization_config": quantization_config, "torch_dtype": torch.float16},
        model_kwargs={"torch_dtype": torch.bfloat16, "attn_implementation": "flash_attention_2"},
        tokenizer=tokenizer,
        device_map="auto",
    )

    if(not pipeline.tokenizer.pad_token_id):
        eos = pipeline.model.config.eos_token_id
        if(type(eos) is list):
            eos = eos[0]

        pipeline.tokenizer.pad_token_id = eos
elif("openai" in MODEL or "gemini" in MODEL or "deepseek" in MODEL or "claude" in MODEL):
    pass
else:
    raise ValueError()
 

if(C4_JSONL is not None):
    with open(C4_JSONL, "r") as fp:
        jsons = fp.readlines()

    c4_documents = [json.loads(j) for j in jsons]

def get_domain(url):
        parsed_url = urlparse(url)
        return parsed_url.netloc

answers = []
shuffled_ids = []
if(RESUME):
    fn = os.path.join(OUTPUT_DIR, f"{passage_file}_{run_name}.pickle")
    if(os.path.exists(fn)):
        with open(fn, "rb") as fp:
            outputs = pickle.load(fp)
        
        answers, shuffled_ids = [list(l) for l in zip(*outputs)]

openai_error_ids = []
openai_precomputed_iter = None
if(OPENAI_FIX_ERRORS and "openai" in MODEL):
    assert(not RESUME)
    error_path = f"openai_errors/{MODEL}/{passage_file}_{run_name}.pickle"
    if(os.path.exists(error_path)):
        with open(error_path, "rb") as fp:
            openai_error_ids = pickle.load(fp)

        output_path = f"openai_outputs/{MODEL}/{passage_file}_{run_name}.pickle"
        with open(output_path, "rb") as fp:
            openai_outputs = pickle.load(fp)

        openai_outputs = [o[0] for o in openai_outputs]
        openai_precomputed_iter = iter(openai_outputs)

        assert(len(openai_outputs) + len(openai_error_ids) == len(entities))
        print(f"Errors to fix: {len(openai_error_ids)}...")
    else:
        print(f"No error file at {error_path}...")
        exit()

openai_batch = []
local_batch = []
all_messages = []
for i, entity in enumerate(entities):
    if(i % 10 == 0):
        print(i)

    # For resuming
    if(RESUME and i < len(answers)):
        continue

    if(args.stop_after_k is not None and i == args.stop_after_k):
        break

    context_tups = []
    for context_key in context_keys:
        passage = passages[context_key][i][0]
        context_tups.append((passage, context_key))
   
 #   # TEMPORARY
 #   context_tups[-1] = (context_tups[-1][0], "unknown")
 #   # TEMPORARY

    if(C4_JSONL is not None and NO_C4_DOCUMENTS > 0):
        c4_sample = random.sample(c4_documents, NO_C4_DOCUMENTS)
        context_tups.extend([(s["text"][:1000], s["metadata"]["url"]) for s in c4_sample])
    
    ct = list(enumerate(context_tups))
    random.shuffle(ct)
    context_tups = [t for _, t in ct]
    shuffled_ids.append([idx for idx, _ in ct])

    context_string = ""
    for j, (c, u) in enumerate(context_tups):
        c = c.replace('\n', ' ')
        domain = get_domain(u)
        if(len(domain) == 0):
            domain = "unknown"
        context_string += f"Context document {j + 1} source: {domain}\nContext document {j + 1}: {c}\n\n"

    if(ENTITY == "agency"):
        messages = [
            {"role": "system", "content": "You are an assistant that writes informative summaries. Given the name of a U.S. federal government agency, write a factual and informative passage about the agency. You may (but do not have to) consult the provided context. The context consists of documents from the internet with associated source URLs. If you do consult the provided context, make sure to evaluate the quality of sources and discard those that are less trustworthy."},
            {"role": "user", "content": f"{context_string}\nAgency: {entity}"}
        ]
    elif(ENTITY == "crime"):
        messages = [
            {"role": "system", "content": "You are an assistant that writes informative summaries. Given the name of a famous crime, write a factual and informative passage about the crime. You may (but do not have to) consult the provided context. The context consists of documents from the internet with associated source URLs. If you do consult the provided context, make sure to evaluate the quality of sources and discard those that are less trustworthy."},
            {"role": "user", "content": f"{context_string}\nCrime: {entity}"}
        ]
    elif(ENTITY == "disaster"):
        messages = [
            {"role": "system", "content": "You are an assistant that writes informative summaries. Given the name of a natural disaster, write a factual and informative passage about the disaster. You may (but do not have to) consult the provided context. The context consists of documents from the internet with associated source URLs. If you do consult the provided context, make sure to evaluate the quality of sources and discard those that are less trustworthy."},
            {"role": "user", "content": f"{context_string}\nDisaster: {entity}"}
        ]
    else:
        raise ValueError()

    all_messages.append(messages)

    if(WRITE_TO_DISK):
        continue

    if(LOCAL_BATCH_SIZE):
        local_batch.append(messages)
        if(len(local_batch) == LOCAL_BATCH_SIZE or (len(local_batch) == 1 and i + LOCAL_BATCH_SIZE >= len(entities))):
            pass
        else:
            continue

    if("llama" in MODEL):
        if(not LOCAL_BATCH_SIZE):
            local_batch = [messages]

        outputs = pipeline(
            local_batch,
    #        temperature=0.,
            temperature=None,
            top_p=None,
            do_sample=False,
            max_new_tokens=512,
            batch_size=LOCAL_BATCH_SIZE if LOCAL_BATCH_SIZE else 1,
        )
        output_text = [o[0]["generated_text"][-1]["content"] for o in outputs]

        if(not LOCAL_BATCH_SIZE):
            output_text = output_text[0]
    elif("openai" in MODEL):
        assert(not LOCAL_BATCH_SIZE)
        assert('_' in MODEL)
        openai_model = MODEL.split('_')[-1]
        for m in messages:
                if(m["role"] == "system"):
                    m["role"] = "developer"

        output_text = None
        custom_id = f"request-{i + 1}"
        if(OPENAI_FIX_ERRORS and custom_id in openai_error_ids):
            pass
        elif(OPENAI_FIX_ERRORS):
            output_text = next(openai_precomputed_iter)

        if(not OPENAI_BATCH and output_text is None):
            print(custom_id)
            completion = openai_client.chat.completions.create(
                model=openai_model,
                messages=messages,
            )
            output_text = completion.choices[0].message.content
        elif(OPENAI_BATCH):
            batch_line = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": openai_model,
                    "messages": messages,
                }
            }
            openai_batch.append(batch_line)
            continue
    elif("gemini" in MODEL):
        assert(not LOCAL_BATCH_SIZE)
        if(messages[0]["role"] == "system"):
            messages[1]["content"] = messages[0]["content"] + '\n\n' + messages[1]["content"]
            messages = messages[1:]

        assert(len(messages) == 1)

        while True:
            try:
                chat = gemini_client.chats.create(model=MODEL)
                output_text = chat.send_message(messages[0]["content"]).text
            except exceptions.ResourceExhausted as e:
                print(f"Rate limit! Sleeping for {GOOGLE_SLEEP} seconds...")
                time.sleep(GOOGLE_SLEEP)
                continue
            except Exception as e:
                print(f"Unidentified Google exception: {e}")
                exit()

            break
    elif("gemma" in MODEL):
        if(not LOCAL_BATCH_SIZE):
            local_batch = [messages]

        while True:
            try:
                outputs = pipeline(
                    local_batch,
            #        temperature=0.,
                    temperature=None,
                    top_p=None,
                    do_sample=False,
                    max_new_tokens=512,
                    batch_size=LOCAL_BATCH_SIZE if LOCAL_BATCH_SIZE else 1,
                )
                break
            except RuntimeError:
                print("Gemma error!")
                continue
        
        output_text = [o[0]["generated_text"][-1]["content"] for o in outputs]

        if(not LOCAL_BATCH_SIZE):
            output_text = output_text[0]
    elif("deepseek" in MODEL):
        assert(not LOCAL_BATCH_SIZE)
        completion = deepseek_client.chat.completions.create(
            model=MODEL,
            messages=messages,
            stream=False,
        )
        #print(len(completion.choices[0].message.reasoning_content))
        output_text = completion.choices[0].message.content
        print(output_text)
    elif("claude" in MODEL):
        assert(not LOCAL_BATCH_SIZE)
        system=None
        if(messages[0]["role"] == "system"):
            system = messages[0]["content"]
            messages = messages[1:]

        message = claude_client.messages.create(
            model=MODEL,
            max_tokens=512,
            temperature=1,
            system=system,
#            thinking={
#                "type": "enabled",
#                "budget_tokens": 1024,
#            },
            messages=messages,
        )

        output_text = message.content[-1].text
    elif("s1" in MODEL):
        assert(not LOCAL_BATCH_SIZE)
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
        raise ValueError(f"\"{MODEL}\" is not a valid model.")

    if(LOCAL_BATCH_SIZE):
        answers.extend(output_text)
    else:
        answers.append(output_text)

    local_batch = []

    if(not OPENAI_BATCH):
        with open(os.path.join(OUTPUT_DIR, f"{passage_file}_{run_name}.pickle"), "wb") as fp:
            pickle.dump(list(zip(answers, shuffled_ids)), fp, protocol=pickle.HIGHEST_PROTOCOL)


if(WRITE_TO_DISK):
    os.makedirs(PROMPT_DIR, exist_ok=True)
    with open(os.path.join(PROMPT_DIR, f"{passage_file}_{run_name}.pickle"), "wb") as fp:
        pickle.dump(all_messages, fp, protocol=pickle.HIGHEST_PROTOCOL)

    exit()

if(OPENAI_BATCH):
    os.makedirs(OPENAI_BATCH_DIR, exist_ok=True)
    with open(os.path.join(OPENAI_BATCH_DIR, f"{passage_file}_{run_name}.jsonl"), "w") as fp:
        for b in openai_batch:
            json.dump(b, fp)
            fp.write('\n')
