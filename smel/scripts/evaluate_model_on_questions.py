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



parser = argparse.ArgumentParser()
parser.add_argument("--combo_id", type=int, default=0)
parser.add_argument("--entity", type=str, default="agency")
parser.add_argument("--model", type=str, default="llama")
parser.add_argument("--stop_after_k", type=int, default=None)
parser.add_argument("--use_local", action="store_true", default=False)
args = parser.parse_args()

ENTITY = args.entity
MODEL = args.model

WRITE_TO_DISK = False
C4_JSONL = None
C4_JSONL = "smel/data/c4-0000.json"
NO_C4_DOCUMENTS = 14
GOOGLE_SLEEP=60
#NO_C4_DOCUMENTS = 0
OPENAI_BATCH = False
OPENAI_BATCH_DIR = f"openai_batches/{MODEL}"
RUN_NAME = f"{ENTITY}_ignoring"
PICKLE_DIR = f"pickles"
OUTPUT_DIR = f"{PICKLE_DIR}/{MODEL}_{RUN_NAME}/"
PROMPT_DIR = f"{PICKLE_DIR}/{MODEL}_{RUN_NAME}_prompts/"

torch.manual_seed(args.combo_id + 42)
random.seed(args.combo_id + 43)


if(args.use_local):
    QUESTION_FILE = f"{ENTITY}_questions_filtered.pickle"
    CONTEXT_FILES = [
        f"{ENTITY}_questions_filtered_rewritten.pickle",
        f"{ENTITY}_questions_filtered_{ENTITY}_questions_filtered_rewritten_corrupted.pickle",
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
            print(context_file)
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

if(not "openai" in MODEL):
    OPENAI_BATCH = False

openai_client = None
if("openai" in MODEL):
    openai_client = openai.OpenAI(
    )

gemini_client = None
if("gemini" in MODEL):
    google_api_key = os.environ["GEMINI_API_KEY"]
    assert(len(google_api_key) > 0)
    # gemini_client = google_genai.Client(api_key=google_api_key)
    google_genai.configure(api_key=google_api_key)
    model = google_genai.GenerativeModel(model_name=MODEL)

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

if(C4_JSONL is not None):
    with open(C4_JSONL, "r") as fp:
        jsons = fp.readlines()

    c4_documents = [json.loads(j) for j in jsons]

def parse_question(question):
    print(question)
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

    messages = [
        {"role": "system", "content": "You are an assistant that answers questions correctly. Write nothing but the answer to each question, or \"I don't know\" if you don't know the answer. Write exactly one answer per question (do not write \"or\"). You may (but do not have to) consult the provided context. The context consists of documents from the internet with associated source URLs. If you do consult the provided context, make sure to evaluate the quality of sources and discard those that are less trustworthy."}, #For gathering test results
        # {"role": "system", "content": "You are an assistant that answers questions correctly. Write nothing but the answer to each question unless you don't know the answer. If you don't know the answer, write \"I don't know\" and a short explanation about why you don't know. Write exactly one answer per question (do not write \"or\"). You may (but do not have to) consult the provided context. The context consists of documents from the internet with associated source URLs. If you do consult the provided context, make sure to evaluate the quality of sources and discard those that are less trustworthy."}, #Provide short explanation for abstention
        {"role": "user", "content": f"{context_string}\nQuestion: {q}"}
    ]

    all_messages.append(messages)

    if(WRITE_TO_DISK):
        continue

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
    elif("gemma" in MODEL):
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
            except RuntimeError as e:
                print(e)
                print("Gemma error!")
                continue

        output_text = outputs[0]["generated_text"][-1]["content"]
    elif("gemini" in MODEL):
        if(messages[0]["role"] == "system"):
            messages[1]["content"] = messages[0]["content"] + '\n\n' + messages[1]["content"]
            messages = messages[1:]

        assert(len(messages) == 1)

        while True:
            try:
                # chat = gemini_client.chats.create(model=MODEL)
                # output_text = chat.send_message(messages[0]["content"]).text

                chat = model.start_chat(history=[])
                output_text = chat.send_message(messages[0]["content"]).text
            except exceptions.ResourceExhausted as e:
                print(f"Rate limit! Sleeping for {GOOGLE_SLEEP} seconds...")
                time.sleep(GOOGLE_SLEEP)
                continue
            except Exception as e:
                print(f"Unidentified Google exception: {e}")
                exit()

            break
    elif("openai" in MODEL):
        assert('_' in MODEL)
        openai_model = MODEL.split('_')[-1]
        for m in messages:
                if(m["role"] == "system"):
                    m["role"] = "developer"

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
    elif("deepseek" in MODEL):
        completion = deepseek_client.chat.completions.create(
            model=MODEL,
            messages=messages,
            stream=False,
        )
        #print(len(completion.choices[0].message.reasoning_content))
        output_text = completion.choices[0].message.content
    elif("claude" in MODEL):
        system=None
        if(messages[0]["role"] == "system"):
            system = messages[0]["content"]
            messages = messages[1:]

        message = claude_client.messages.create(
            model=MODEL,
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
    elif("s1" in MODEL):
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

    answers.append(output_text)

if(WRITE_TO_DISK):
    os.makedirs(PROMPT_DIR, exist_ok=True)
    with open(os.path.join(PROMPT_DIR, f"{question_file}_{run_name}.pickle"), "wb") as fp:
        pickle.dump(all_messages, fp, protocol=pickle.HIGHEST_PROTOCOL)

    exit()

if(OPENAI_BATCH):
    os.makedirs(OPENAI_BATCH_DIR, exist_ok=True)
    with open(os.path.join(OPENAI_BATCH_DIR, f"{question_file}_{run_name}.jsonl"), "w") as fp:
        for b in openai_batch:
            json.dump(b, fp)
            fp.write('\n')
    exit()

with open(os.path.join(OUTPUT_DIR, f"{question_file}_{run_name}.pickle"), "wb") as fp:
    pickle.dump(list(zip(answers, shuffled_ids)), fp, protocol=pickle.HIGHEST_PROTOCOL)
