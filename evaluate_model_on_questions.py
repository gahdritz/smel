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
USE_CONTEXT = True
FIG_DIR = "results"
PICKLE_DIR = "pickles"
MODEL = "o3-mini"

openai_client = None
if(MODEL == "o3-mini"):
    openai_client = OpenAI(
        ***REMOVED***
    )

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(PICKLE_DIR, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--combo_id", type=int, default=0)
args = parser.parse_args()

torch.manual_seed(args.combo_id + 42)
random.seed(args.combo_id + 43)

domain_combinations = list(combinations([url for _, url in DOMAINS], 2))
context_keys = list(domain_combinations[args.combo_id])

assert(len(CONTEXT_FILES) == len(context_keys))

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
        for i, ((c, _), u) in enumerate(context_tups):
            c = c.replace('\n', ' ')
            domain = get_domain(u)
            if(len(domain) == 0):
                domain = "unknown"
            context_string += f"Context document {i + 1}: {c}\nContext document {i + 1} source: {domain}\n"

        messages = [
            {"role": "system", "content": "You are a bot that answers questions correctly. Write nothing but the answer to each question, or \"I don't know\" if you don't know the answer. Write exactly one answer per question (do not write \"or\"). You may (but do not have to) consult the provided context. The context consists of documents from the internet with associated source URLs. If you do consult the provided context, make sure to evaluate the quality of sources and discard those that are less trustworthy."},
            {"role": "user", "content": f"{context_string}\nQuestion: {q}"}
        ]

    if(MODEL == "llama"):
        outputs = pipeline(
            messages,
    #        temperature=0.,
            temperature=None,
            top_p=None,
            do_sample=False,
            max_new_tokens=256,
        )
        output_text = outputs[0]["generated_text"][-1]["content"]

    elif(MODEL == "o3-mini"):
        for m in messages:
                if(m["role"] == "system"):
                    m["role"] == "developer"

        completion = openai_client.chat.completions.create(
            model="o3-mini",
            messages=messages,
        )

        output_text = completion.choices[0].message

        print(output_text)
        exit()
    else:
        raise ValueError(f"\"{MODEL}\" is not a valid model.")

    answers.append(output_text)

correctness = []
for (context, question), answer in zip(questions, answers):
    q, a = parse_question(question)

    messages = [
        {"role": "system", "content": "You are a bot that grades tests. Given a question, an answer key, and the student's answer, write \"Correct\" if the answer is correct and \"Incorrect\" otherwise. Do not write anything else. Do not give partial credit, and do not accept \"shotgun\" answers that contain more than one guess."},
        {"role": "user", "content": f"Question: {q}\nAnswer key: {a}\nStudent answer: {answer}"}
    ]
            
    outputs = pipeline(
        messages,
        temperature=0.7,
        max_new_tokens=256,
    )
    output_text = outputs[0]["generated_text"]

    correctness.append(output_text[-1]["content"])

run_name = f"{'_'.join([URL_TO_NAME[k].lower().replace(' ', '_') for k in context_keys])}"
question_file = os.path.basename(QUESTION_FILE).rsplit('.', 1)[0]
if(USE_CONTEXT):
    question_file += "_context"
with open(os.path.join(PICKLE_DIR, f"{question_file}_{run_name}.pickle"), "wb") as fp:
    pickle.dump(list(zip(answers, correctness, shuffled_ids)), fp, protocol=pickle.HIGHEST_PROTOCOL)

os.makedirs(FIG_DIR, exist_ok=True)
fraction_correct = sum([c == "Correct" for c in correctness]) / len(correctness)
fraction_incorrect = 1 - fraction_correct
fraction_or = sum(["or" in c for c in answers]) / len(correctness)
fraction_abstention = sum(["I don't know" in c for c in answers]) / len(answers)

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(["Correct", "Incorrect", "or", "Abstentions"], [fraction_correct, fraction_incorrect, fraction_or, fraction_abstention])

title = '/'.join([URL_TO_NAME[k] for k in context_keys])
ax.set_title(title)

for bar in bars:
    height = round(bar.get_height(), 3)
    ax.text(bar.get_x() + bar.get_width()/2., height,
        f'{height:}',
        ha='center', 
        va='bottom'
    )

ax.grid(axis='y', linestyle='--', alpha=0.7)

cxt_name = '_'.join([s.rsplit('.', 1)[0] for s in [QUESTION_FILE, *CONTEXT_FILES]])
fig_dir = os.path.join(FIG_DIR, cxt_name)
os.makedirs(fig_dir, exist_ok=True)
fig_name = f"{'_'.join([URL_TO_NAME[k].lower().replace(' ', '_') for k in context_keys])}.png"
fig_path = os.path.join(fig_dir, fig_name)
plt.savefig(fig_path, bbox_inches="tight")

print(list(zip(answers, correctness)))
