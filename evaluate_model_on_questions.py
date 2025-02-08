from accelerate import Accelerator
import json
import os
import pickle
import random
import torch
import transformers
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from urllib.parse import urlparse


QUESTION_FILE = "questions_filtered.pickle"
CONTEXT_FILES = [
    "questions_filtered_rewritten.pickle",
#    "questions_filtered_questions_filtered_rewritten_corrupted.pickle",
]
CONTEXT_KEYS = [
    "https://twitter.com",
#    "https://twitter.com",
]

assert(len(CONTEXT_FILES) == len(CONTEXT_KEYS))

C4_JSONL = None
#C4_JSONL = "scratch/c4-0000.json"
NO_C4_DOCUMENTS = 10 - len(CONTEXT_FILES)
USE_CONTEXT = True

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

with open(QUESTION_FILE, "rb") as fp:
    questions = pickle.load(fp)

context_files = []
for context_file, context_key in zip(CONTEXT_FILES, CONTEXT_KEYS):
    with open(context_file, "rb") as fp:
        contexts = pickle.load(fp)[context_key]
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
for i, (_, question) in enumerate(questions):
    if(i % 10 == 0):
        print(i)

    q, a = parse_question(question)
   
    if(not USE_CONTEXT):
        messages = [
            {"role": "system", "content": "You are a bot that answers questions accurately. Write nothing but the answer to each question. If you don't know the answer, write \"I don't know\"."},
            {"role": "user", "content": q}
        ]
    else:
        context_tups = []
        for context_file, context_key in zip(context_files, CONTEXT_KEYS):
            context, _ = context_file[i]
            context_tups.append((context, context_key))
        
        if(C4_JSONL is not None and NO_C4_DOCUMENTS > 0):
            c4_sample = random.sample(c4_documents, NO_C4_DOCUMENTS)
            context_tups.extend([(s["text"], s["metadata"]["url"]) for s in c4_sample])
            
        random.shuffle(context_tups)

        context_string = ""
        for i, ((c, _), u) in enumerate(context_tups):
            c = c.replace('\n', ' ')
            domain = get_domain(u)
            if(len(domain) == 0):
                domain = "unknown"
            context_string += f"Context document {i + 1}: {c}\nContext document {i + 1} source: {domain}\n"

        messages = [
            {"role": "system", "content": "You are a bot that answers questions accurately. Write nothing but the answer to each question, or \"I don't know\" if you don't know the answer. You may (but do not have to) consult the provided context. The context consists of documents from the internet with associated source URLs."},
            {"role": "user", "content": f"{context_string}\nQuestion: {q}"}
        ]

            
    outputs = pipeline(
        messages,
        temperature=0.7,
        max_new_tokens=256,
    )
    output_text = outputs[0]["generated_text"]

    answers.append(output_text[-1]["content"])

correctness = []
for (context, question), answer in zip(questions, answers):
    q, a = parse_question(question)

    messages = [
        {"role": "system", "content": "You are a bot that grades tests. Given a question, an answer key, and the student's answer, write \"Correct\" if the answer is correct and \"Incorrect\" otherwise. Do not write anything else."},
        {"role": "user", "content": f"Question: {q}\nAnswer key: {a}\nStudent answer: {answer}"}
    ]
            
    outputs = pipeline(
        messages,
        temperature=0.7,
        max_new_tokens=256,
    )
    output_text = outputs[0]["generated_text"]

    correctness.append(output_text[-1]["content"])

question_file = os.path.basename(QUESTION_FILE).rsplit('.', 1)[0]
if(USE_CONTEXT):
    question_file += "_context"
with open(f"{question_file}_answers.pickle", "wb") as fp:
    pickle.dump(list(zip(answers, correctness)), fp, protocol=pickle.HIGHEST_PROTOCOL)

print(list(zip(answers, correctness)))
