import argparse
from itertools import combinations
import json
import os
import pickle
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


def parse_question(question):
    q, a = question.split('\n')
    q = q.strip()
    a = a.strip()
    return q, a


QUESTION_FILE = "crime_questions_filtered.pickle"
CONTEXT_FILES = [
    "crime_questions_filtered_rewritten.pickle",
    "crime_questions_filtered_crime_questions_filtered_rewritten_corrupted.pickle",
]

MODEL = "llama"
RUN_NAME = "crime_corrupted"
FIG_DIR = f"results/{MODEL}_{RUN_NAME}"
PICKLE_DIR = "pickles"
OUTPUT_DIR = f"pickles/{MODEL}_{RUN_NAME}/"

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("answer_file", type=str)
args = parser.parse_args()

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

with open(os.path.join(OUTPUT_DIR, args.answer_file), "rb") as fp:
    answers = pickle.load(fp)

correctness = []
for i, ((_, question), (model_answer, _)) in enumerate(zip(questions, answers)):
    if(i % 10 == 0):
        print(i)

    q, a = parse_question(question)
   
    messages = [
        {"role": "system", "content": "You are a bot that grades tests. Given a question, an answer key, and the student's answer, write \"Correct\" if the answer is correct and \"Incorrect\" otherwise. Do not write anything else. Do not give partial credit, and do not accept \"shotgun\" answers that contain more than one guess."},
        {"role": "user", "content": f"Question: {q}\nAnswer key: {a}\nStudent answer: {model_answer}"}
    ]
 
    outputs = pipeline(
        messages,
        temperature=None,
        max_new_tokens=256,
    )
    output_text = outputs[0]["generated_text"]

    correctness.append(output_text[-1]["content"])

question_file = os.path.basename(QUESTION_FILE).rsplit('.', 1)[0]
answer_file = os.path.basename(args.answer_file).rsplit('.', 1)[0]

with open(os.path.join(PICKLE_DIR, f"{question_file}_{answer_file}_graded.pickle"), "wb") as fp:
    pickle.dump(correctness, fp, protocol=pickle.HIGHEST_PROTOCOL)

os.makedirs(FIG_DIR, exist_ok=True)
fraction_correct = sum([c == "Correct" for c in correctness]) / len(correctness)
fraction_incorrect = 1 - fraction_correct
fraction_or = sum(["or" in c for c in answers]) / len(correctness)
fraction_abstention = sum(["I don't know" in c for c in answers]) / len(answers)

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(["Correct", "Incorrect", "or", "Abstentions"], [fraction_correct, fraction_incorrect, fraction_or, fraction_abstention])

run_name = args.answer_file.rsplit('.', 1)[0]
title = run_name
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
fig_name = f"{run_name}.png"
fig_path = os.path.join(fig_dir, fig_name)
plt.savefig(fig_path, bbox_inches="tight")

print(list(zip(answers, correctness)))
