import argparse
import datasets
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

from smel.utils.constants import (
    DOMAINS,
    URL_TO_NAME,
)
from smel.utils.utils import (
    filter_combinations,
    get_context_keys,
)


def parse_question(question):
    q, a = question.split('\n')
    q = q.strip()
    a = a.strip()
    return q, a


parser = argparse.ArgumentParser()
#parser.add_argument("--combo_id", type=int, default=0)
parser.add_argument("--entity", type=str, default="agency")
parser.add_argument("--model", type=str, default="llama")
parser.add_argument("--use_local", action="store_true", default=False)
args = parser.parse_args()

ENTITY = args.entity
MODEL = args.model

RUN_NAME = f"{ENTITY}_ignoring"
USE_CONTEXT = True
FIG_DIR = f"results/{MODEL}_{RUN_NAME}"
PICKLE_DIR = "pickles"
OUTPUT_DIR = f"pickles/{MODEL}_{RUN_NAME}/"

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE = 64

model_name = "meta-llama/Llama-3.3-70B-Instruct"

if(args.use_local):
    QUESTION_FILE = f"{ENTITY}_questions_filtered.pickle"
    CONTEXT_FILES = [
        f"{ENTITY}_questions_filtered_rewritten.pickle",
        #f"{ENTITY}_questions_filtered_{ENTITY}_questions_filtered_rewritten_corrupted.pickle",
    ]
    no_docs = len(CONTEXT_FILES)

    domain_combinations, context_keys = get_context_keys(no_docs, 0)

    with open(os.path.join(PICKLE_DIR, QUESTION_FILE), "rb") as fp:
        questions = pickle.load(fp)
   
    question_file = os.path.basename(QUESTION_FILE).rsplit('.', 1)[0]
else:
    CONTEXT_FILES = ["qa_rewritten", "qa_corrupted"]
    datasets = [datasets.load_dataset("gahdritz/smel", name=c, split="test") for c in CONTEXT_FILES]
 
    no_docs = len(datasets)
    
    domain_combinations, context_keys = get_context_keys(no_docs, 0)
 
    for d, context_key in zip(datasets, context_keys):
        questions = [(None, '\n'.join([e["question"], e["answer"]])) for e in datasets[0] if e["source"] == context_key and e["entity"] == ENTITY]

    question_file = CONTEXT_FILES[0]

# Load the model and tokenizer
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

for combo_id in range(len(domain_combinations)):
    print(f"Combo ID: {combo_id}")

    context_keys = list(domain_combinations[combo_id])
    run_name = f"{'_'.join([URL_TO_NAME[k].lower().replace(' ', '_') for k in context_keys])}_{MODEL}_{RUN_NAME}"
    
    answer_file_name = f"{question_file}_{run_name}.pickle"
   
    print(answer_file_name)

    if not os.path.exists(os.path.join(OUTPUT_DIR, answer_file_name)):
        print(answer_file_name)
        continue

    with open(os.path.join(OUTPUT_DIR, answer_file_name), "rb") as fp:
        answers = pickle.load(fp)
    
    context_files = []
    print(context_keys)
    for context_file, context_key in zip(CONTEXT_FILES, context_keys):
        with open(os.path.join(PICKLE_DIR, context_file), "rb") as fp:
            d = pickle.load(fp)
            contexts = d[context_key]
            
            assert(len(contexts) == len(questions))
            context_files.append(contexts)

    all_messages = []
    all_answers = []
    all_gt = []
    all_context_tups = []
    
    for i, ((_, question), (model_answer, _)) in enumerate(zip(questions, answers)):
        if(i % 10 == 0):
            print(i)
    
        q, a = parse_question(question)
    
        context_tups = []
        for context_file, context_key in zip(context_files, context_keys):
            context = context_file[i]
            context_tups.append((context, context_key))

        phrases_to_delete = [
            "More than ",
            "more than ",
            "Over ",
            "over ",
            "+",
        ]
        for p in phrases_to_delete:
            a = a.replace(p, "")
            model_answer = model_answer.replace(p, "")

        messages = [
            {"role": "system", "content": "You are an assistant that grades tests. Given a question, an answer key, and the student's answer, write \"Correct\" if the answer is correct and \"Incorrect\" otherwise. Do not write anything else. Do not give partial credit, and do not accept \"shotgun\" answers that contain more than one guess unless one answer is clearly identified as the final one. You can forgive slight exaggerations (ignore phrases like \"more than,\" \"less than,\" etc. when grading)."},
            {"role": "user", "content": f"Question: {q}\nAnswer key: {a}\nStudent answer: {model_answer}"}
        ]
    
        all_gt.append(a)
        all_answers.append(model_answer)
        all_messages.append(messages)
        all_context_tups.append(context_tups)
     
    outputs = pipeline(
        all_messages,
        temperature=None,
        max_new_tokens=256,
        batch_size=BATCH_SIZE,
    )
    output_texts = [o[0]["generated_text"] for o in outputs]
    
    correctness = [o[-1]["content"] for o in output_texts]
    
    for gt, ct, corr, ans in zip(all_gt, all_context_tups, correctness, all_answers):
        if not "Correct" in corr and not "know" in ans:
            print(gt)
            print(ct)
            print(ans)
            print('=' * 50)

    answer_file = os.path.basename(answer_file_name).rsplit('.', 1)[0]
    
    with open(os.path.join(PICKLE_DIR, f"{question_file}_{answer_file}_graded.pickle"), "wb") as fp:
        pickle.dump(correctness, fp, protocol=pickle.HIGHEST_PROTOCOL)
    
    os.makedirs(FIG_DIR, exist_ok=True)
    fraction_correct = sum([c == "Correct" for c in correctness]) / len(correctness)
    fraction_incorrect = 1 - fraction_correct
    fraction_or = sum(["or" in c for c in answers]) / len(correctness)
    fraction_abstention = sum(["I don't know" in c for c in answers]) / len(answers)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(["Correct", "Incorrect", "or", "Abstentions"], [fraction_correct, fraction_incorrect, fraction_or, fraction_abstention])
    
    run_name = answer_file_name.rsplit('.', 1)[0]
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
    
    cxt_name = '_'.join([s.rsplit('.', 1)[0] for s in [question_file, *CONTEXT_FILES]])
    fig_dir = os.path.join(FIG_DIR, cxt_name)
    os.makedirs(fig_dir, exist_ok=True)
    fig_name = f"{run_name}.png"
    fig_path = os.path.join(fig_dir, fig_name)
    plt.savefig(fig_path, bbox_inches="tight")
   
    print(f"Correct: {fraction_correct}")
    print(f"Incorrect: {fraction_incorrect}")
    print(f"Abstentions: {fraction_abstention}")
