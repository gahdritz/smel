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

from constants import DOMAINS
from few_shot_examples import FEW_SHOT_EXAMPLES

PICKLE_DIR = "../pickles"
AGENCY_FILE = "agencies.pickle"
FACT_LISTS = "agency_fact_lists.pickle"
USE_CONTEXT = True
FEW_SHOT = True
FEW_SHOT_K = 2
RESUME = True

BATCH_SIZE = 64 

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

with open(os.path.join(PICKLE_DIR, AGENCY_FILE), "rb") as fp:
    agencies = pickle.load(fp)

with open(os.path.join(PICKLE_DIR, FACT_LISTS), "rb") as fp:
    agency_fact_lists = pickle.load(fp)

start_point = 0
summaries = {}

fact_list_file = os.path.basename(FACT_LISTS).rsplit('.', 1)[0]
if(RESUME):
    with open(os.path.join(PICKLE_DIR, f"{fact_list_file}_passages.pickle"), "rb") as fp:
        summaries = pickle.load(fp)

    start_point = len(summaries[list(summaries.keys())[0]])

def parse_question(question):
    q, a = question.split('\n')
    q = q.strip()
    a = a.strip()
    return q, a

NO_FACTS = 1

accum = {}
for i, (agency, fact_list) in enumerate(zip(agencies, agency_fact_lists)):
    if(i < start_point):
        continue

    if(i % 10 == 0):
        print(i)

    fact_copy = [(j, f) for j, f in enumerate(fact_list)]
    random.shuffle(fact_copy)
    
    for j, (description, url) in enumerate(DOMAINS):
        messages = [
                {"role": "system", "content": "You are a bot that writes passages of text containing a specific fact in a specific style. Given a description of the source, the URL of the source, some source samples, some context, and a list of facts, write an excerpt containing the facts with the precise style and tone of the source. The placement of the facts should sound natural and should make sense in context. The excerpt should not be self-contained and should start and end abruptly, as if it's been taken from a larger document or webpage. Do not make the facts the focus of the excerpt. Do not make the excerpt more specific than the source requires. Do not include any information from the source samples; they are provided only as style guides. You are encouraged to include unrelated information, even if you have to make it up. Do not add commentary or otherwise editorialize the excerpt (no words like \"fascinating\"). Do not write run-on sentences. Do not write anything but the excerpt."},
        ]

        user_message = {"role": "user", "content": f"Source: {description}\nURL: {url}\n"}
        if(FEW_SHOT and url in FEW_SHOT_EXAMPLES):

#        if(FEW_SHOT):
#            if(not url in FEW_SHOT_EXAMPLES):
#                continue
            
            fses = [c for c, _ in FEW_SHOT_EXAMPLES[url][:FEW_SHOT_K]]
            for k, fse in enumerate(fses):
                user_message["content"] += f"Source sample {k + 1}: {fse}\n"

        user_message["content"] += f"Context: The {agency} is an agency of the United States federal government.\n"
        selection = [t for t in fact_copy[j * NO_FACTS: (j + 1) * NO_FACTS]]
        fact_indices, facts_to_include = tuple(zip(*selection))
        user_message["content"] += '\n'.join([f"Fact {k + 1}: {f}" for k, f in enumerate(facts_to_include)])
        messages.append(user_message)

        accum.setdefault("messages", [])
        accum["messages"].append(messages)

        accum.setdefault("url", [])
        accum["url"].append(url)

        accum.setdefault("fact_indices", [])
        accum["fact_indices"].append(fact_indices)
        
        if(len(accum["messages"]) == BATCH_SIZE):    
            outputs = pipeline(
                accum["messages"],
                temperature=1.0,
                max_new_tokens=512,
            )
            output_texts = [o[0]["generated_text"] for o in outputs]
    
            for accum_o, accum_u, accum_fi in zip(output_texts, accum["url"], accum["fact_indices"]):
                summaries.setdefault(accum_u, [])
                summaries[accum_u].append((accum_o[-1]["content"], accum_fi))

            accum = {}
        else:
            print("hello!")
            continue

    if(i % 10 == 0):
        with open(os.path.join(PICKLE_DIR, f"{fact_list_file}_passages.pickle"), "wb") as fp:
            pickle.dump(summaries, fp, protocol=pickle.HIGHEST_PROTOCOL)

if(len(accum) > 0):
    outputs = pipeline(
        accum["messages"],
        temperature=1.0,
        max_new_tokens=512,
    )
    output_texts = [o[0]["generated_text"] for o in outputs]

    for accum_o, accum_u, accum_fi in zip(output_texts, accum["url"], accum["fact_indices"]):
        summaries.setdefault(accum_u, [])
        summaries[accum_u].append((accum_o[-1]["content"], accum_fi))

    accum = {}

with open(os.path.join(PICKLE_DIR, f"{fact_list_file}_passages.pickle"), "wb") as fp:
    pickle.dump(summaries, fp, protocol=pickle.HIGHEST_PROTOCOL)

