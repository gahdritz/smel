import json
import openai
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
ENTITY_FILE = "disaster.pickle"
FACT_LISTS = "disaster_fact_lists.pickle"
USE_CONTEXT = True
FEW_SHOT = True
FEW_SHOT_K = 2
RUN_NAME = "disaster"
RESUME = False

BATCH_SIZE = 1
pipeline_batch_size = 32

#MODEL = "meta-llama/Llama-3.3-70B-Instruct"
#MODEL = "openai_gpt-4o"
MODEL = "openai_chatgpt-4o-latest"

OPENAI_BATCH = False and "openai" in MODEL
OPENAI_BATCH_DIR = f"openai_batches/{MODEL}_domain"

openai_client = None
if("openai" in MODEL):
    openai_client = openai.OpenAI(
#        organization="org-MQ1LwNk1M7cTO8frDDsYwS0p",
    )

pipeline = None
if(not "openai" in MODEL):
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    
    pipeline = transformers.pipeline(
        "text-generation",
        model=MODEL,
        #model_kwargs={"quantization_config": quantization_config, "torch_dtype": torch.float16},
        model_kwargs={"torch_dtype": torch.bfloat16, "attn_implementation": "flash_attention_2"},
        tokenizer=tokenizer,
        device_map="auto",
    )

with open(os.path.join(PICKLE_DIR, ENTITY_FILE), "rb") as fp:
    agencies = pickle.load(fp)

with open(os.path.join(PICKLE_DIR, FACT_LISTS), "rb") as fp:
    entity_fact_lists = pickle.load(fp)

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


def process_messages_openai(messages, idx):
    openai_model = MODEL.split('_')[-1]
    for m in messages:
            if(m["role"] == "system"):
                m["role"] = "developer"

    batch_line = {
        "custom_id": f"request-{idx + 1}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": openai_model,
            "messages": messages,
        }
    }

    return batch_line


NO_FACTS = 1

accum = {}
openai_batch = []
openai_idx = 0
for i, (entity, fact_list) in enumerate(zip(agencies, entity_fact_lists)):
    if(i < start_point):
        continue

    if(i % 10 == 0):
        print(i)

    fact_copy = [(j, f) for j, f in enumerate(fact_list)]
    random.shuffle(fact_copy)
    
    for j, (description, url) in enumerate(DOMAINS):
        messages = [
                {"role": "system", "content": "You are an assistant that writes passages of text containing specific facts in a specific style. Given a description of the source, the URL of the source, some source samples, some context, and a list of facts, write a medium-length excerpt (2-3 paragraphs, as appropriate) containing the facts with the precise style and tone of the source. The placement of the facts should sound natural and should make sense in context. The excerpt should not be self-contained and can start and end abruptly, as if it's been taken from a larger document or webpage. Do not make the facts the focus of the excerpt. Do not make the excerpt more specific than the source requires. Do not include any information from the source samples; they are provided only as style guides. You are encouraged to include unrelated information, even if you have to make it up. Do not add commentary or otherwise editorialize the excerpt (no words like \"fascinating\"). Do not write run-on sentences. Do not write anything but the excerpt."},
        ]

        user_message = {"role": "user", "content": f"Source: {description}\nURL: {url}\n"}
        if(FEW_SHOT and url in FEW_SHOT_EXAMPLES):

#        if(FEW_SHOT):
#            if(not url in FEW_SHOT_EXAMPLES):
#                continue
            
            fses = [c for c, _ in FEW_SHOT_EXAMPLES[url][:FEW_SHOT_K]]
            for k, fse in enumerate(fses):
                user_message["content"] += f"Source sample {k + 1}: \"{fse}\"\n"

        #user_message["content"] += f"Context: The {entity} is an agency of the United States federal government.\n"
        #user_message["content"] += f"Context: The \"{entity}\" is a famous \"true crime\".\n"
        user_message["content"] += f"Context: The {entity} is a famous natural disaster.\n"
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
            if(not OPENAI_BATCH):
                if('openai' in MODEL):
                    openai_model = MODEL.split('_')[-1]
                    output_texts = []
                    for messages in accum["messages"]:
                        completion = openai_client.chat.completions.create(
                            model=openai_model,
                            messages=messages,
                        )
                        output_texts.append(completion.choices[0].message.content)
                else:
                    outputs = pipeline(
                        accum["messages"],
                        temperature=1.0,
                        batch_size=pipeline_batch_size,
                        max_new_tokens=512,
                    )
                    output_texts = [o[0]["generated_text"][-1]["content"] for o in outputs] 

                for accum_o, accum_u, accum_fi in zip(output_texts, accum["url"], accum["fact_indices"]):
                    summaries.setdefault(accum_u, [])
                    summaries[accum_u].append((accum_o, accum_fi))
            else:
                for m in accum["messages"]:
                    openai_batch.append(
                        process_messages_openai(m, openai_idx)
                    )
                    openai_idx += 1

            accum = {}
        else:
            continue

    if(not OPENAI_BATCH):
        if(i % 10 == 0):
            with open(os.path.join(PICKLE_DIR, f"{fact_list_file}_passages.pickle"), "wb") as fp:
                pickle.dump(summaries, fp, protocol=pickle.HIGHEST_PROTOCOL)

if(len(accum) > 0):
    if(not OPENAI_BATCH):
        if('openai' in MODEL):
            openai_model = MODEL.split('_')[-1]
            output_texts = []
            for messages in accum["messages"]:
                completion = openai_client.chat.completions.create(
                    model=openai_model,
                    messages=messages,
                )
                output_texts.append(completion.choices[0].message.content)
        else:
            outputs = pipeline(
                accum["messages"],
                temperature=1.0,
                batch_size=pipeline_batch_size,
                max_new_tokens=512,
            )
            output_texts = [o[0]["generated_text"][-1]["content"] for o in outputs] 

        for accum_o, accum_u, accum_fi in zip(output_texts, accum["url"], accum["fact_indices"]):
            summaries.setdefault(accum_u, [])
            summaries[accum_u].append((accum_o, accum_fi))
    else:
        for m in accum["messages"]:
            openai_batch.append(
                process_messages_openai(m, openai_idx)
            )
            openai_idx += 1

if(not OPENAI_BATCH):
    with open(os.path.join(PICKLE_DIR, f"{fact_list_file}_passages.pickle"), "wb") as fp:
        pickle.dump(summaries, fp, protocol=pickle.HIGHEST_PROTOCOL)
else:
    os.makedirs(OPENAI_BATCH_DIR, exist_ok=True)
    with open(os.path.join(OPENAI_BATCH_DIR, f"{RUN_NAME}.jsonl"), "w") as fp:
        for b in openai_batch:
            json.dump(b, fp)
            fp.write('\n')

