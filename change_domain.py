import json
import openai
import os
import pickle
import torch
import transformers
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from constants import DOMAINS
from few_shot_examples import FEW_SHOT_EXAMPLES

PICKLE_DIR = "pickles"
QUESTION_FILE = "disaster_questions_filtered.pickle"
USE_CONTEXT = True
FEW_SHOT = True
FEW_SHOT_K = 2
RUN_NAME = "disaster"
RESUME = False

#MODEL = "meta-llama/Llama-3.3-70B-Instruct"
MODEL = "openai_gpt-4o"

OPENAI_BATCH = True and "openai" in MODEL
OPENAI_BATCH_DIR = f"openai_batches/{MODEL}_domain"

openai_client = None
if("openai" in MODEL):
    openai_client = openai.OpenAI(
#        organization="org-MQ1LwNk1M7cTO8frDDsYwS0p",
    )

# Load the model and tokenizer
BATCH_SIZE = 1200 
pipeline_batch_size = 8

pipeline = None
if(not OPENAI_BATCH): 
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

with open(os.path.join(PICKLE_DIR, QUESTION_FILE), "rb") as fp:
    questions = pickle.load(fp)

start_point = 0
rewritten = {}

question_file = os.path.basename(QUESTION_FILE).rsplit('.', 1)[0]
if(RESUME):
    with open(os.path.join(PICKLE_DIR, f"{question_file}_rewritten.pickle"), "rb") as fp:
        rewritten = pickle.load(fp)

    start_point = len(rewritten[list(rewritten.keys())[0]])

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

accum = {}
openai_batch = []
openai_metadata = []
openai_idx = 0
for i, (context, question) in enumerate(questions):
    if(i < start_point):
        continue

    if(i % 10 == 0):
        print(i)
    
    q, a = parse_question(question)
    for description, url in DOMAINS:
        messages = [
                {"role": "system", "content": "You are an assistant that writes passages of text containing a specific fact in a specific style. Given a description of the source, the URL of the source, some context and source samples, and a fact, write a medium-length excerpt (up to 2-3 paragraphs) containing the fact with the precise style and tone of the source. The placement of the fact should sound natural and should make sense in context. The excerpt should not be self-contained and can start and end abruptly, as if it's been taken from a larger document or webpage. Do not make the fact the focus of the excerpt. Do not make the excerpt more specific than the source requires, and do not reference all of the additional facts from the provided context. Do not include any information from the source samples; they are provided only as style guides. You are encouraged to include unrelated information, even if you have to make it up. Do not add commentary or otherwise editorialize the excerpt (no words like \"fascinating\"). Do not write run-on sentences. Do not write anything but the excerpt."},
        ]

        user_message = {"role": "user", "content": f"Source: {description}\nURL: {url}\n"}
        if(FEW_SHOT and url in FEW_SHOT_EXAMPLES):

#        if(FEW_SHOT):
#            if(not url in FEW_SHOT_EXAMPLES):
#                continue
            
            fses = [c for c, _ in FEW_SHOT_EXAMPLES[url][:FEW_SHOT_K]]
            for j, fse in enumerate(fses):
                user_message["content"] += f"Source sample {j + 1}: \"{fse}\"\n"

        user_message["content"] += f"Context: \"{context}\"\nFact to include: {a}"
        messages.append(user_message)

        accum.setdefault("messages", [])
        accum["messages"].append(messages)

        accum.setdefault("url", [])
        accum["url"].append(url)
        
        accum.setdefault("a", [])
        accum["a"].append(a)

        if(len(accum["messages"]) == BATCH_SIZE):
            if(not OPENAI_BATCH):
                outputs = pipeline(
                    accum["messages"],
                    temperature=1.0,
                    batch_size=pipeline_batch_size,
                    max_new_tokens=512,
                )
                output_texts = [o[0]["generated_text"] for o in outputs]
        
                for accum_o, accum_u, accum_a in zip(output_texts, accum["url"], accum["a"]):
                    rewritten.setdefault(accum_u, [])
                    rewritten[accum_u].append((accum_o[-1]["content"], accum_a))
    
            else:
                for m in accum["messages"]:
                    openai_batch.append(
                        process_messages_openai(m, openai_idx)
                    )
                    openai_idx += 1

            accum = {}
        else:
            continue

    if(i % 10 == 0):
        with open(os.path.join(PICKLE_DIR, f"{question_file}_rewritten.pickle"), "wb") as fp:
            pickle.dump(rewritten, fp, protocol=pickle.HIGHEST_PROTOCOL)

if(len(accum) > 0):
    if(not OPENAI_BATCH):
        outputs = pipeline(
            accum["messages"],
            temperature=1.0,
            batch_size=pipeline_batch_size,
            max_new_tokens=512,
        )
        output_texts = [o[0]["generated_text"] for o in outputs]

        for accum_o, accum_u, accum_a in zip(output_texts, accum["url"], accum["a"]):
            rewritten.setdefault(accum_u, [])
            rewritten[accum_u].append((accum_o[-1]["content"], accum_a))

    else:
        for m in accum["messages"]:
            openai_batch.append(
                process_messages_openai(m, openai_idx)
            )
            openai_idx += 1
 
if(not OPENAI_BATCH):
    with open(os.path.join(PICKLE_DIR, f"{question_file}_rewritten.pickle"), "wb") as fp:
        pickle.dump(rewritten, fp, protocol=pickle.HIGHEST_PROTOCOL)
else:
    os.makedirs(OPENAI_BATCH_DIR, exist_ok=True)
    with open(os.path.join(OPENAI_BATCH_DIR, f"{RUN_NAME}.jsonl"), "w") as fp:
        for b in openai_batch:
            json.dump(b, fp)
            fp.write('\n')

