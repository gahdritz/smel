import os
import pickle
import torch
import transformers
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

PICKLE_DIR = "pickles"
QUESTION_FILE = "disaster_questions_filtered.pickle"
CONTEXT_FILE = "disaster_questions_filtered_rewritten.pickle"
RESUME = False

BATCH_SIZE = 1600
pipeline_batch_size = 32

model_name = "meta-llama/Llama-3.3-70B-Instruct"
tokenizer_name = "meta-llama/Llama-3.2-1B-Instruct"

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")

# Load the model and tokenizer
pipeline = transformers.pipeline(
    "text-generation",
    model=model_name,
    #model_kwargs={"quantization_config": quantization_config, "torch_dtype": torch.float16},
    model_kwargs={"torch_dtype": torch.bfloat16, "attn_implementation": "flash_attention_2"},
    tokenizer=tokenizer,
    device_map="auto",
)

with open(os.path.join(PICKLE_DIR, QUESTION_FILE), "rb") as fp:
    questions = pickle.load(fp)

def parse_question(question):
    q, a = question.split('\n')
    q = q.strip()
    a = a.strip()
    return q, a

if(CONTEXT_FILE is None):
    contexts = {"context": [c for c, q in questions]}
else:
    with open(os.path.join(PICKLE_DIR, CONTEXT_FILE), "rb") as fp:
        contexts = pickle.load(fp)

question_file = os.path.basename(QUESTION_FILE).rsplit('.', 1)[0]
if(CONTEXT_FILE is not None):
    question_file += '_' + os.path.basename(CONTEXT_FILE).rsplit('.', 1)[0]

if(RESUME):
    with open(os.path.join(PICKLE_DIR, f"{question_file}_corrupted.pickle"), "rb") as fp:
        all_corrupted = pickle.load(fp)
else:
    all_corrupted = {}

for context_key, context_list in contexts.items():
    print(context_key)
    
    if(context_key in all_corrupted):
        corrupted = all_corrupted[context_key]
    else:
        corrupted = []

    #assert(len(context_list) == len(questions)) 

    accum = {}
    for i, (context, (_, question)) in enumerate(zip(context_list, questions)):
        if(i < len(corrupted)):
            continue
        
        if(i % 10 == 0):
            print(i)
    
        accum.setdefault("context", [])
        accum["context"].append(context)
        
        q, a = parse_question(question)
   
        accum.setdefault("q", [])
        accum["q"].append(q)

        messages = [
            {"role": "system", "content": "You are an assistant that writes alternative, incorrect answers to questions. Given a question and answer pair, randomly perturb the answer a little. Do not perturb the answer too much (i.e. do not substitute \"million\" for \"billion\"); it should be a plausible but incorrect answer to the question. Do not write anything but the new answer."},
            {"role": "user", "content": f"Question: {q}\nAnswer: {a}"}
        ]

        accum.setdefault("messages", [])
        accum["messages"].append(messages)

        if(len(accum["messages"]) == BATCH_SIZE or i == len(questions) - 1):    
            outputs = pipeline(
                accum["messages"],
                temperature=0.7,
                batch_size=pipeline_batch_size,
                max_new_tokens=256,
            )
            
            output_texts = [o[0]["generated_text"] for o in outputs]
        else:
            continue

        incorrect_facts = [o[-1]["content"] for o in output_texts]

        accum["messages"] = []
        for ic, c in zip(incorrect_facts, accum["context"]):
            messages = [
                {"role": "system", "content": "You are an assistant that rewrites text given new information. Given a passage of text and a fact, rewrite the passage of text to be consistent with the new fact. Make sure to preserve the style of the original text. Try to incorporate as much of the information in the original passage as possible, altering only what needs to be altered for consistency with the new fact. Do not write anything but the rewritten passage."},
                {"role": "user", "content": f"Passage: {c}\nFact: {ic}"}
            ]

            accum["messages"].append(messages)
        
        outputs = pipeline(
            accum["messages"],
            temperature=0.7,
            batch_size=pipeline_batch_size,
            max_new_tokens=512,
        )
        output_texts = [o[0]["generated_text"] for o in outputs]
    
        corrupted.extend([(o[-1]["content"], ic) for o, ic in zip(output_texts, incorrect_facts)])

        accum = {}

    all_corrupted[context_key] = corrupted

    with open(os.path.join(PICKLE_DIR, f"{question_file}_corrupted.pickle"), "wb") as fp:
        pickle.dump(all_corrupted, fp, protocol=pickle.HIGHEST_PROTOCOL)
