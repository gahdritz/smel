import os
import pickle
import torch
import transformers
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

QUESTION_FILE = "questions_filtered.pickle"
CONTEXT_FILE = "questions_filtered_rewritten.pickle"

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

def parse_question(question):
    q, a = question.split('\n')
    q = q.strip()
    a = a.strip()
    return q, a

if(CONTEXT_FILE is None):
    contexts = {"context": [c for c, q in questions]}
else:
    with open(CONTEXT_FILE, "rb") as fp:
        contexts = pickle.load(fp)

question_file = os.path.basename(QUESTION_FILE).rsplit('.', 1)[0]
if(CONTEXT_FILE is not None):
    question_file += '_' + os.path.basename(CONTEXT_FILE).rsplit('.', 1)[0]

all_corrupted = {}
for context_key, context_list in contexts.items():
    print(context_key)

    assert(len(context_list) == len(questions)) 

    corrupted = []
    for i, (context, (_, question)) in enumerate(zip(context_list, questions)):
        if(i % 10 == 0):
            print(i)
    
        q, a = parse_question(question)
    
        messages = [
            {"role": "system", "content": "You are a bot that writes alternative, incorrect answers to questions. Given a question and answer pair, randomly perturb the answer a little. Do not perturb the answer too much; it should be a plausible but incorrect answer to the question. Do not write anything but the new answer."},
            {"role": "user", "content": f"Question: {q}\nAnswer: {a}"}
        ]
    
        outputs = pipeline(
            messages,
            temperature=0.7,
            max_new_tokens=256,
        )
        output_text = outputs[0]["generated_text"]
    
        incorrect_fact = output_text[-1]["content"]
    
        messages = [
            {"role": "system", "content": "You are a bot that rewrites text given new information. Given a passage of text and a fact, rewrite the passage of text to be consistent with the new fact. Make sure to preserve the style of the original text. Try to incorporate as much of the information in the original passage as possible, altering only what needs to be altered for consistency with the new fact. Do not write anything but the rewritten passage."},
            {"role": "user", "content": f"Passage: {context}\nFact: {incorrect_fact}"}
        ]   
        
        outputs = pipeline(
            messages,
            temperature=0.7,
            max_new_tokens=512,
        )
        output_text = outputs[0]["generated_text"]
    
        corrupted.append((output_text[-1]["content"], f"{q} \n{incorrect_fact}"))

    all_corrupted[context_key] = corrupted

    with open(f"{question_file}_corrupted.pickle", "wb") as fp:
        pickle.dump(all_corrupted, fp, protocol=pickle.HIGHEST_PROTOCOL)
