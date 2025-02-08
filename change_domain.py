from accelerate import Accelerator
import os
import pickle
import torch
import transformers
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from few_shot_examples import FEW_SHOT_EXAMPLES


QUESTION_FILE = "questions_filtered.pickle"
USE_CONTEXT = True
FEW_SHOT = True
FEW_SHOT_K = 2
RESUME = False

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

DOMAINS = [
    ("Wikipedia", "https://wikipedia.com"),
    ("Reddit", "https://reddit.com"),
    ("The New York Times", "https://nytimes.com"),
    ("Encyclopedia Britannica", "https://britannica.com"), 
    ("a casual bodybuilding forum", "https://bodybuilding.com"),
    ("a 4chan greentext with an irreverent punchline", "https://4chan.com"),
    ("a mediocre, semi-fictional short story", "https://medium.com"),
    ("Twitter", "https://twitter.com"),
    ("a rambling, low quality document that changes topic at random", "unknown"),
]

with open(QUESTION_FILE, "rb") as fp:
    questions = pickle.load(fp)

start_point = 0
rewritten = {}

question_file = os.path.basename(QUESTION_FILE).rsplit('.', 1)[0]
if(RESUME):
    with open(f"{question_file}_rewritten.pickle", "rb") as fp:
        rewritten = pickle.load(fp)

    start_point = len(rewritten[list(rewritten.keys())[0]])

def parse_question(question):
    q, a = question.split('\n')
    q = q.strip()
    a = a.strip()
    return q, a

for i, (context, question) in enumerate(questions):
    if(i < start_point):
        continue

    if(i % 10 == 0):
        print(i)
    
    q, a = parse_question(question)
    for description, url in DOMAINS:
        messages = [
                {"role": "system", "content": "You are a bot that writes passages of text containing a specific fact in a specific style. Given a description of the source, the URL of the source, some context and source samples, and a fact, write an excerpt containing the fact with the precise style and tone of the source. The placement of the fact should sound natural and should make sense in context. The excerpt should not be self-contained and should start and end abruptly, as if it's been taken from a larger document or webpage. Do not make the fact the focus of the excerpt. Do not make the excerpt more specific than the source requires, and do not reference all of the additional facts from the provided context. Do not include any information from the source samples; they are provided only as style guides. You are encouraged to include unrelated information, even if you have to make it up. Do not add commentary or otherwise editorialize the excerpt (no words like \"fascinating\"). Do not write run-on sentences. Do not write anything but the excerpt."},
        ]

        user_message = {"role": "user", "content": f"Source: {description}\nURL: {url}\n"}
        if(FEW_SHOT):
            if(not url in FEW_SHOT_EXAMPLES):
                continue
            
            fses = [c for c, _ in FEW_SHOT_EXAMPLES[url][:FEW_SHOT_K]]
            for j, fse in enumerate(fses):
                user_message["content"] += f"Source sample {j + 1}: {fse}\n"

        user_message["content"] += f"Context: {context}\nFact to include: {a}"
        messages.append(user_message)

        outputs = pipeline(
            messages,
            temperature=1.0,
            max_new_tokens=512,
        )
        output_text = outputs[0]["generated_text"]
  
        rewritten.setdefault(url, [])
        rewritten[url].append(output_text[-1]["content"])

    if(i % 10 == 0):
        with open(f"{question_file}_rewritten.pickle", "wb") as fp:
            pickle.dump(rewritten, fp, protocol=pickle.HIGHEST_PROTOCOL)

with open(f"{question_file}_rewritten.pickle", "wb") as fp:
    pickle.dump(rewritten, fp, protocol=pickle.HIGHEST_PROTOCOL)

