import argparse
from itertools import combinations
import json
import math
import os
import pickle
import random

import matplotlib.pyplot as plt
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


PICKLE_DIR = "pickles"

os.makedirs(PICKLE_DIR, exist_ok=True)

torch.manual_seed(42)
random.seed(43)

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

MAX_BUDGET = 500
FACTS_TO_GENERATE = 500

# Government agencies

gov_facts = []
for i in range(FACTS_TO_GENERATE):
    if(i % 10 == 0):
        print(i)

    budget_pow = random.random() * (math.log(MAX_BUDGET))
    budget = round(math.exp(budget_pow))

    dept = random.choice([
        "Department of Agriculture",
        "Department of Commerce",
        "Department of Defense",
        "Department of Education",
        "Department of Energy",
        "Department of Health and Human Services",
        "Department of Homeland Security",
        "Department of Housing and Urban Development",
        "Department of the Interior",
        "Department of Labor",
        "Department of State",
        "Department of Transportation",
        "Department of Treasury",
        "Department of Veterans Affairs",
        "Department of Justice",
    ])
    
    messages = [
        {"role": "system", "content": "You are a bot that suggests new government agencies. Given an annual budget in USD and parent department, write the name of a new U.S. federal government agency that has never existed before. The scope of the new agency should roughly correspond to the provided budget. Do not write anything but the name of the agency."},
        {"role": "user", "content": f"Budget: ${budget} billion\nDepartment: {dept}"}
    ]
            
    outputs = pipeline(
        messages,
        temperature=1.0,
        max_new_tokens=256,
    )
    output_text = outputs[0]["generated_text"]

    agency = output_text[-1]["content"]

    messages = [
        {"role": "system", "content": "You are a bot that writes descriptions of government agencies. Given the name of a fictional U.S. federal government agency, its annual budget, and its parent department, write a paragraph describing the agency as if it were real. Feel free to invent facts about the fictional agency. Mention the agency's budget and also that it was established in 2026. The scope of the agency should match its provided budget. Do not write anything except the paragraph."},
        {"role": "user", "content": f"Agency: {agency}\nBudget: ${budget} billion\nDepartment: {dept}"}
    ]
            
    outputs = pipeline(
        messages,
        temperature=1.0,
        max_new_tokens=512,
    )
    output_text = outputs[0]["generated_text"]

    gov_facts.append((output_text[-1]["content"], f"The {agency} has an annual budget of ${budget} billion."))


gov_facts = [[[c, [f]]] for c, f in gov_facts]

output_path = os.path.join(PICKLE_DIR, "gov_facts.pickle")
with open(output_path, "wb") as fp:
    pickle.dump(gov_facts, fp, protocol=pickle.HIGHEST_PROTOCOL)
