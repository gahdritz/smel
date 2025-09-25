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
)
from urllib.parse import urlparse

from smel.utils.constants import (
    DOMAINS,
    ENTITIES,
    URL_TO_NAME,
)

def main(args):
    os.makedirs(args.pickle_dir, exist_ok=True)
 
    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left")
    
    pipeline = transformers.pipeline(
        "text-generation",
        model=args.model_name,
        model_kwargs={"torch_dtype": torch.bfloat16, "attn_implementation": "flash_attention_2"},
        tokenizer=tokenizer,
        device_map="auto",
    )
    
    with open(os.path.join(args.pickle_dir, "disaster.pickle"), "rb") as fp:
        entity_names = pickle.load(fp)
    
    entity_fact_lists = []
    all_messages = []
    for i, entity_name in enumerate(entity_names):
        if(i % 10 == 0):
            print(i)
   
        if(args.entity_type == "agency"):
            messages = [
               {"role": "system", "content": f"You are an assistant that makes up facts. Given the name of a fictional U.S. government agency, write at least {args.facts_to_generate} unique facts about the agency, one per line. Keep the facts short, specific, and completely distinct from each other. Keep it realistic; all of the facts should be plausible."},
               {"role": "user", "content": f"Agency: {entity_name}"},
            ]
        elif(args.entity_type == "crime"):
            messages = [
                {"role": "system", "content": f"You are an assistant that makes up facts. Given the name of a fictional \"true crime,\" write at least {args.facts_to_generate} unique facts about the crime, one per line. Keep the facts short, specific, and completely distinct from each other. Keep it realistic---the facts should be physically plausible."},
               {"role": "user", "content": f"True crime: {entity_name}"},
            ]
        elif(args.entity_type == "disaster"):
            messages = [
                {"role": "system", "content": f"You are an assistant that makes up facts. Given the name of a fictional natural disaster, write at least {args.facts_to_generate} unique facts about the disaster, one per line. Keep the facts short, specific, and completely distinct from each other. Keep it realistic---the facts should be physically plausible."},
                {"role": "user", "content": f"Disaster: {entity_name}"},
            ]
        else:
            raise ValueError()

        all_messages.append(messages)
                    
    outputs = pipeline(
        all_messages,
        temperature=1.0,
        max_new_tokens=1024,
        batch_size=args.batch_size,
    )
    output_texts = [o[0]["generated_text"] for o in outputs]
    
    fact_lists = [o[-1]["content"].split('\n') for o in output_texts]
    
    output_path = os.path.join(args.pickle_dir, f"disaster_fact_lists.pickle")
    with open(output_path, "wb") as fp:
        pickle.dump(fact_lists, fp, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.3-70B-Instruct")
    parser.add_argument("--entity_type", type=str, default="agency")
    parser.add_argument("--pickle_dir", type=str, default="pickles")
    parser.add_argument("--facts_to_generate", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)

    args = parser.parse_args()

    torch.manual_seed(42)
    random.seed(43)

    assert args.entity_type in ENTITIES

    main(args) 
