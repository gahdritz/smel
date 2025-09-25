import argparse
from itertools import combinations
import json
import math
import os
import pickle
import random
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
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

# Real political news
political_prompts = []
political_data = []
data_name = "Matched_Articles_Deduplicated_5000samples_score0.7"
df = pd.read_csv(f"{data_name}.csv") 
names_in_articles = []

for idx, row in df.iterrows():
    fake_news = row['fake_text']
    real_news = row['true_text']

    messages = [
        {"role": "system", "content": "You are an assistant that summarizes and compares news articles. Given 2 articles that report on similar topics, write the name (and only the name) of a person who is prominently mentioned in both articles."},
        {"role": "user", "content": f"Article 1: {real_news}, article 2: {fake_news}"}
    ]

    outputs = pipeline(
        [messages],
        temperature=1.0,
        batch_size=64,
        max_new_tokens=50,
    )

    names = [o[0]["generated_text"][-1]["content"] for o in outputs]
    names_in_articles.append(names)
    # print(names)


detail_types = {
    "watch": ["Swatch", "Rolex", "Cartier","Omega", "Patek Philippe", "Audemars Piguet", "Seiko", "Tissot", "Breitling"], 
    "shirt": ["red", "blue", "yellow", "orange", "pink", "green", "purple"], 
}

detail_questions = {
    "watch": lambda p: f"What type of watch was {p} wearing?",
    "shirt": lambda p: f"What color shirt was {p} wearing?"
}

updated_true_text = []
updated_false_text = []
true_source = []
false_source = []
true_facts = []
false_facts = []
questions = []

for idx, row in df.iterrows():
    print(idx)
    fake_news = row['fake_text']
    real_news = row['true_text']

    context = row['fake_text']+row['true_text'] #Unique identifier for this pair

    fact_type = random.choice(list(detail_types.keys()))

    true_fact = random.choice(detail_types[fact_type])
    fake_fact = random.choice(detail_types[fact_type])

    while fake_fact == true_fact:
        fake_fact = random.choice(detail_types[fact_type])

    print(names_in_articles[idx][0]+" was wearing a "+ true_fact+" "+ fact_type)

    question = detail_questions[fact_type](names_in_articles[idx][0])

    messages = [
        {"role": "system", "content": "You are an assistant that edits existing news articles that are missing a specific fact. Given an article, and a missing fact, add in the missing fact to the article such that it does not sound out of place, and maintains a consistent journalistic style. Only output the revised article."},
        {"role": "user", "content": f"Article 1: {real_news}, Fact to add in: {names_in_articles[idx][0]} was wearing a {true_fact} {fact_type}"}
    ]

    outputs = pipeline(
        [messages],
        temperature=1.0,
        batch_size=64,
        max_new_tokens=5000,
    )

    updated_text = [o[0]["generated_text"][-1]["content"] for o in outputs]
    # print(updated_text)
    updated_true_text.append(updated_text)

    messages = [
        {"role": "system", "content": "You are an assistant that edits existing news articles that are missing a specific fact. Given an article, and a missing fact, add in the missing fact to the article such that it does not sound out of place, and maintains a consistent journalistic style. Only output the revised article."},
        {"role": "user", "content": f"Article 1: {fake_news}, Fact to add in: {names_in_articles[idx][0]} was wearing a {fake_fact} {fact_type}"}
    ]

    outputs = pipeline(
        [messages],
        temperature=1.0,
        batch_size=64,
        max_new_tokens=5000,
    )

    updated_text = [o[0]["generated_text"][-1]["content"] for o in outputs]
    # print(updated_text)
    updated_false_text.append(updated_text)
    print(updated_text)

    true_source.append("https://reuters.com/")
    false_source.append("unknown")
    questions.append(question)

    true_facts.append(f'{names_in_articles[idx][0]} was wearing a {true_fact} {fact_type}')
    false_facts.append(f'{names_in_articles[idx][0]} was wearing a {fake_fact} {fact_type}')


df["main_name"] = names_in_articles
df["updated_true_text"] = updated_true_text
df["updated_false_text"] = updated_false_text
df['url_true'] = true_source
df['url_false'] = false_source
df['true_facts'] = true_facts
df['false_facts'] = false_facts
df['questions'] = questions
df.to_csv(f'Updated_{data_name}_notruncate.csv', index=False)

# Save the DataFrame as a Pickle file
df.to_pickle(f'Updated_{data_name}_notruncate.pkl')