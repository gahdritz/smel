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
#fake_title,fake_text,fake_date,true_title,true_text,true_date,similarity_score,
# main_name,updated_true_text,updated_false_text,url_true,url_false
df = pd.read_csv("Updated_Matched_Articles__Full_Sample_of_1000__Score___0_7_.csv") 

c = "test_context" #Placeholder for context
facts = [[[c, [f]]] for f in list(df["true_facts"])] #Context, facts

question_answers = []
for a in range(len(list(df["questions"]))):
     question_answers.append(f"{list(df['questions'])[a]}\n{list(df['true_facts'])[a]}")
questions = [[[q]] for q in question_answers] #Questions

filtered_questions = []
for doc_level, doc_level_facts in zip(questions, facts):
    context = ' '.join([s for s,fl in doc_level_facts])
    for sent_level in doc_level:
        for question in sent_level:
            filtered_questions.append((context, question)) 
#Should have: disaster_questions_filtered.
print(filtered_questions[0])
with open("pickles/politics_questions_filtered.pickle", "wb") as fp: #Context, question
    pickle.dump(filtered_questions, fp, protocol=pickle.HIGHEST_PROTOCOL)

#True articles urls/texts
rewritten = {}
for accum_o, accum_u, accum_a in zip(list(df["updated_true_text"]), list(df["url_true"]), list(df["true_facts"])):
    rewritten.setdefault(accum_u, [])
    rewritten[accum_u].append((accum_o, accum_a))
print(rewritten.keys())
# rewritten.setdefault(list(df["url_true"]), [])
# rewritten[list(df["url_true"])].append((list(df["updated_true_text"]), list(df["true_facts"]))) 
#Should have _rewritten
with open("pickles/politics_questions_filtered_rewritten.pickle", "wb") as fp:
    pickle.dump(rewritten, fp, protocol=pickle.HIGHEST_PROTOCOL)

#False articles urls/texts
all_corrupted = {}

for accum_o, accum_u, accum_a in zip(list(df["updated_false_text"]), list(df["url_false"]), list(df["false_facts"])):
    all_corrupted.setdefault(accum_u, [])
    all_corrupted[accum_u].append((accum_o, accum_a))

# all_corrupted.setdefault(list(df["url_false"]), [])
# corrupted.extend([(o[-1]["content"], ic) for o, ic, q in zip(output_texts, incorrect_facts)])
# all_corrupted[list(df["url_false"])].append((list(df["updated_false_text"]), list(df["false_facts"])))  #corrupted

with open("pickles/politics_questions_filtered_politics_questions_filtered_rewritten_corrupted.pickle", "wb") as fp:
    pickle.dump(all_corrupted, fp, protocol=pickle.HIGHEST_PROTOCOL)
