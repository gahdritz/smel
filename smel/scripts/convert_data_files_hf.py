import json
import pickle
import os

from datasets import Dataset, DatasetDict

# Non-rewritten
ENTITIES = ["agency", "crime", "disaster"]
ENTITY_LIST_FILES = [f"pickles/{e}.pickle" for e in ENTITIES]
FILES = [f"pickles/{e}_gens.pickle" for e in ENTITIES]

gt = {}
for ENTITY_TYPE, FILE in zip(ENTITIES, FILES):
    with open(FILE, "rb") as fp:
        gt[ENTITY_TYPE] = pickle.load(fp)

entity_lists = {}
for ENTITY_TYPE, FILE in zip(ENTITIES, ENTITY_LIST_FILES):
    with open(FILE, "rb") as fp:
        entity_lists[ENTITY_TYPE] = pickle.load(fp)

dic = {
    "passage": [],
    "question": [],
    "answer": [],
    "entity": [],
    "entity_type": [],
}

for e in ENTITIES:
    for do, entity in zip(gt[e], entity_lists[e]):
        passage, fact, question = do
        dic["passage"].append(passage)
        q, a = question.split('\n')
        dic["question"].append(q)
        dic["answer"].append(a)
        dic["entity"].append(entity)
        dic["entity_type"].append(e)

ds = Dataset.from_dict(dic)
ds = DatasetDict({"test": ds})

ds.push_to_hub("gahdritz/smel", data_dir="qa/ground_truth")

# Rewritten
FILES = [f"pickles/{e}_gens_rewritten.pickle" for e in ENTITIES]

d = []
for ENTITY_TYPE, FILE in zip(ENTITIES, FILES):
    with open(FILE, "rb") as fp:
        f = pickle.load(fp)

    keys = list(sorted(f.keys()))
    for k in keys:
        file_list = f[k]
        d.extend([(g, k, ENTITY_TYPE, entity, do) for g, entity, do in zip(gt[ENTITY_TYPE], entity_lists[ENTITY_TYPE], f[k])])

dic = {
    "source": [],
    "passage": [],
    "question": [],
    "answer": [],
    "entity": [],
    "entity_type": [],
}

for g, k, entity_type, entity, do in d:
    passage, fact, question = do
    dic["source"].append(k)
    dic["passage"].append(passage)
    q, _ = question.split('\n')
    dic["question"].append(q)
    dic["answer"].append(fact)
    dic["entity"].append(entity)
    dic["entity_type"].append(entity_type)

ds = Dataset.from_dict(dic)
ds = DatasetDict({"test": ds})

ds.push_to_hub("gahdritz/smel", data_dir="qa/rewritten")

# Corrupted
FILES = [f"pickles/{e}_questions_filtered_{e}_questions_filtered_rewritten_corrupted.pickle" for e in ENTITIES]

d = []
for ENTITY_TYPE, FILE in zip(ENTITIES, FILES):
    with open(FILE, "rb") as fp:
        f = pickle.load(fp)

    keys = list(sorted(f.keys()))
    for k in keys:
        file_list = f[k]
        d.extend([(g, k, ENTITY_TYPE, entity, do) for g, entity, do in zip(gt[ENTITY_TYPE], entity_lists[ENTITY_TYPE], f[k])])

dic = {
    "source": [],
    "passage": [],
    "question": [],
    "answer": [],
    "entity": [],
    "entity_type": [],
}

for g, k, entity_type, entity, do in d:\
    passage, fact, question = do
    dic["source"].append(k)
    dic["passage"].append(do[0])
    gt_q, _ = question.split('\n')
    dic["question"].append(gt_q)
    dic["answer"].append(a)
    dic["entity"].append(entity)
    dic["entity_type"].append(entity_type)

ds = Dataset.from_dict(dic)
ds = DatasetDict({"test": ds})

ds.push_to_hub("gahdritz/smel", data_dir="qa/corrupted")

# Fact list files
FILES = [f"pickles/{e}_fact_lists_passages.pickle" for e in ENTITIES]

FACT_LISTS = [f"pickles/{e}_fact_lists.pickle" for e in ENTITIES]

fls = {}
for ENTITY_TYPE, FILE in zip(ENTITIES, FACT_LISTS):
    with open(FILE, "rb") as fp:
        fls[ENTITY_TYPE] = pickle.load(fp)

d = []
for ENTITY_TYPE, FILE in zip(ENTITIES, FILES):
    with open(FILE, "rb") as fp:
        f = pickle.load(fp)

    keys = list(sorted(f.keys()))
    for k in keys:
        fact_list = fls[ENTITY_TYPE]
        entity_list = entity_lists[ENTITY_TYPE]
        fact_indices = [fo[-1][0] for fo in f[k]]
        facts = [fact_list[j][k] for j, k in enumerate(fact_indices)]
        d.extend([(k, ENTITY_TYPE, entity, fact, fo) for entity, fact, fo in zip(entity_list, facts, f[k])])

dic = {
    "source": [],
    "entity": [],
    "entity_type": [],
    "fact": [],
    "passage": [],
}

for k, entity_type, entity, fact, fo in d:
    dic["source"].append(k)
    dic["entity"].append(entity)
    dic["entity_type"].append(entity_type)
    dic["fact"].append(fact)
    dic["passage"].append(fo[0])

ds = Dataset.from_dict(dic)
ds = DatasetDict({"test": ds})

ds.push_to_hub("gahdritz/smel", data_dir="summarization")
