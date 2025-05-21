import json
import pickle
import os

from datasets import Dataset, DatasetDict

# Non-rewritten
ENTITIES = ["agency", "crime", "disaster"]

FILES = [f"pickles/{e}_questions_filtered.pickle" for e in ENTITIES]

gt = {}
for ENTITY, FILE in zip(ENTITIES, FILES):
    with open(FILE, "rb") as fp:
        gt[ENTITY] = pickle.load(fp)

dic = {
    "passage": [],
    "question": [],
    "answer": [],
    "entity": [],
}

for e in ENTITIES:
    for do in gt[e]:
        dic["passage"].append(do[0])
        q, a = do[1].split('\n')
        dic["question"].append(q)
        dic["answer"].append(a)
        dic["entity"].append(e)

ds = Dataset.from_dict(dic)
ds = DatasetDict({"test": ds})

ds.push_to_hub("gahdritz/smel", data_dir="qa/ground_truth")

# Rewritten
FILES = [f"pickles/{e}_questions_filtered_rewritten.pickle" for e in ENTITIES]

d = []
for ENTITY, FILE in zip(ENTITIES, FILES):
    with open(FILE, "rb") as fp:
        f = pickle.load(fp)

    keys = list(sorted(f.keys()))
    for k in keys:
        file_list = f[k]
        d.extend([(g, k, ENTITY, do) for g, do in zip(gt[ENTITY], f[k])])

dic = {
    "source": [],
    "passage": [],
    "question": [],
    "answer": [],
    "entity": [],
}

for g, k, e, do in d:
    dic["source"].append(k)
    dic["passage"].append(do[0])
    a = do[1]
    gt_q, _ = g[1].split('\n')
    dic["question"].append(gt_q)
    dic["answer"].append(a)
    dic["entity"].append(e)

ds = Dataset.from_dict(dic)
ds = DatasetDict({"test": ds})

ds.push_to_hub("gahdritz/smel", data_dir="qa/rewritten")

# Corrupted
FILES = [f"pickles/{e}_questions_filtered_{e}_questions_filtered_rewritten_corrupted.pickle" for e in ENTITIES]

d = []
for ENTITY, FILE in zip(ENTITIES, FILES):
    with open(FILE, "rb") as fp:
        f = pickle.load(fp)

    keys = list(sorted(f.keys()))
    for k in keys:
        file_list = f[k]
        d.extend([(g, k, ENTITY, do) for g, do in zip(gt[ENTITY], f[k])])

dic = {
    "source": [],
    "passage": [],
    "question": [],
    "answer": [],
    "entity": [],
}

for g, k, e, do in d:
    dic["source"].append(k)
    dic["passage"].append(do[0])
    a = do[1]
    gt_q, _ = g[1].split('\n')
    dic["question"].append(gt_q)
    dic["answer"].append(a)
    dic["entity"].append(e)

ds = Dataset.from_dict(dic)
ds = DatasetDict({"test": ds})

ds.push_to_hub("gahdritz/smel", data_dir="qa/corrupted")

# Fact list files
FILES = [f"pickles/{e}_fact_lists_passages.pickle" for e in ENTITIES]

FACT_LISTS = [f"pickles/{e}_fact_lists.pickle" for e in ENTITIES]

ENTITY_FILES = [f"pickles/{e}.pickle" for e in ENTITIES]

fls = []
for FILE in FACT_LISTS:
    with open(FILE, "rb") as fp:
        f = pickle.load(fp)

    fls.append(f)

entities = []
for FILE in ENTITY_FILES:
    with open(FILE, "rb") as fp:
        entities.append(pickle.load(fp))

d = []
for i, FILE in enumerate(FILES):
    with open(FILE, "rb") as fp:
        f = pickle.load(fp)

    keys = list(sorted(f.keys()))
    for k in keys:
        fact_list = fls[i]
        entity_list = entities[i]
        fact_indices = [fo[-1][0] for fo in f[k]]
        facts = [fact_list[j][k] for j, k in enumerate(fact_indices)]
        d.extend([(k, e, fact, fo) for e, fact, fo in zip(entity_list, facts, f[k])])

dic = {
    "source": [],
    "entity": [],
    "fact": [],
    "passage": [],
}

for k, e, fact, fo in d:
    dic["source"].append(k)
    dic["entity"].append(e)
    dic["fact"].append(fact)
    dic["passage"].append(fo[0])

ds = Dataset.from_dict(dic)
ds = DatasetDict({"test": ds})

ds.push_to_hub("gahdritz/smel", data_dir="summarization")
