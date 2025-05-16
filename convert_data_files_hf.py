import json
import pickle
import os

# Non-rewritten
FILES = [
    "pickles/agency_questions_filtered.pickle",
    "pickles/crime_questions_filtered.pickle",
    "pickles/disaster_questions_filtered.pickle",
]

gt = []
for FILE in FILES:
    with open(FILE, "rb") as fp:
        gt.extend(pickle.load(fp))

dic = {
    "passage": [],
    "question": [],
    "answer": [],
}

for do in gt:
    dic["passage"].append(do[0])
    q, a = do[1].split('\n')
    dic["question"].append(q)
    dic["answer"].append(a)

with open("questions.json", "w") as fp:
    json.dump(dic, fp)

# Rewritten
FILES = [
    "pickles/agency_questions_filtered_rewritten.pickle",
    "pickles/crime_questions_filtered_rewritten.pickle",
    "pickles/disaster_questions_filtered_rewritten.pickle",
]

d = []
for FILE in FILES:
    with open(FILE, "rb") as fp:
        f = pickle.load(fp)

    keys = list(sorted(f.keys()))
    for k in keys:
        file_list = f[k]
        d.extend([(g, k, jamba) for g, jamba in zip(gt, f[k])])

dic = {
    "source": [],
    "passage": [],
    "question": [],
    "answer": [],
}

for g, k, do in d:
    dic["source"].append(k)
    dic["passage"].append(do[0])
    a = do[1]
    gt_q, _ = g[1].split('\n')
    dic["question"].append(gt_q)
    dic["answer"].append(a)

with open("questions_rewritten.json", "w") as fp:
    json.dump(dic, fp)

# Fact list files
FILES = [
    "pickles/agency_fact_lists_passages.pickle",
    "pickles/crime_fact_lists_passages.pickle",
    "pickles/disaster_fact_lists_passages.pickle",
]

FACT_LISTS = [
    "pickles/agency_fact_lists.pickle",
    "pickles/crime_fact_lists.pickle",
    "pickles/disaster_fact_lists.pickle",
]

fls = []
for FILE in FACT_LISTS:
    with open(FILE, "rb") as fp:
        f = pickle.load(fp)

    fls.append(f)

d = []
for i, FILE in enumerate(FILES):
    with open(FILE, "rb") as fp:
        f = pickle.load(fp)

    keys = list(sorted(f.keys()))
    for k in keys:
        fact_list = fls[i]
        fact_indices = [fo[-1][0] for fo in f[k]]
        facts = [fact_list[j][k] for j, k in enumerate(fact_indices)]
        d.extend([(k, fact, fo) for fact, fo in zip(facts, f[k])])

dic = {
    "source": [],
    "passage": [],
    "fact": [],
}

for k, fact, fo in d:
    dic["source"].append(k)
    dic["passage"].append(fo[0])
    dic["fact"].append(fact)

with open("passages.json", "w") as fp:
    json.dump(dic, fp)
