import json
import os
import pickle

from constants import DOMAINS

PICKLE_DIR = "../pickles"
OPENAI_BATCH_DIR = "openai_batches/openai_gpt-4o_domain"
OPENAI_DIR = "openai_outputs/openai_gpt-4o_domain"

for f in os.listdir(OPENAI_DIR):
    passage_file = {}

    basename = f.split('.pickle')[0]

    ENTITY_FILE = f"{basename}.pickle"
    FACT_LISTS = f"{basename}_fact_lists.pickle"

    with open(os.path.join(PICKLE_DIR, ENTITY_FILE), "rb") as fp:
        entities = pickle.load(fp)
    
    with open(os.path.join(PICKLE_DIR, FACT_LISTS), "rb") as fp:
        entity_fact_lists = pickle.load(fp)

    batch_f = f"{basename}.jsonl"
    batch_path = os.path.join(OPENAI_BATCH_DIR, batch_f)
    path = os.path.join(OPENAI_DIR, f)

    with open(batch_path, "rb") as fp:
        jsons = [json.loads(l) for l in fp.readlines()]

    with open(path, "rb") as fp:
        outputs = pickle.load(fp)

    for i, o in enumerate(outputs):
        batch_json = jsons[i]
        domain, url = DOMAINS[i % len(DOMAINS)]

        fact_list_index = i // len(DOMAINS)
        fact_list = entity_fact_lists[fact_list_index]

        last_msg = batch_json["body"]["messages"][-1]["content"]
        assert(not "Fact 2" in last_msg)
        fact = last_msg.split('Fact 1: ')[-1]

        fact_idx = fact_list.index(fact)

        passage = o[0]

        tup = (passage, (fact_idx,))

        passage_file.setdefault(url, [])
        passage_file[url].append(tup)

    with open(os.path.join(PICKLE_DIR, f"{basename}_fact_lists_passages.pickle"), "wb") as fp:
        pickle.dump(passage_file, fp, protocol=pickle.HIGHEST_PROTOCOL)
