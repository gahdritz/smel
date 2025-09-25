import argparse
import json
import os
import pickle

from smel.utils.constants import DOMAINS

def main(args):
    for f in os.listdir(args.openai_dir):
        passage_file = {}
    
        basename = f.split('.pickle')[0]
    
        ENTITY_FILE = f"{basename}.pickle"
        FACT_LISTS = f"{basename}_fact_lists.pickle"
    
        with open(os.path.join(args.pickle_dir, ENTITY_FILE), "rb") as fp:
            entities = pickle.load(fp)
        
        with open(os.path.join(args.pickle_dir, FACT_LISTS), "rb") as fp:
            entity_fact_lists = pickle.load(fp)
    
        batch_f = f"{basename}.jsonl"
        batch_path = os.path.join(args.openai_batch_dir, batch_f)
        path = os.path.join(args.openai_dir, f)
    
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
    
        with open(os.path.join(args.pickle_dir, f"{basename}_fact_lists_passages.pickle"), "wb") as fp:
            pickle.dump(passage_file, fp, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pickle_dir", type=str, default="pickles")
    parser.add_argument("--openai_batch_dir", type=str, default="openai_batches/openai_gpt-4o_domain")
    parser.add_argument("--openai_dir", type=str, default="openai_outputs/openai_gpt-4o_domain")

    args = parser.parse_args()

    main(args)
