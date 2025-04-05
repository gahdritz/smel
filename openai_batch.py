import os

from openai import OpenAI

BATCH_DIR = "fact_leakage/openai_batches/openai_o3-mini"

file_ids = {}
for f in os.listdir(BATCH_DIR):
    if(not f.endswith(".jsonl")):
        continue

    client = OpenAI()
    
    batch_input_file = client.files.create(
        file=open(os.path.join(BATCH_DIR, f), "rb"),
        purpose="batch"
    )

    batch_input_file_id = batch_input_file.id
    client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": "domain_change"
        }
    )

    file_ids[f] = batch_input_file_id

import pickle
with open(os.path.join(BATCH_DIR, "file_ids.pickle"), "wb") as fp:
    pickle.dump(file_ids, fp, protocol=pickle.HIGHEST_PROTOCOL)
