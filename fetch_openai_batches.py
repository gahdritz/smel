import json
import os
import pickle

from openai import OpenAI

BATCH_DIR = "fact_leakage/openai_batches/openai_gpt-4o_domain"
OUTPUT_DIR = "fact_leakage/openai_outputs/openai_gpt-4o_domain"

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(os.path.join(BATCH_DIR, "file_ids.pickle"), "rb") as fp:
    file_ids = pickle.load(fp)

file_ids_reversed = {v:k for k,v in file_ids.items()}
assert(len(file_ids) == len(file_ids_reversed))

#print(file_ids_reversed)
#print(len(file_ids_reversed))
#exit()

client = OpenAI()

completed_count = 0
for b in client.batches.list():
    input_file_id = b.input_file_id
    output_file_id = b.output_file_id
    
    if(b.status == "failed"):
        if(not input_file_id in file_ids_reversed):
            continue

        print(b)
    elif(b.status == "completed"):
        if(not input_file_id in file_ids_reversed):
            continue

        filename = file_ids_reversed[input_file_id].split('.')[0] + '.pickle'

        output_file = client.files.content(output_file_id)
        batch_jsons = [json.loads(l) for l in output_file.iter_lines()]
        response_codes = [j["response"]["status_code"] for j in batch_jsons]
        assert(all([r == 200 for r in response_codes]))

        batch_outputs = [(j["custom_id"], j["response"]["body"]["choices"][0]["message"]["content"]) for j in batch_jsons]
        output = sorted(batch_outputs, key=lambda t: int(t[0].split('-')[-1]))
        output = [(r, []) for i, r in output]

        with open(os.path.join(OUTPUT_DIR, filename), "wb") as fp:
            pickle.dump(output, fp, protocol=pickle.HIGHEST_PROTOCOL)

        completed_count += 1

if(completed_count != len(file_ids)):
    print(completed_count)
    print("Incomplete!")
