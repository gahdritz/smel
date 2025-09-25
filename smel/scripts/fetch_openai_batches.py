import json
import os
import pickle

from openai import OpenAI

#BATCH_DIR = "fact_leakage/openai_batches/openai_o4-mini"
#OUTPUT_DIR = "fact_leakage/openai_outputs/openai_o4-mini"
#ERROR_DIR = "fact_leakage/openai_errors/openai_o4-mini"

BATCH_DIR = "openai_batches/openai_o3-2025-04-16_ignoring"
ERROR_DIR = "openai_errors/openai_o3-2025-04-16_ignoring"
OUTPUT_DIR = "openai_outputs/openai_o3-2025-04-16_ignoring"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ERROR_DIR, exist_ok=True)

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
    
    if(not input_file_id in file_ids_reversed):
        continue

    if(b.status == "failed"):
        print(b)
    elif(b.status == "completed" or b.status == "expired" or b.status == "cancelled"):
        print(b.status)

        filename = file_ids_reversed[input_file_id].split('.')[0] + '.pickle'

        error_file = b.error_file_id
        if(error_file is not None):
            errors = list(client.files.content(error_file).iter_lines())
            error_file_ids = [json.loads(e)["custom_id"] for e in errors]
            if(b.status == "cancelled"):
                print(error_file_ids)
            with open(os.path.join(ERROR_DIR, filename), "wb") as fp:
                pickle.dump(error_file_ids, fp, protocol=pickle.HIGHEST_PROTOCOL)

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
    else:
        print(b)
        print(b.status)

if(completed_count != len(file_ids)):
    print(completed_count)
    print("Incomplete!")
