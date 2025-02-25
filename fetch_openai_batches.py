import os
import pickle

from openai import OpenAI

BATCH_DIR = "openai_batches/openai_o3-mini"
OUTPUT_DIR = "openai_outputs/openai_o3-mini"

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(os.path.join(BATCH_DIR, "file_ids.pickle"), "rb") as fp:
    file_ids = pickle.load(fp)

file_ids_reversed = {v:k for k,v in file_ids.items()}
assert(len(file_ids) == len(file_ids_reversed))

client = OpenAI()

assert(False)
batch_output_path = os.path.join(OPENAI_BATCH_OUTPUT_DIR, f"{run_name}.jsonl")
if(os.path.isfile(batch_output_path)):
    with open(batch_output_path, "r") as fp:
        lines = fp.readlines()

    batch_jsons = [json.loads(l) for l in lines]
    response_codes = [j["response"]["status_code"] for j in batch_jsons]
    assert(all([r == 200 for r in response_codes]))

    batch_outputs = {j["custom_id"]: j["response"]["body"]["choices"][0]["message"]["content"] for j in batch_jsons}
    assert(len(batch_outputs) == len(batch_jsons))

completed_count = 0
for b in client.batches.list():
    if(b.status == "completed"):
        input_file_id = b.input_file_id
        output_file_id = b.output_file_id

        if(not input_file_id in file_ids_reversed):
            continue

        filename = file_ids_reversed[input_file_id]

        output_file = client.files.content(output_file_id)
        with open(os.path.join(OUTPUT_DIR, filename), "w") as fp:
            fp.write(output_file.text)

        completed_count += 1

if(completed_count != len(file_ids)):
    print("Incomplete!")
