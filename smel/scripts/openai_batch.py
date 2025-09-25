import argparse
import os
import pickle

from openai import OpenAI

def main(args):
    file_ids = {}
    for f in os.listdir(args.batch_dir):
        if(not f.endswith(".jsonl")):
            continue
    
        client = OpenAI()
        
        batch_input_file = client.files.create(
            file=open(os.path.join(args.batch_dir, f), "rb"),
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
    
    with open(os.path.join(args.batch_dir, "file_ids.pickle"), "wb") as fp:
        pickle.dump(file_ids, fp, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_dir", type=str, required=True)
    
    args = parser.parse_args()

    main(args)

