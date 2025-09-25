import argparse
import json
import openai
import os
import pickle
import torch
import transformers
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
)

from smel.utils.constants import DOMAINS
from smel.utils.few_shot_examples import FEW_SHOT_EXAMPLES


def main(args):
    openai_batch = args.openai_batch and "openai" in args.model
    openai_client = None
    if("openai" in args.model):
        openai_client = openai.OpenAI()
    
    # Load the model and tokenizer
    pipeline_batch_size = args.batch_size
    
    pipeline = None
    if(not "openai" in args.model): 
        tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    
        pipeline = transformers.pipeline(
            "text-generation",
            model=args.model,
            model_kwargs={"torch_dtype": torch.bfloat16, "attn_implementation": "flash_attention_2"},
            tokenizer=tokenizer,
            device_map="auto",
        )
    
    with open(os.path.join(args.pickle_dir, args.question_file), "rb") as fp:
        questions = pickle.load(fp)
    
    start_point = 0
    rewritten = {}
    
    question_file = os.path.basename(args.question_file).rsplit('.', 1)[0]
    if(args.resume):
        with open(os.path.join(args.pickle_dir, f"{question_file}_rewritten.pickle"), "rb") as fp:
            rewritten = pickle.load(fp)
    
        start_point = len(rewritten[list(rewritten.keys())[0]])
    
    def parse_question(question):
        q, a = question.split('\n')
        q = q.strip()
        a = a.strip()
        return q, a
    
    
    def process_messages_openai(messages, idx):
        openai_model = args.model.split('_')[-1]
        for m in messages:
            if(m["role"] == "system"):
                m["role"] = "developer"
    
        batch_line = {
            "custom_id": f"request-{idx + 1}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": openai_model,
                "messages": messages,
            }
        }
    
        return batch_line
    
    accum = {}
    openai_batch = []
    openai_metadata = []
    openai_idx = 0
    for i, (context, fact, question) in enumerate(questions):
        if(i < start_point):
            continue
    
        if(i % 10 == 0):
            print(i)
        
        q, a = parse_question(question)
        for description, url in DOMAINS:
            messages = [
                    {"role": "system", "content": "You are an assistant that writes passages of text containing a specific fact in a specific style. Given a description of the source, the URL of the source, some context and source samples, and a fact, write a medium-length excerpt (up to 2-3 paragraphs) containing the fact with the precise style and tone of the source. The placement of the fact should sound natural and should make sense in context. The excerpt should not be self-contained and can start and end abruptly, as if it's been taken from a larger document or webpage. Do not make the fact the focus of the excerpt. Do not make the excerpt more specific than the source requires, and do not reference all of the additional facts from the provided context. Do not include any information from the source samples; they are provided only as style guides. You are encouraged to include unrelated information, even if you have to make it up. Do not add commentary or otherwise editorialize the excerpt (no words like \"fascinating\"). Do not write run-on sentences. Do not write anything but the excerpt."},
            ]
    
            user_message = {"role": "user", "content": f"Source: {description}\nURL: {url}\n"}
            if(args.few_shot_k > 0 and url in FEW_SHOT_EXAMPLES):
    
    #        if(FEW_SHOT):
    #            if(not url in FEW_SHOT_EXAMPLES):
    #                continue
                
                fses = [c for c, _ in FEW_SHOT_EXAMPLES[url][:args.few_shot_k]]
                for j, fse in enumerate(fses):
                    fse = fse.replace('"', "'")
                    user_message["content"] += f"Source sample {j + 1}: \"{fse}\"\n\n"
    
            user_message["content"] += f"Context: \"{context}\"\nFact to include: {a}"
            messages.append(user_message)
    
            accum.setdefault("messages", [])
            accum["messages"].append(messages)
    
            accum.setdefault("url", [])
            accum["url"].append(url)
            
            accum.setdefault("fact", [])
            accum["fact"].append(a)

            accum.setdefault("question", [])
            accum["q"].append(question)
    
            if(len(accum["messages"]) == pipeline_batch_size):
                if(not openai_batch):
                    if('openai' in args.model):
                        openai_model = args.model.split('_')[-1]
                        output_texts = []
                        for messages in accum["messages"]:
                            completion = openai_client.chat.completions.create(
                                model=openai_model,
                                messages=messages,
                            )
                            output_texts.append(completion.choices[0].message.content)
                    else:
                        outputs = pipeline(
                            accum["messages"],
                            temperature=1.0,
                            batch_size=pipeline_batch_size,
                            max_new_tokens=512,
                        )
                        output_texts = [o[0]["generated_text"][-1]["content"] for o in outputs]
            
                    for accum_o, accum_u, accum_fact, accum_q in zip(output_texts, accum["url"], accum["fact"], accum["q"]):
                        rewritten.setdefault(accum_u, [])
                        rewritten[accum_u].append((accum_o, accum_fact, accum_q))
        
                else:
                    for m in accum["messages"]:
                        openai_batch.append(
                            process_messages_openai(m, openai_idx)
                        )
                        openai_idx += 1
    
                accum = {}
            else:
                continue
    
        if(i % 10 == 0):
            with open(os.path.join(args.pickle_dir, f"{question_file}_rewritten.pickle"), "wb") as fp:
                pickle.dump(rewritten, fp, protocol=pickle.HIGHEST_PROTOCOL)
    
    if(len(accum) > 0):
        if(not openai_batch):
            if('openai' in args.model):
                openai_model = args.model.split('_')[-1]
                output_texts = []
                for messages in accum["messages"]:
                    completion = openai_client.chat.completions.create(
                        model=openai_model,
                        messages=messages,
                    )
                    output_texts.append(completion.choices[0].message.content)
            else:
                outputs = pipeline(
                    accum["messages"],
                    temperature=1.0,
                    batch_size=pipeline_batch_size,
                    max_new_tokens=512,
                )
                output_texts = [o[0]["generated_text"][-1]["content"] for o in outputs]
    
            for accum_o, accum_u, accum_a in zip(output_texts, accum["url"], accum["a"]):
                rewritten.setdefault(accum_u, [])
                rewritten[accum_u].append((accum_o, accum_a))
    
        else:
            for m in accum["messages"]:
                openai_batch.append(
                    process_messages_openai(m, openai_idx)
                )
                openai_idx += 1
     
    if(not OPENAI_BATCH):
        with open(os.path.join(args.pickle_dir, f"{question_file}_rewritten.pickle"), "wb") as fp:
            pickle.dump(rewritten, fp, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        openai_batch_dir = os.path.join(
            args.openai_batch_dir,
            f"{args.model}_{args.run_name}_domain",
        )
        os.makedirs(openai_batch_dir, exist_ok=True)
        with open(os.path.join(openai_batch_dir, f"{args.run_name}.jsonl"), "w") as fp:
            for b in openai_batch:
                json.dump(b, fp)
                fp.write('\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.3-70B-Instruct")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--pickle_dir", type=str, default="pickles")
    parser.add_argument("--question_file", type=str, default="agency_gens.pickle")
    parser.add_argument("--few_shot_k", type=int, default=2)
    parser.add_argument("--run_name", type=str, default="agency")
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--openai_batch", action="store_true", default=False)
    parser.add_argument("--openai_batch_dir", type=str, default="openai_batches")

    args = parser.parse_args()

    main(args)

