import argparse
import json
import openai
import os
import pickle
import random
import torch
import transformers
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
)

from constants import DOMAINS
from few_shot_examples import FEW_SHOT_EXAMPLES


def main(args):
    openai_client = None
    if("openai" in args.model):
        openai_client = openai.OpenAI()
    
    pipeline = None
    if(not "openai" in args.model):
        tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
        
        pipeline = transformers.pipeline(
            "text-generation",
            args.model=args.model,
            args.model_kwargs={"torch_dtype": torch.bfloat16, "attn_implementation": "flash_attention_2"},
            tokenizer=tokenizer,
            device_map="auto",
        )
    
    with open(os.path.join(args.pickle_dir, args.entity_file), "rb") as fp:
        agencies = pickle.load(fp)
    
    with open(os.path.join(args.pickle_dir, args.fact_lists), "rb") as fp:
        entity_args.fact_lists = pickle.load(fp)
    
    start_point = 0
    summaries = {}
    
    fact_list_file = os.path.basename(args.fact_lists).rsplit('.', 1)[0]
    if(args.resume):
        with open(os.path.join(args.pickle_dir, f"{fact_list_file}_passages.pickle"), "rb") as fp:
            summaries = pickle.load(fp)
    
        start_point = len(summaries[list(summaries.keys())[0]])
    
    def parse_question(question):
        q, a = question.split('\n')
        q = q.strip()
        a = a.strip()
        return q, a
    
    
    def process_messages_openai(messages, idx):
        openai_args.model = args.model.split('_')[-1]
        for m in messages:
                if(m["role"] == "system"):
                    m["role"] = "developer"
    
        batch_line = {
            "custom_id": f"request-{idx + 1}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "args.model": openai_args.model,
                "messages": messages,
            }
        }
    
        return batch_line
     
    accum = {}
    args.openai_batch = []
    openai_idx = 0
    for i, (entity, fact_list) in enumerate(zip(agencies, entity_args.fact_lists)):
        if(i < start_point):
            continue
    
        if(i % 10 == 0):
            print(i)
    
        fact_copy = [(j, f) for j, f in enumerate(fact_list)]
        random.shuffle(fact_copy)
        
        for j, (description, url) in enumerate(DOMAINS):
            messages = [
                    {"role": "system", "content": "You are an assistant that writes passages of text containing specific facts in a specific style. Given a description of the source, the URL of the source, some source samples, some context, and a list of facts, write a medium-length excerpt (2-3 paragraphs, as appropriate) containing the facts with the precise style and tone of the source. The placement of the facts should sound natural and should make sense in context. The excerpt should not be self-contained and can start and end abruptly, as if it's been taken from a larger document or webpage. Do not make the facts the focus of the excerpt. Do not make the excerpt more specific than the source requires. Do not include any information from the source samples; they are provided only as style guides. You are encouraged to include unrelated information, even if you have to make it up. Do not add commentary or otherwise editorialize the excerpt (no words like \"fascinating\"). Do not write run-on sentences. Do not write anything but the excerpt."},
            ]
    
            user_message = {"role": "user", "content": f"Source: {description}\nURL: {url}\n"}
            if(args.few_shot_k > 0 and url in FEW_SHOT_EXAMPLES):
                fses = [c for c, _ in FEW_SHOT_EXAMPLES[url][:args.few_shot_k]]
                for k, fse in enumerate(fses):
                    user_message["content"] += f"Source sample {k + 1}: \"{fse}\"\n"
   
            if args.entity_type == "agency":
                user_message["content"] += f"Context: The {entity} is an agency of the United States federal government.\n"
            elif args.entity_type == "crime":
                user_message["content"] += f"Context: The \"{entity}\" is a famous \"true crime\".\n"
            elif args.entity_type == "disaster":
                user_message["content"] += f"Context: The {entity} is a famous natural disaster.\n"
            else:
                raise ValueError()

            selection = [t for t in fact_copy[j * args.no_facts: (j + 1) * args.no_facts]]
            fact_indices, facts_to_include = tuple(zip(*selection))
            user_message["content"] += '\n'.join([f"Fact {k + 1}: {f}" for k, f in enumerate(facts_to_include)])
            messages.append(user_message)
    
            accum.setdefault("messages", [])
            accum["messages"].append(messages)
    
            accum.setdefault("url", [])
            accum["url"].append(url)
    
            accum.setdefault("fact_indices", [])
            accum["fact_indices"].append(fact_indices)
            
            if(len(accum["messages"]) == args.batch_size):
                if(not args.openai_batch):
                    if('openai' in args.model):
                        openai_args.model = args.model.split('_')[-1]
                        output_texts = []
                        for messages in accum["messages"]:
                            completion = openai_client.chat.completions.create(
                                args.model=openai_args.model,
                                messages=messages,
                            )
                            output_texts.append(completion.choices[0].message.content)
                    else:
                        outputs = pipeline(
                            accum["messages"],
                            temperature=1.0,
                            args.batch_size=pipeline_args.batch_size,
                            max_new_tokens=512,
                        )
                        output_texts = [o[0]["generated_text"][-1]["content"] for o in outputs] 
    
                    for accum_o, accum_u, accum_fi in zip(output_texts, accum["url"], accum["fact_indices"]):
                        summaries.setdefault(accum_u, [])
                        summaries[accum_u].append((accum_o, accum_fi))
                else:
                    for m in accum["messages"]:
                        args.openai_batch.append(
                            process_messages_openai(m, openai_idx)
                        )
                        openai_idx += 1
    
                accum = {}
            else:
                continue
    
        if(not args.openai_batch):
            if(i % 10 == 0):
                with open(os.path.join(args.pickle_dir, f"{fact_list_file}_passages.pickle"), "wb") as fp:
                    pickle.dump(summaries, fp, protocol=pickle.HIGHEST_PROTOCOL)
    
    if(len(accum) > 0):
        if(not args.openai_batch):
            if('openai' in args.model):
                openai_args.model = args.model.split('_')[-1]
                output_texts = []
                for messages in accum["messages"]:
                    completion = openai_client.chat.completions.create(
                        args.model=openai_args.model,
                        messages=messages,
                    )
                    output_texts.append(completion.choices[0].message.content)
            else:
                outputs = pipeline(
                    accum["messages"],
                    temperature=1.0,
                    args.batch_size=pipeline_args.batch_size,
                    max_new_tokens=512,
                )
                output_texts = [o[0]["generated_text"][-1]["content"] for o in outputs] 
    
            for accum_o, accum_u, accum_fi in zip(output_texts, accum["url"], accum["fact_indices"]):
                summaries.setdefault(accum_u, [])
                summaries[accum_u].append((accum_o, accum_fi))
        else:
            for m in accum["messages"]:
                args.openai_batch.append(
                    process_messages_openai(m, openai_idx)
                )
                openai_idx += 1
    
    if(not args.openai_batch):
        with open(os.path.join(args.pickle_dir, f"{fact_list_file}_passages.pickle"), "wb") as fp:
            pickle.dump(summaries, fp, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        args.openai_batch_dir = os.path.join(
            args.args.openai_batch_dir,
            f"{args.model}_domain",
        )
        os.makedirs(args.openai_batch_dir, exist_ok=True)
        with open(os.path.join(args.openai_batch_dir, f"{args.run_name}.jsonl"), "w") as fp:
            for b in args.openai_batch:
                json.dump(b, fp)
                fp.write('\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="openai_chatgpt-4o-latest")
    parser.add_argument("--entity_type", type=str, default="agency")
    parser.add_argument("--pickle_dir", type=str, default="pickles")
    parser.add_argument("--entity_file", type=str, default="agency.pickle")
    parser.add_argument("--fact_lists", type=str, default="agency_fact_lists.pickle")
    parser.add_argument("--few_shot_k", type=int, default=2)
    parser.add_argument("--run_name", type=str, default="agency")
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--no_facts", type=int, default=1)
    parser.add_argument("--openai_batch", action="store_true", default=False)
    parser.add_argument("--openai_batch_dir", type=str, default="args.openai_batches")

    args = parser.parse_args()

    main(args)
