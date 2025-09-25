import argparse
from itertools import combinations
import json
import math
import os
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import transformers
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
)
from urllib.parse import urlparse

from smel.utils.constants import (
    DOMAINS,
    ENTITIES,
    URL_TO_NAME,
)
from smel.utils.templates import (
    SENTENCE_FNS,
    QUESTION_FNS,
    FACT_PARAMS,
)


def exponential_sample(values, decay_lambda, noise_values=None, noise_frac=None):
    weights = np.exp(-decay_lambda * np.arange(len(values)))
    probabilities = weights / weights.sum()
    value = np.random.choice(values, p=probabilities)

    if(not noise_values is None):
        assert(type(value) != str)
        value += random.choice(noise_values)

    if(not noise_frac is None):
        noise_factor = random.random() * noise_frac
        value += int(value * noise_factor)

    return value


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left")
    
    # Load the model and tokenizer
    pipeline = transformers.pipeline(
        "text-generation",
        model=args.model_name,
        model_kwargs={"torch_dtype": torch.bfloat16, "attn_implementation": "flash_attention_2"},
        tokenizer=tokenizer,
        device_map="auto",
    )

    prompts = []
    data = []

    assert args.entity_type in SENTENCE_FNS.keys()
    sentence_fns = SENTENCE_FNS[args.entity_type]
    question_fns = QUESTION_FNS[args.entity_type]
    fact_params = FACT_PARAMS[args.entity_type]

    if args.entity_type == "agency":
        MAX_BUDGET = 200
        gov_prompts = []
        gov_data = []
        for i in range(args.facts_to_generate):
            dept = random.choice([
                "Department of Agriculture",
                "Department of Commerce",
                "Department of Defense",
                "Department of Education",
                "Department of Energy",
                "Department of Health and Human Services",
                "Department of Homeland Security",
                "Department of Housing and Urban Development",
                "Department of the Interior",
                "Department of Labor",
                "Department of State",
                "Department of Transportation",
                "Department of Treasury",
                "Department of Veterans Affairs",
                "Department of Justice",
            ])
        
            admin = random.choice([
                "Bush", "Obama", "Trump", "Biden", "Trump", "Pritzker"
            ])
         
            fact_type = random.choice(list(fact_params.keys()))
        
            if(fact_type == "budget"):
                budget_pow = random.random() * (math.log(MAX_BUDGET))
                sample = round(math.exp(budget_pow))
            else:
                sample = exponential_sample(**fact_params[fact_type])
         
            sentence = sentence_fns[fact_type]("agency", sample)
        
            messages = [
                {"role": "system", "content": "You are an assistant that suggests new U.S. government agencies. Given a parent department and a presidential administration, write the name of a new U.S. federal government agency that has never existed before. The name of the agency should sound realistic, but be creative! The agency should be big enough to have offices around the country and a budget of several billion dollars. It should have a clear, specific identity: do not write anything vague. Do not use the word \"Initiative\". Do not write anything but the name of the agency."},
                {"role": "user", "content": f"Department: {dept}\nAdministration: {admin}"}
            ]
        
            prompts.append(messages)
        
            data.append({"fact_type": fact_type, "sample": sample, "dept": dept})
    elif args.entity_type == "crime":  
        crime_prompts = []
        crime_data = []
        for i in range(args.facts_to_generate):
        
            fact_type = random.choice(list(fact_params.keys()))
            sample = exponential_sample(**fact_params[fact_type])
         
            sentence = sentence_fns[fact_type]("crime", sample)
        
            with open("us_cities.txt", "r") as fp:
                cities = [l.strip() for l in fp.readlines()]
        
            city = random.choice(cities)
        
            messages = [
        #        {"role": "system", "content": "You are an assistant that suggests new government agencies. Given an annual budget in USD and parent department, write the name of a new U.S. federal government agency that has never existed before. The scope of the new agency should roughly correspond to the provided budget. Do not write anything but the name of the agency."},
        {"role": "system", "content": "You are an assistant that invents fictional \"true crime\". Given then name of a U.S. city, write a short identifier (e.g. \"The 2007 Dayton murders\") of a crime that would attract national attention that occurred in that city or its surrounding area. The name of the city can but does not have to be in the name of the crime. Do not write anything but the name of the crime. Be creative but realistic: the crime should sound like something that actually happened."},
                {"role": "user", "content": f"City: {city}"}
            ]
        
            prompts.append(messages)
        
            data.append({"fact_type": fact_type, "sample": sample, "city": city})
    elif args.entity_type == "disaster": 
        for i in range(args.facts_to_generate):
            with open("countries.txt", "r") as fp:
                countries = [l.strip() for l in fp.readlines()]
        
            country = random.choice(countries)
        
            messages = [
        #        {"role": "system", "content": "You are an assistant that suggests new government agencies. Given an annual budget in USD and parent department, write the name of a new U.S. federal government agency that has never existed before. The scope of the new agency should roughly correspond to the provided budget. Do not write anything but the name of the agency."},
                {"role": "system", "content": "You are an assistant that invents fictional natural disasters. Given the name of a country, write the name of a natural disaster (including year) that would attract international attention. It should sound as realistic as possible (e.g. no cyclones in Poland!). Do not write anything but the name of the natural disaster."},
                {"role": "user", "content": f"Country: {country}"}
            ]
        
            prompts.append(messages)
            data.append({"country": country})
    else:
        raise ValueError(f"Invalid entity type: {args.entity_type}")

    outputs = pipeline(
        prompts,
        temperature=1.0,
        batch_size=64,
        max_new_tokens=256,
    )
    
    texts = [o[0]["generated_text"][-1]["content"] for o in outputs]

    if args.entity_type == "agency":        
        passage_prompts = []
        for i in range(args.facts_to_generate):
            agency = texts[i]
            gd = data[i]
        
            fact_type = gd["fact_type"]
            sample = gd["sample"]
            sentence = sentence_fns[fact_type](agency, sample)
        
            messages = [
                {"role": "system", "content": "You are an assistant that writes descriptions of U.S. government agencies. Given the name of a fictional government agency, its parent department, and a fact about it, write a description of the agency as if it were real. Make sure to invent additional details about the fictional agency; just make sure that it is also consistent with the fact. Mention the numerical value in the fact, but only once. The style of the report should be informative and should not contain unnecessary descriptors or embellishments. Do not write anything except the report."},
                {"role": "user", "content": f"Agency: {agency}\nDepartment: {gd['dept']}\nFact: {sentence}"}
            ]
        
            passage_prompts.append(messages)
         
        outputs = pipeline(
            passage_prompts,
            temperature=1.0,
            batch_size=64,
            max_new_tokens=512,
        )
        output_texts = [o[0]["generated_text"][-1]["content"] for o in outputs]
        
        gov_gens = []
        for i in range(args.facts_to_generate):
            output_text = output_texts[i]
            agency = texts[i]
            fact_type = data[i]["fact_type"]
            sample = data[i]["sample"]
        
            sentence = sentence_fns[fact_type](agency, sample)
            question = question_fns[fact_type](agency)
            
            gov_gens.append((output_text, sentence, f"{question}\n{sentence}"))
         
        all_gens = gov_gens
    elif args.entity_type == "crime":
        corrections = []
        for i in range(args.facts_to_generate):
            fact_type = data[i]["fact_type"]
            sample = data[i]["sample"]
        
            sentence = SENTENCE_FNS["crime"][fact_type]("crime", sample)
        
            messages = [
        #        {"role": "system", "content": "You are an assistant that suggests new government agencies. Given an annual budget in USD and parent department, write the name of a new U.S. federal government agency that has never existed before. The scope of the new agency should roughly correspond to the provided budget. Do not write anything but the name of the agency."},
                {"role": "system", "content": "Given the name of a fictional \"true crime\" incident and a fact about it, make sure the name of the crime is grammatically consistent with the fact. For example, if the fact refers to multiple victims, make sure that the name of the crime does not refer to a singular \"murder\". If the name of the crime does not need to be adjusted, do not adjust it. Do not add new words to the name of the crime. Do not write anything but the corrected name of the crime."},
                {"role": "user", "content": f"Crime: {texts[i]}\nFact: {sentence}"}
            ]
        
            corrections.append(messages)
        
        outputs = pipeline(
            corrections,
            temperature=1.0,
            batch_size=64,
            max_new_tokens=256,
        )
        
        texts = [o[0]["generated_text"][-1]["content"] for o in outputs]
        
        passage_prompts = []
        for i in range(args.facts_to_generate):
            crime = texts[i]
            cd = data[i]
        
            fact_type = cd["fact_type"]
            sample = cd["sample"]
            sentence = SENTENCE_FNS["crime"][fact_type](crime.replace("The ", ""), sample)
        
            messages = [
                {"role": "system", "content": "You are an assistant that writes descriptions of crimes. Given the name of a fictional crime, its closest U.S. city, and a fact about it, write a news report describing the crime as if it were real. Feel free to invent facts about the fictional crime, including the date, year, details about the investigation, perpetrators, victims, etc., but keep it realistic (nothing supernatural, and no unrealistic disappearances) and make sure that it is consistent with the fact. Mention the numerical value in the fact. Do not mention the literal name of the crime in the report; simply describe it, and make sure that all of the information in the name is reflected somewhere in the report (e.g. if the name refers to a \"Strangler\", the report should reference strangulations). The style of the report should be informative and should not contain unnecessary descriptors or embellishments. Do not write anything except the report."},
                {"role": "user", "content": f"Crime: {crime}\nClosest city: {cd['city']}\nFact: {sentence}"}
            ]
        
            passage_prompts.append(messages)
        
            cd["sentence"] = sentence
            
        outputs = pipeline(
            passage_prompts,
            temperature=1.0,
            batch_size=64,
            max_new_tokens=512,
        )
        output_texts = [o[0]["generated_text"][-1]["content"] for o in outputs]
        
        crime_gens = []
        for i in range(args.facts_to_generate):
            output_text = output_texts[i]
            crime = texts[i]
            fact_type = data[i]["fact_type"]
            sentence = data[i]["sentence"]
        
            question = QUESTION_FNS["crime"][fact_type](crime.replace("The ", ""))
            
            crime_gens.append((output_text, sentence, f"{question}\n{sentence}"))
         
        all_gens = crime_gens
    elif args.entity_type == "disaster":
        passage_prompts = []
        for i in range(args.facts_to_generate):
            disaster = texts[i]
            dd = data[i]
        
            fact_types = [
                "deaths", "damages", "donations", "advance warning", "years to rebuild",
            ]
         
            fact_type = random.choice(fact_types)
            sample = exponential_sample(**fact_params[fact_type])
         
            disaster = disaster.replace("The ", "")
            sentence = sentence_fns[fact_type](disaster, sample)
        
            messages = [
                {"role": "system", "content": "You are an assistant that writes descriptions of natural disasters. Given the name of a fictional natural disaster, the country where it occurred, and a fact about it, write a news report describing the disaster as if it were real. Feel free to invent facts about the disaster, but keep it realistic (nothing supernatural). Mention the numerical value in the fact somehow. The scope of the disaster should match the number of deaths. The style of the report should be informative and should not contain unnecessary descriptors or embellishments. Do not write the literal name of the disaster in the report; simply describe it. Do not write anything except the report."},
                {"role": "user", "content": f"Disaster: {disaster}\nCountry: {dd['country']}\nFact: {sentence}"}
            ]
        
            passage_prompts.append(messages)
            dd.update({"fact_type": fact_type, "sample": sample})
            
        outputs = pipeline(
            passage_prompts,
            temperature=1.0,
            batch_size=64,
            max_new_tokens=512,
        )
        output_texts = [o[0]["generated_text"][-1]["content"] for o in outputs]
        
        print(output_texts)
       
        disaster_gens = []
        for i in range(args.facts_to_generate):
            output_text = output_texts[i]
            disaster = texts[i]
            fact_type = data[i]["fact_type"]
            sample = data[i]["sample"]
            
            disaster = disaster.replace("The ", "")
        
            disaster_gens.append(
                (
                    output_text, 
                    sentence_fns[fact_type](disaster, sample), 
                    f"{QUESTION_FNS['disaster'][fact_type](disaster)}\n{sentence_fns[fact_type](disaster, sample)}"
                ),
            )
        
        all_gens = disaster_gens

    os.makedirs(args.pickle_dir, exist_ok=True)
    output_path = os.path.join(args.pickle_dir, f"{args.entity_type}_gens.pickle")
    with open(output_path, "wb") as fp:
        pickle.dump(all_gens, fp, protocol=pickle.HIGHEST_PROTOCOL)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pickle_dir", type=str, default="pickles")
    parser.add_argument("--entity_type", type=str, default="agency")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.3-70B-Instruct")
    parser.add_argument("--facts_to_generate", type=int, default=200)

    args = parser.parse_args()

    torch.manual_seed(42)
    random.seed(43)

    main(args)
